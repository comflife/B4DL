"""Qwen3.5 + joint VoxelNeXt LiDAR encoder spliced into input_ids.

Forward / generate behaviour:
  - input_ids contains exactly one IMAGE token per sample (a registered
    additional special token, default "<image>").
  - `points` is a list of (N_i, 5) raw LiDAR tensors (one per sample). The
    in-model VoxelNeXt encoder turns each into a (K, 128) feature blob with
    per-token (x, y, z). For backwards compat `images` is still accepted
    when the caller pre-extracted features.
  - At every IMAGE-token position we replace the single token with the
    output of the linear projector applied to the K-token feature blob,
    so a single id expands into K hidden-dim vectors.
  - attention_mask and labels are extended to cover the new vectors;
    inserted positions in `labels` get -100 (no loss).

The VoxelNeXt encoder is loaded via `init_voxelnext()` (called by the
trainer) and is frozen by default — only `mm_projector` + `pos_projector`
+ the LLM update during training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class MMQwenConfig:
    """Bundle of options the trainer needs to pass into the model wrapper."""

    mm_input_dim: int = 128  # VoxelNeXt backbone output channels
    image_token_id: Optional[int] = None  # set after tokenizer registration


class MMQwen(Qwen3_5ForCausalLM):
    """Qwen3.5 text-only LLM + LiDAR projector (LLaVA-style splicing)."""

    config_class = Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        # mm_input_dim and image_token_id are populated by the trainer after
        # construction — defaults keep loading from a checkpoint cheap.
        self.mm_input_dim = getattr(config, "mm_input_dim", 128)
        self.image_token_id: Optional[int] = getattr(config, "image_token_id", None)
        self.mm_projector = nn.Linear(self.mm_input_dim, config.hidden_size)

        # Positional projector for VoxelNeXt features (each token carries
        # an (x, y, z) location). When `mm_pos_dim > 0` each token embedding
        # becomes `mm_projector(feat) + pos_projector(xyz_norm)`, giving the
        # LLM an explicit spatial signal alongside the content vector.
        self.mm_pos_dim: int = int(getattr(config, "mm_pos_dim", 0))
        if self.mm_pos_dim > 0:
            self.pos_projector = nn.Linear(self.mm_pos_dim, config.hidden_size)
        else:
            self.pos_projector = None
        # Hard-coded normalisation range for xyz inputs. nuScenes VoxelNeXt
        # uses [-54, 54] xy and [-5, 3] z; we normalise to [-1, 1] for both
        # projector stability and so the model isn't forced to learn the
        # absolute scale at the same time as semantics.
        self.register_buffer(
            "_xyz_norm_scale",
            torch.tensor(getattr(config, "mm_xyz_scale", [54.0, 54.0, 5.0]), dtype=torch.float32),
            persistent=False,
        )

        # Joint VoxelNeXt encoder is attached lazily by `init_voxelnext`.
        self.voxelnext = None

    # ------------------------------------------------------------------
    # Joint VoxelNeXt encoder (frozen by default).
    # ------------------------------------------------------------------
    def init_voxelnext(
        self,
        voxelnext_root: str,
        ckpt_path: str,
        cfg_file=None,
        top_k: int = 256,
        freeze: bool = True,
        device=None,
    ):
        """Attach a pcdet VoxelNeXt encoder. Requires pcdet to be importable
        (run `pip install -e .` in the cloned VoxelNeXt repo first)."""
        from .voxelnext_encoder import VoxelNeXtEncoder

        dev = str(device) if device is not None else (
            f"cuda:{self.device.index}" if (self.device.type == "cuda") else "cpu"
        )
        self.voxelnext = VoxelNeXtEncoder(
            voxelnext_root=voxelnext_root,
            ckpt_path=ckpt_path,
            cfg_file=cfg_file,
            top_k=top_k,
            freeze=freeze,
            device=dev,
        )

    # ------------------------------------------------------------------
    # Setters used by trainer scripts after tokenizer surgery.
    # ------------------------------------------------------------------
    def set_image_token_id(self, tid: int):
        self.image_token_id = int(tid)
        # Also persist on the Hf config so save/load round-trips correctly.
        self.config.image_token_id = int(tid)
        self.config.mm_input_dim = self.mm_input_dim

    def get_mm_projector(self) -> nn.Module:
        return self.mm_projector

    # ------------------------------------------------------------------
    # Splicing helper.
    # ------------------------------------------------------------------
    def _project_images(
        self, images
    ) -> List[torch.Tensor]:
        """Project per-sample VoxelNeXt feature blobs into the LM hidden
        space. `images` is a list of dicts:
            {"feat": (K, mm_input_dim), "xyz": (K, 3), "mask": (K,) bool}
        Each token gets `mm_projector(feat) + pos_projector(xyz_norm)`,
        where `mask=False` rows are dropped before splicing so the LLM
        only attends to real voxels.
        """
        target_dtype = self.mm_projector.weight.dtype
        if not (isinstance(images, (list, tuple)) and len(images) > 0
                and isinstance(images[0], dict)):
            raise TypeError(
                "MMQwen now expects `images` to be a list of "
                "{feat, xyz, mask} dicts (VoxelNeXt blobs). Pass raw points "
                "via `points=` instead, or pre-extract VoxelNeXt features."
            )

        out = []
        for blob in images:
            feat = blob["feat"].to(target_dtype).to(self.device)
            xyz = blob["xyz"].to(target_dtype).to(self.device)
            mask = blob.get("mask")
            if mask is not None:
                mask = mask.to(self.device).bool()
                feat = feat[mask]
                xyz = xyz[mask]
            tok = self.mm_projector(feat)
            if self.pos_projector is not None and xyz.shape[0] > 0:
                scale = self._xyz_norm_scale.to(target_dtype).to(self.device)
                xyz_n = (xyz / scale).clamp(-1.5, 1.5)
                tok = tok + self.pos_projector(xyz_n)
            out.append(tok)
        return out

    def _splice(
        self,
        input_ids: torch.Tensor,
        images: Union[List[torch.Tensor], torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ):
        """Build (inputs_embeds, attention_mask, labels) by replacing every
        IMAGE-token position with that sample's projected LiDAR sequence."""
        if self.image_token_id is None:
            raise RuntimeError(
                "MMQwen.image_token_id is not set. Call set_image_token_id() "
                "after registering '<image>' in the tokenizer."
            )

        proj = self._project_images(images)
        embed = self.get_input_embeddings()
        bsz = input_ids.shape[0]
        device = input_ids.device

        new_embeds: List[torch.Tensor] = []
        new_masks: List[torch.Tensor] = []
        new_labels: List[Optional[torch.Tensor]] = []

        for i in range(bsz):
            ids = input_ids[i]
            mask = attention_mask[i] if attention_mask is not None else torch.ones_like(ids)
            lbl = labels[i] if labels is not None else None

            # Drop right-padding before splicing so we don't carry zeros around.
            keep = mask.bool()
            ids_v = ids[keep]
            lbl_v = lbl[keep] if lbl is not None else None

            emb = embed(ids_v)  # (T, H)
            img = proj[i].to(emb.dtype)  # (N_lidar, H)

            pos = (ids_v == self.image_token_id).nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                # Sample has no image placeholder — pass through unchanged.
                new_embeds.append(emb)
                new_masks.append(torch.ones(emb.shape[0], dtype=torch.long, device=device))
                new_labels.append(lbl_v)
                continue

            p = int(pos[0])  # use first image token; we expect exactly one
            before_emb = emb[:p]
            after_emb = emb[p + 1 :]
            merged_emb = torch.cat([before_emb, img, after_emb], dim=0)

            if lbl_v is not None:
                lbl_pad = torch.full(
                    (img.shape[0],), -100, dtype=lbl_v.dtype, device=device
                )
                merged_lbl = torch.cat([lbl_v[:p], lbl_pad, lbl_v[p + 1 :]], dim=0)
            else:
                merged_lbl = None

            new_embeds.append(merged_emb)
            new_masks.append(
                torch.ones(merged_emb.shape[0], dtype=torch.long, device=device)
            )
            new_labels.append(merged_lbl)

        max_len = max(e.shape[0] for e in new_embeds)
        hid = new_embeds[0].shape[1]
        out_dtype = new_embeds[0].dtype

        padded_emb = torch.zeros(bsz, max_len, hid, dtype=out_dtype, device=device)
        padded_mask = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
        padded_lbl = (
            torch.full((bsz, max_len), -100, dtype=torch.long, device=device)
            if labels is not None
            else None
        )
        for i in range(bsz):
            L = new_embeds[i].shape[0]
            padded_emb[i, :L] = new_embeds[i]
            padded_mask[i, :L] = new_masks[i]
            if padded_lbl is not None and new_labels[i] is not None:
                padded_lbl[i, :L] = new_labels[i]
        return padded_emb, padded_mask, padded_lbl

    # ------------------------------------------------------------------
    # forward / generate overrides.
    # ------------------------------------------------------------------
    def _encode_points_to_blobs(self, points: List[torch.Tensor]):
        """Run the (typically frozen) VoxelNeXt encoder on raw points."""
        if self.voxelnext is None:
            raise RuntimeError(
                "MMQwen.voxelnext is not initialised. Call init_voxelnext() "
                "before forwarding raw points, or pass pre-extracted blobs "
                "via `images=` instead."
            )
        return self.voxelnext(points)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        points: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Joint path: raw points -> VoxelNeXt -> blob list, then splice.
        if (
            points is not None
            and images is None
            and inputs_embeds is None
            and input_ids is not None
        ):
            images = self._encode_points_to_blobs(points)

        if images is not None and inputs_embeds is None and input_ids is not None:
            inputs_embeds, attention_mask, labels = self._splice(
                input_ids, images, attention_mask, labels
            )
            input_ids = None
            position_ids = None  # let Qwen recompute from attention_mask
        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        points: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ):
        if points is not None and images is None and input_ids is not None:
            images = self._encode_points_to_blobs(points)
        if images is not None and input_ids is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            inputs_embeds, attention_mask, _ = self._splice(
                input_ids, images, attention_mask, None
            )
            return super().generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )
        return super().generate(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

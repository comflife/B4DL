"""Qwen3 with a 3D-AVS SMAP LiDAR projector spliced into input_ids.

Forward / generate behaviour:
  - input_ids contains exactly one IMAGE token per sample (a registered
    additional special token, default "<image>").
  - `images` is either a list of (N, smap_dim) tensors (one per sample) or
    a stacked (B, N, smap_dim) tensor.
  - At every IMAGE-token position we replace the single token with the
    output of the linear projector applied to the corresponding sample's
    LiDAR feature, so a single id expands into N hidden-dim vectors.
  - attention_mask and labels are extended to cover the new vectors;
    inserted positions in `labels` get -100 (no loss).

Training only updates `mm_projector` when the caller freezes the rest of
the model (warmup phase); under full fine-tuning we leave everything
trainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class MMQwenConfig:
    """Bundle of options the trainer needs to pass into the model wrapper."""

    mm_input_dim: int = 512  # SMAP output dim
    image_token_id: Optional[int] = None  # set after tokenizer registration


class MMQwen(Qwen3ForCausalLM):
    """Qwen3ForCausalLM + LiDAR projector (LLaVA-style splicing)."""

    config_class = Qwen3Config

    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        # mm_input_dim and image_token_id are populated by the trainer after
        # construction — defaults keep loading from a checkpoint cheap.
        self.mm_input_dim = getattr(config, "mm_input_dim", 512)
        self.image_token_id: Optional[int] = getattr(config, "image_token_id", None)
        self.mm_projector = nn.Linear(self.mm_input_dim, config.hidden_size)

        # Optional positional projector for VoxelNeXt-style features that
        # carry a per-token (x, y, z) location. When `mm_pos_dim > 0` each
        # token embedding becomes `mm_projector(feat) + pos_projector(xyz_norm)`,
        # giving the LLM an explicit spatial signal alongside the content
        # vector. SMAP-style features (no per-token xyz) leave this disabled.
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

        # Q3D coord-vocab metadata. `set_coord_vocab` populates these after
        # tokenizer surgery; defaults keep checkpoint loading cheap.
        self.coord_token_min: Optional[int] = getattr(config, "coord_token_min", None)
        self.coord_token_max: Optional[int] = getattr(config, "coord_token_max", None)
        self.box_start_id: Optional[int] = getattr(config, "box_start_id", None)
        self.box_end_id: Optional[int] = getattr(config, "box_end_id", None)
        # Weight of the per-axis L1 expected-coord auxiliary loss. The trainer
        # overwrites this from --coord_aux_weight at startup.
        self.coord_aux_weight: float = float(getattr(config, "coord_aux_weight", 0.0))
        # The bin_value table is static (8, 1024) — registered as a buffer so
        # it follows .to(device)/.to(dtype) transparently. We initialise from
        # the quantizer here so a model loaded via from_pretrained() already
        # has the table populated; the trainer can still override via
        # `set_coord_vocab` if a future iteration changes the codebook layout.
        try:
            from .quantizer import bin_value_table as _bvt
            init = torch.as_tensor(_bvt(), dtype=torch.float32)
        except Exception:
            init = torch.zeros(8, 1024)
        self.register_buffer("bin_value_table", init, persistent=False)

    # ------------------------------------------------------------------
    # Setters used by trainer scripts after tokenizer surgery.
    # ------------------------------------------------------------------
    def set_image_token_id(self, tid: int):
        self.image_token_id = int(tid)
        # Also persist on the Hf config so save/load round-trips correctly.
        self.config.image_token_id = int(tid)
        self.config.mm_input_dim = self.mm_input_dim

    def set_coord_vocab(
        self,
        coord_token_min: int,
        coord_token_max: int,
        bin_value_table,
        box_start_id: int,
        box_end_id: int,
    ):
        """Wire the contiguous Q3D coord-token range, the (8, 1024) bin-value
        table, and the box-marker ids onto the model. The trainer calls this
        once after tokenizer surgery."""
        self.coord_token_min = int(coord_token_min)
        self.coord_token_max = int(coord_token_max)
        self.box_start_id = int(box_start_id)
        self.box_end_id = int(box_end_id)
        # bin_value_table: numpy or torch, shape (8, 1024). We keep it as a
        # buffer in fp32; the aux-loss path casts to logits' dtype.
        if not isinstance(bin_value_table, torch.Tensor):
            bin_value_table = torch.as_tensor(bin_value_table, dtype=torch.float32)
        if bin_value_table.shape != (8, 1024):
            raise ValueError(f"bin_value_table must be (8, 1024); got {tuple(bin_value_table.shape)}")
        self.bin_value_table = bin_value_table.to(dtype=torch.float32)
        # Persist range/box ids on config so save/load round-trips them.
        self.config.coord_token_min = self.coord_token_min
        self.config.coord_token_max = self.coord_token_max
        self.config.box_start_id = self.box_start_id
        self.config.box_end_id = self.box_end_id

    def get_mm_projector(self) -> nn.Module:
        return self.mm_projector

    # ------------------------------------------------------------------
    # Q3D auxiliary loss: CE on coord tokens is already covered by the
    # standard LM loss; here we add a *differentiable expected-coord* L1
    # term that gives the model direct distance gradient instead of the
    # all-or-nothing token-level CE.
    # ------------------------------------------------------------------
    def _build_axis_ids(self, labels: torch.Tensor) -> Optional[torch.Tensor]:
        """Per-position axis id (0..7) for tokens that belong to a Q3D box;
        -1 elsewhere. Same shape as `labels`. Returns None if no box markers
        are present, so the caller can short-circuit cheaply.

        We rely on the dataset emitting the canonical 8-token layout
        `<|box_start|><X><Y><Z><W><L><H><SIN><COS><|box_end|>`. For each
        `<|box_start|>` position we walk forward up to 8 coord tokens, tag
        them with axis 0..7, and stop at the first non-coord (which should
        be `<|box_end|>`). Anything else is left at -1.

        This is O(B * #boxes_per_sample) — for det_area that's ~5 per sample
        and the loop overhead is negligible relative to forward pass.
        """
        if self.box_start_id is None or self.coord_token_min is None:
            return None
        starts = (labels == self.box_start_id).nonzero(as_tuple=False)
        if starts.numel() == 0:
            return None

        axis_ids = torch.full_like(labels, -1)
        T = labels.shape[1]
        cmin = self.coord_token_min
        cmax = self.coord_token_max
        # Operate on CPU lists for control flow — much simpler than vector
        # gather and the count is tiny.
        starts_cpu = starts.tolist()
        labels_cpu = labels.tolist()
        for b, s in starts_cpu:
            row = labels_cpu[b]
            for k in range(8):
                pos = s + 1 + k
                if pos >= T:
                    break
                tok = row[pos]
                if cmin <= tok <= cmax:
                    axis_ids[b, pos] = k
                else:
                    break
        return axis_ids

    def _coord_aux_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """L1 loss between the per-axis *expected* coord (softmax-weighted
        sum of bin values) and the GT bin's value. Differentiable with
        respect to the coord-vocab logits, complementing standard CE."""
        axis_ids = self._build_axis_ids(labels)
        if axis_ids is None:
            return None

        # Standard causal-LM shift: logits[:, :-1] predict labels[:, 1:].
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_axis = axis_ids[:, 1:]

        coord_mask = shift_axis >= 0
        if not coord_mask.any():
            return None

        valid_logits = shift_logits[coord_mask]               # (N, V)
        valid_labels = shift_labels[coord_mask]               # (N,)
        valid_axis = shift_axis[coord_mask]                   # (N,)

        # Slice to the contiguous coord-vocab range.
        cmin = self.coord_token_min
        cmax = self.coord_token_max
        coord_logits = valid_logits[:, cmin : cmax + 1]       # (N, 1024)
        # Float32 for stable softmax even when logits are bf16.
        probs = torch.softmax(coord_logits.float(), dim=-1)   # (N, 1024)

        bins = self.bin_value_table.to(probs.device).float()   # (8, 1024)
        bv = bins[valid_axis]                                  # (N, 1024)
        expected = (probs * bv).sum(dim=-1)                    # (N,)
        target = bins[valid_axis, valid_labels - cmin]         # (N,)

        # Per-axis weighting: xy/z get unit weight; sizes (log-space) and
        # sin/cos (unit interval) are inherently smaller-scale, so we scale
        # them up so they contribute on the same order as the centre L1.
        # (Sizes range up to ~20m, sin/cos in [-1, 1] — without scaling, the
        # centre L1 would dominate even though all axes matter for IoU.)
        per_axis_scale = torch.tensor(
            [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0],
            dtype=probs.dtype,
            device=probs.device,
        )
        scale = per_axis_scale[valid_axis]
        l1 = (torch.abs(expected - target) * scale).mean()
        return l1

    # ------------------------------------------------------------------
    # Splicing helper.
    # ------------------------------------------------------------------
    def _project_images(
        self, images
    ) -> List[torch.Tensor]:
        """Project per-sample LiDAR features into the LM hidden space.

        Two input layouts are supported:

        1. **SMAP-style** — `images` is a list/tensor of `(N, mm_input_dim)`
           feature vectors (one per "view"). Each becomes a row that we
           pass through `mm_projector`. Backwards compatible with the
           pre-VoxelNeXt pipeline.
        2. **VoxelNeXt-style** — `images` is a list of dicts:
               {"feat": (K, mm_input_dim), "xyz": (K, 3), "mask": (K,) bool}
           In this layout each token gets `mm_projector(feat) +
           pos_projector(xyz_norm)`, where `mask=False` rows are dropped
           before splicing so the LLM only attends to *real* voxels.
           This is the spatial-aware path; enabled when `mm_pos_dim > 0`
           and the entries are dicts with a `feat` key.
        """
        target_dtype = self.mm_projector.weight.dtype

        # VoxelNeXt-style: list of dicts per-sample.
        if isinstance(images, (list, tuple)) and len(images) > 0 and isinstance(images[0], dict):
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

        # SMAP-style fallback (original pre-VoxelNeXt behaviour).
        if isinstance(images, (list, tuple)):
            return [
                self.mm_projector(img.to(target_dtype).to(self.device))
                for img in images
            ]
        if images.dim() == 2:
            images = images.unsqueeze(0)
        x = self.mm_projector(images.to(target_dtype).to(self.device))
        return [x[i] for i in range(x.shape[0])]

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
        **kwargs,
    ) -> CausalLMOutputWithPast:
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

        # Per-axis L1 aux loss on Q3D coord positions. Skipped during
        # generation (labels=None) and when the weight is 0.
        if (
            labels is not None
            and self.coord_aux_weight > 0.0
            and self.coord_token_min is not None
            and out.loss is not None
        ):
            aux = self._coord_aux_loss(out.logits, labels)
            if aux is not None:
                out.loss = out.loss + self.coord_aux_weight * aux
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
        **kwargs,
    ):
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

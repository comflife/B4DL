"""SFT trainer for Qwen3-0.6B + 3D-AVS SMAP LiDAR features.

Two phases share this entry point:

  --tune_mm_only True   Stage 1a: freeze the LLM, train only `mm_projector`
                        and the embedding rows for the newly-added special
                        tokens. High LR, fast convergence.
  --tune_mm_only False  Stage 1b: full-parameter fine-tune of LLM + projector,
                        loading the warmed-up projector via
                        --pretrain_mm_projector.

Both phases use the same dataset (combined nuCaption + nuGrounding) and the
same Qwen ChatML template (built into the tokenizer). User-side tokens are
masked with -100 so we only get loss on the assistant response.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

NEW_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NEW_CODE_DIR))

from qwen_mm import (  # noqa: E402
    BOX_END,
    BOX_START,
    Collator,
    IMAGE_PLACEHOLDER,
    LiDARMMDataset,
    MMQwen,
)


# ---------------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3.5-9B")
    mm_input_dim: int = field(default=128)
    mm_pos_dim: int = field(
        default=3,
        metadata={
            "help": "Positional projector dim for VoxelNeXt token (x,y,z). "
            "3 is the only sane value; left configurable for ablations."
        },
    )
    voxelnext_root: str = field(default=None)
    voxelnext_ckpt: str = field(default=None)
    voxelnext_top_k: int = field(default=256)
    voxelnext_freeze: bool = field(default=True)
    pretrain_mm_projector: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a stage-1a mm_projector.bin to load before training."},
    )
    tune_mm_only: bool = field(
        default=False,
        metadata={"help": "If true, freeze the LLM and only train the projector + new token rows."},
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    nuscenes_root: str = field(default=None)
    nuscenes_version: str = field(default="v1.0-trainval")
    n_sweeps: int = field(default=10)
    max_length: int = field(default=2048)


# ---------------------------------------------------------------------------
def add_special_tokens_and_resize(tokenizer, model: MMQwen):
    """Register <image>, <|box_start|>, <|box_end|>, and <0>..<999> as
    additional special tokens for 0-999 coordinate prediction.
    """
    # 1,000 coordinate tokens: <0> .. <999>
    coord_names = [f"<{i}>" for i in range(1000)]
    extras = [IMAGE_PLACEHOLDER, BOX_START, BOX_END] + coord_names
    needed = [t for t in extras if t not in tokenizer.get_vocab()]
    added = 0
    if needed:
        added = tokenizer.add_special_tokens({"additional_special_tokens": needed})
    if added:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        with torch.no_grad():
            in_emb = model.get_input_embeddings().weight
            out_emb = model.get_output_embeddings().weight
            mean_in = in_emb[:-added].mean(dim=0, keepdim=True)
            mean_out = out_emb[:-added].mean(dim=0, keepdim=True)
            in_emb[-added:] = mean_in
            out_emb[-added:] = mean_out

    image_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    model.set_image_token_id(image_id)
    return added


def freeze_llm_except_projector(model: MMQwen, num_added_tokens: int):
    """Stage 1a: freeze everything except `mm_projector` (and `pos_projector`
    when configured). Embedding/lm_head rows for the new special tokens are
    left frozen — same trade-off as the original LLaVA stage-1 recipe.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mm_projector.parameters():
        p.requires_grad = True
    if getattr(model, "pos_projector", None) is not None:
        for p in model.pos_projector.parameters():
            p.requires_grad = True


def load_mm_projector(model: MMQwen, path: str):
    """Load both the feature projector and (when present) the positional
    projector from a stage-1a checkpoint. The on-disk format may be:
      - raw `mm_projector.state_dict()` only (legacy SMAP)
      - `{ "mm_projector.weight": ..., "mm_projector.bias": ...,
           "pos_projector.weight": ..., "pos_projector.bias": ... }`
    We tolerate either layout."""
    sd = torch.load(path, map_location="cpu")

    feat_sd = {}
    pos_sd = {}
    legacy = True
    for k, v in sd.items():
        if k.startswith("mm_projector."):
            feat_sd[k.split("mm_projector.", 1)[1]] = v
            legacy = False
        elif k.startswith("pos_projector."):
            pos_sd[k.split("pos_projector.", 1)[1]] = v
            legacy = False
    if legacy:
        feat_sd = sd

    feat_missing, feat_unexpected = model.mm_projector.load_state_dict(feat_sd, strict=True)
    print(f"[sft] loaded mm_projector from {path} "
          f"(missing={feat_missing}, unexpected={feat_unexpected})")
    if pos_sd:
        if model.pos_projector is None:
            print(f"[sft] WARNING: ckpt has pos_projector but model.mm_pos_dim=0 — skipping")
        else:
            pmiss, punexp = model.pos_projector.load_state_dict(pos_sd, strict=True)
            print(f"[sft] loaded pos_projector (missing={pmiss}, unexpected={punexp})")


def save_mm_projector(model: MMQwen, output_dir: str):
    """Save both projectors with `mm_projector.X` / `pos_projector.X` keys
    so `load_mm_projector` can restore them in any later stage."""
    out = {}
    for k, v in model.mm_projector.state_dict().items():
        out[f"mm_projector.{k}"] = v
    if model.pos_projector is not None:
        for k, v in model.pos_projector.state_dict().items():
            out[f"pos_projector.{k}"] = v
    torch.save(out, Path(output_dir) / "mm_projector.bin")


# ---------------------------------------------------------------------------
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.local_rank in (-1, 0):
        print("[sft] model_args:", model_args)
        print("[sft] data_args :", data_args)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if tokenizer.pad_token is None:
        # Qwen tokenisers usually have <|endoftext|>; fall back if not.
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = MMQwen.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.mm_input_dim = model_args.mm_input_dim
    # NOTE: keep projectors in fp32 — tiny module, bf16 grad underflow is
    # a common source of nan when only the projector trains (stage 1a).
    # Also work around a transformers 5.8.0 bug: when from_pretrained is called
    # with torch_dtype=torch.bfloat16, parameters that are NOT present in the
    # checkpoint (like our custom mm_projector) get corrupted with NaN/garbage
    # values. We always rebuild the projector here so it has clean init.
    model.mm_projector = nn.Linear(
        model_args.mm_input_dim, model.config.hidden_size
    ).to(torch.float32)
    model.config.mm_input_dim = model_args.mm_input_dim

    # Positional projector (VoxelNeXt path). Build if requested even when
    # not present in the loaded checkpoint, and rebuild if dim changed.
    model.mm_pos_dim = int(model_args.mm_pos_dim)
    model.config.mm_pos_dim = model.mm_pos_dim
    if model.mm_pos_dim > 0:
        need_new = (
            model.pos_projector is None
            or model.pos_projector.in_features != model.mm_pos_dim
        )
        if need_new:
            model.pos_projector = nn.Linear(
                model.mm_pos_dim, model.config.hidden_size
            ).to(torch.float32)
        elif model.pos_projector is not None:
            model.pos_projector = model.pos_projector.to(torch.float32)
    else:
        model.pos_projector = None

    n_added = add_special_tokens_and_resize(tokenizer, model)
    if training_args.local_rank in (-1, 0):
        print(
            f"[sft] added {n_added} special tokens; image_id={model.image_token_id} "
            f"coord_tokens=<0>..<999>"
        )

    if model_args.pretrain_mm_projector:
        load_mm_projector(model, model_args.pretrain_mm_projector)

    # Joint VoxelNeXt encoder. Loaded after the LLM so it lives on the same
    # CUDA device. Frozen by default; gradient never flows through it.
    if model_args.voxelnext_root and model_args.voxelnext_ckpt:
        if training_args.local_rank in (-1, 0):
            print(
                f"[sft] init VoxelNeXt: root={model_args.voxelnext_root} "
                f"ckpt={model_args.voxelnext_ckpt} top_k={model_args.voxelnext_top_k} "
                f"freeze={model_args.voxelnext_freeze}"
            )
        model.init_voxelnext(
            voxelnext_root=model_args.voxelnext_root,
            ckpt_path=model_args.voxelnext_ckpt,
            top_k=model_args.voxelnext_top_k,
            freeze=model_args.voxelnext_freeze,
        )

    if model_args.tune_mm_only:
        freeze_llm_except_projector(model, n_added)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        if training_args.local_rank in (-1, 0):
            print(f"[sft] tune_mm_only: trainable={n_trainable}/{n_total} "
                  f"({100*n_trainable/n_total:.4f}%)")

    # Dataset
    train_ds = LiDARMMDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        nuscenes_root=data_args.nuscenes_root,
        nuscenes_version=data_args.nuscenes_version,
        n_sweeps=data_args.n_sweeps,
        max_length=data_args.max_length,
    )
    collator = Collator(pad_token_id=tokenizer.pad_token_id)

    class _DebugTrainer(Trainer):
        """Temporary wrapper to surface why loss becomes 0 / grad_norm nan."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            # ---- nan hunt: VoxelNeXt output ---------------------------------
            points = inputs.get("points")
            if points is not None and hasattr(model, "voxelnext") and model.voxelnext is not None:
                with torch.no_grad():
                    try:
                        blobs = model.voxelnext(points)
                        for idx, b in enumerate(blobs):
                            if b is not None:
                                feat_nan = torch.isnan(b["feat"]).any().item()
                                xyz_nan = torch.isnan(b["xyz"]).any().item()
                                feat_inf = torch.isinf(b["feat"]).any().item()
                                if feat_nan or xyz_nan or feat_inf:
                                    print(
                                        f"[NAN ALERT] VoxelNeXt blob[{idx}] "
                                        f"feat_nan={feat_nan} xyz_nan={xyz_nan} feat_inf={feat_inf}"
                                    )
                        p = next(model.voxelnext.model.parameters(), None)
                        if p is not None:
                            print(f"[DEBUG] voxelnext.model dtype={p.dtype} device={p.device}")
                    except Exception as e:
                        print(f"[DEBUG] voxelnext check failed: {e}")

            # ---- forward ----------------------------------------------------
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logits = outputs.logits if hasattr(outputs, "logits") else None
                if logits is not None:
                    print(
                        f"[NAN ALERT] loss=nan. logits nan={torch.isnan(logits).any().item()} "
                        f"inf={torch.isinf(logits).any().item()} "
                        f"min={logits.min().item():.4f} max={logits.max().item():.4f}"
                    )
                # check first hidden state if available
                if hasattr(outputs, "hidden_states") and outputs.hidden_states:
                    hs = outputs.hidden_states[0]
                    print(
                        f"[NAN ALERT] hidden[0] nan={torch.isnan(hs).any().item()} "
                        f"inf={torch.isinf(hs).any().item()}"
                    )

            # DEBUG: uncomment below if you need per-step diagnostics again
            # if self.state.global_step < 10:
            #     lbl = inputs.get("labels")
            #     n_valid = (lbl != -100).sum().item() if lbl is not None else -1
            #     loss_val = (
            #         loss.item()
            #         if not (torch.isnan(loss).any() or torch.isinf(loss).any())
            #         else float("nan")
            #     )
            #     print(
            #         f"[MMTrainer step {self.state.global_step}] loss={loss_val:.6f} "
            #         f"valid_labels={n_valid}/{lbl.numel() if lbl is not None else 0} "
            #         f"input_ids={inputs['input_ids'].shape}"
            #     )
            return (loss, outputs) if return_outputs else loss

    trainer = _DebugTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )
    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    # Always save the projector explicitly so stage 1b can pick it up.
    if training_args.local_rank in (-1, 0):
        save_mm_projector(model, training_args.output_dir)
        if not model_args.tune_mm_only:
            # Stage 1b: also save full model + tokenizer for downstream stages.
            trainer.save_model(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()

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
    MMQwen,
    SMAPDataset,
    quantizer as q3d,
)


# ---------------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-0.6B")
    mm_input_dim: int = field(default=512)
    pretrain_mm_projector: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a stage-1a mm_projector.bin to load before training."},
    )
    tune_mm_only: bool = field(
        default=False,
        metadata={"help": "If true, freeze the LLM and only train the projector + new token rows."},
    )
    coord_aux_weight: float = field(
        default=0.5,
        metadata={
            "help": "Weight for the per-axis L1 expected-coord auxiliary loss "
            "computed on Q3D coord tokens. Set 0 to disable."
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None)
    feat_folder: str = field(default=None)
    max_length: int = field(default=2048)


# ---------------------------------------------------------------------------
def add_special_tokens_and_resize(tokenizer, model: MMQwen):
    """Register <image>, <|box_start|>, <|box_end|>, and the 1024 Q3D coord
    tokens (<coord_0> .. <coord_1023>) as additional special tokens. Expand
    the embedding table once for everything that's actually new and remember
    the relevant ids on the model.

    The coord tokens are added in numeric order so they're guaranteed to be
    contiguous in the vocabulary — that contiguity is what lets MMQwen slice
    the coord logits with a single (min, max) range.
    """
    coord_names = q3d.coord_token_names()
    extras = [IMAGE_PLACEHOLDER, BOX_START, BOX_END] + coord_names
    needed = [t for t in extras if t not in tokenizer.get_vocab()]
    added = 0
    if needed:
        added = tokenizer.add_special_tokens({"additional_special_tokens": needed})
    if added:
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        # Init new rows to the average embedding (helps stability).
        with torch.no_grad():
            in_emb = model.get_input_embeddings().weight
            out_emb = model.get_output_embeddings().weight
            mean_in = in_emb[:-added].mean(dim=0, keepdim=True)
            mean_out = out_emb[:-added].mean(dim=0, keepdim=True)
            in_emb[-added:] = mean_in
            out_emb[-added:] = mean_out

    image_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    model.set_image_token_id(image_id)

    # Register the contiguous coord-token range + box markers on the model so
    # forward() can compute the aux loss without re-running the tokenizer.
    coord_ids = tokenizer.convert_tokens_to_ids(coord_names)
    coord_ids_sorted = sorted(coord_ids)
    if coord_ids_sorted != list(range(coord_ids_sorted[0], coord_ids_sorted[0] + len(coord_ids))):
        raise RuntimeError(
            "Q3D coord tokens did not get contiguous ids — refuse to continue. "
            f"first ids: {coord_ids[:8]} ..."
        )
    model.set_coord_vocab(
        coord_token_min=coord_ids_sorted[0],
        coord_token_max=coord_ids_sorted[-1],
        bin_value_table=q3d.bin_value_table(),
        box_start_id=tokenizer.convert_tokens_to_ids(BOX_START),
        box_end_id=tokenizer.convert_tokens_to_ids(BOX_END),
    )
    return added


def freeze_llm_except_projector(model: MMQwen, num_added_tokens: int):
    """Stage 1a: freeze everything except mm_projector and the rows of the
    embedding/lm_head that correspond to the newly-added special tokens.

    We can't easily mark only specific rows of an Embedding as trainable, so
    we make the whole embedding/lm_head trainable but with zeroed grads
    elsewhere via a hook is overkill — instead we keep them frozen and rely
    on the fact that the projector + the response tokens (which use existing
    Qwen vocabulary) already give a usable signal. This matches the LLaVA
    stage-1 recipe.
    """
    for p in model.parameters():
        p.requires_grad = False
    for p in model.mm_projector.parameters():
        p.requires_grad = True


def load_mm_projector(model: MMQwen, path: str):
    sd = torch.load(path, map_location="cpu")
    # Accept either a raw projector state_dict or a wrapped {mm_projector.X: ...}.
    if any(k.startswith("mm_projector.") for k in sd):
        sd = {k.split("mm_projector.")[1]: v for k, v in sd.items() if k.startswith("mm_projector.")}
    missing, unexpected = model.mm_projector.load_state_dict(sd, strict=True)
    print(f"[sft] loaded mm_projector from {path} (missing={missing}, unexpected={unexpected})")


def save_mm_projector(model: MMQwen, output_dir: str):
    torch.save(model.mm_projector.state_dict(), Path(output_dir) / "mm_projector.bin")


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
    # If we loaded a checkpoint trained with an existing projector, the linear
    # already exists with the right shape. Otherwise rebuild here.
    if model.mm_projector.in_features != model_args.mm_input_dim:
        model.mm_projector = nn.Linear(
            model_args.mm_input_dim, model.config.hidden_size
        ).to(dtype)
    model.config.mm_input_dim = model_args.mm_input_dim

    n_added = add_special_tokens_and_resize(tokenizer, model)
    model.coord_aux_weight = float(model_args.coord_aux_weight)
    if training_args.local_rank in (-1, 0):
        print(
            f"[sft] added {n_added} special tokens; image_id={model.image_token_id} "
            f"coord_range=[{model.coord_token_min}, {model.coord_token_max}] "
            f"coord_aux_weight={model.coord_aux_weight}"
        )

    if model_args.pretrain_mm_projector:
        load_mm_projector(model, model_args.pretrain_mm_projector)

    if model_args.tune_mm_only:
        freeze_llm_except_projector(model, n_added)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        if training_args.local_rank in (-1, 0):
            print(f"[sft] tune_mm_only: trainable={n_trainable}/{n_total} "
                  f"({100*n_trainable/n_total:.4f}%)")

    # Dataset
    train_ds = SMAPDataset(
        data_path=data_args.data_path,
        tokenizer=tokenizer,
        feat_folder=data_args.feat_folder,
        max_length=data_args.max_length,
    )
    collator = Collator(pad_token_id=tokenizer.pad_token_id)

    trainer = Trainer(
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

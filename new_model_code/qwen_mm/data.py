"""SFT dataset and collator for the Qwen3 + SMAP pipeline.

We re-use the same combined JSON the Llama pipeline used:
  - lidarllm_only_dataset/stage1_train_converted.json (nuCaption)
  - 3dtesting_dataset/train.json                      (nuGrounding)

The data format is unchanged. Each conversation has exactly one "<image>"
token in the user turn — that token is registered as a single special id at
training time, and at forward time it is replaced with the SMAP feature
sequence (12 view tokens) by `MMQwen._splice`.

For SFT we apply Qwen's ChatML template (`apply_chat_template`) and mask out
user-side tokens with -100, so only the assistant response contributes to
the cross-entropy loss.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset

IMAGE_PLACEHOLDER = "<image>"
BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"
SPECIAL_TOKENS = [IMAGE_PLACEHOLDER, BOX_START, BOX_END]


def _load_smap_feat(feat_dir: Path, scene_id: str) -> torch.Tensor:
    blob = torch.load(feat_dir / f"{scene_id}.pt", map_location="cpu")
    feat = blob["output_smap"]
    if feat.dim() == 3:
        feat = feat.squeeze(0)
    return feat.float()  # (n_views, smap_dim)


class SMAPDataset(Dataset):
    """One SFT sample = one nuCaption / nuGrounding conversation + its
    matching SMAP feature."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        feat_folder: str,
        max_length: int = 2048,
    ):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.feat_folder = Path(feat_folder)
        self.max_length = max_length

        # Validate that "<image>" tokenises to exactly one id.
        ids = tokenizer.encode(IMAGE_PLACEHOLDER, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"'<image>' must tokenise to a single id; got {ids}. "
                "Did you forget to add it as an additional_special_token?"
            )
        self.image_token_id = ids[0]

        # Box span ids for coord-mask computation. These tokens already exist
        # in Qwen's vocabulary as native ids (151648, 151649).
        self.box_start_id = tokenizer.convert_tokens_to_ids(BOX_START)
        self.box_end_id = tokenizer.convert_tokens_to_ids(BOX_END)

    def __len__(self) -> int:
        return len(self.data)

    def _build_inputs(self, conversations: List[Dict[str, str]]):
        """Apply Qwen ChatML template, mask out everything that is not the
        final assistant response."""
        messages = []
        for turn in conversations:
            role = "user" if turn["from"].lower() == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})

        # Qwen3 enables a "thinking" block by default; we want concise
        # bbox/caption answers, so disable it.
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, enable_thinking=False
        )

        if not full_text.startswith(prompt_text):
            # Fallback for tokenizer quirks: train on the whole thing.
            ids = self.tokenizer(
                full_text,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).input_ids[0]
            labels = ids.clone()
            return ids, labels

        response_text = full_text[len(prompt_text):]
        prompt_ids = self.tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        response_ids = self.tokenizer(
            response_text, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]

        ids = torch.cat([prompt_ids, response_ids])
        labels = ids.clone()
        labels[: prompt_ids.shape[0]] = -100

        if ids.shape[0] > self.max_length:
            ids = ids[: self.max_length]
            labels = labels[: self.max_length]
        return ids, labels

    def __getitem__(self, i: int):
        sample = copy.deepcopy(self.data[i])
        scene_id = sample.get("scene_id") or sample.get("sample_token")
        try:
            feat = _load_smap_feat(self.feat_folder, scene_id)
        except Exception as e:
            # Quietly resample on missing feature; keeps Trainer running.
            return self[random.randint(0, len(self) - 1)]

        # Make sure there's exactly one <image> token in the prompt.
        first_human = sample["conversations"][0]["value"]
        if IMAGE_PLACEHOLDER not in first_human:
            sample["conversations"][0]["value"] = (
                IMAGE_PLACEHOLDER + "\n" + first_human
            )

        ids, labels = self._build_inputs(sample["conversations"])

        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": torch.ones_like(ids),
            "image": feat,
            "task": sample.get("task"),
            "template_type": sample.get("template_type"),
        }


class Collator:
    """Right-pads input_ids / labels / attention_mask. `images` stays as a
    Python list because each sample's feature length is fixed (12) but we
    keep the door open for variable-length features in future ablations."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        ids = [b["input_ids"] for b in batch]
        lbls = [b["labels"] for b in batch]
        masks = [b["attention_mask"] for b in batch]
        images = [b["image"] for b in batch]

        max_len = max(x.shape[0] for x in ids)
        bsz = len(batch)

        padded_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        padded_lbls = torch.full((bsz, max_len), -100, dtype=torch.long)
        padded_mask = torch.zeros((bsz, max_len), dtype=torch.long)

        for i in range(bsz):
            L = ids[i].shape[0]
            padded_ids[i, :L] = ids[i]
            padded_lbls[i, :L] = lbls[i]
            padded_mask[i, :L] = masks[i]

        return {
            "input_ids": padded_ids,
            "labels": padded_lbls,
            "attention_mask": padded_mask,
            "images": images,
        }

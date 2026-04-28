"""
Drop-in replacement entry point for VTimeLLM training that swaps the visual
backbone from LidarCLIP (.npy, 768-d, up to 100 frames) to the 3D-AVS SMAP
point-captioner features (.pt, 512-d, 12 view tokens per scene).

We monkey-patch two pieces of the upstream B4DL/mllm code:

1.  vtimellm.model.vtimellm_arch.VTimeLLMMetaModel.initialize_vision_modules
    The original hardcodes nn.Linear(768, hidden). We replace it with a Linear
    whose input dimension is configurable via --mm_input_dim (default 512).

2.  vtimellm.train.dataset.LazySupervisedDataset.__getitem__
    The original loads `<feat_folder>/<scene_id>.npy` as a (N, 768) array. We
    load `<feat_folder>/<scene_id>.pt` instead, expecting a dict with key
    "output_smap" of shape (1, n_views, smap_dim).

Everything else (LoRA, deepspeed, conversation templates, trainer, save logic)
is reused from the upstream code so we don't duplicate it.

Run with the same argparse flags as vtimellm/train/train.py, plus optionally
--mm_input_dim 512.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Make the upstream mllm package importable.
B4DL_MLLM = Path("/home/byounggun/B4DL/mllm").resolve()
if str(B4DL_MLLM) not in sys.path:
    sys.path.insert(0, str(B4DL_MLLM))

import copy
import random

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Patch 1: projector input dim.
# ---------------------------------------------------------------------------
import vtimellm.model.vtimellm_arch as _arch  # noqa: E402

_DEFAULT_MM_INPUT_DIM = 512  # 3D-AVS SMAP openseg distill output


def _patched_initialize_vision_modules(self, model_args):
    pretrain = getattr(model_args, "pretrain_mm_mlp_adapter", None)
    input_dim = int(getattr(model_args, "mm_input_dim", _DEFAULT_MM_INPUT_DIM))

    if not hasattr(self, "mm_projector"):
        self.mm_projector = nn.Linear(input_dim, self.config.hidden_size)
    else:
        # Already created (e.g. stage 3 path). If shape doesn't match, recreate.
        if self.mm_projector.in_features != input_dim:
            self.mm_projector = nn.Linear(input_dim, self.config.hidden_size)

    if pretrain is not None:
        weights = torch.load(pretrain, map_location="cpu")

        def _get_w(weights, keyword):
            return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

        sd = _get_w(weights, "mm_projector")
        self.mm_projector.load_state_dict(sd)
        print(f"[smap] loaded mm_projector from {pretrain} (input_dim={input_dim})")

    if getattr(model_args, "use_temporal_embedding", False):
        max_len = getattr(model_args, "temporal_embedding_max_len", 100)
        # NOTE: in the original code the temporal embedding lives in the
        # *feature* space (i.e. before projection). Match that — embedding dim
        # equals the projector input dim.
        self.temporal_embedding = nn.Embedding(max_len, input_dim)
        print(f"[smap] temporal embedding init (max_len={max_len}, dim={input_dim})")
    else:
        self.temporal_embedding = None


_arch.VTimeLLMMetaModel.initialize_vision_modules = _patched_initialize_vision_modules


# ---------------------------------------------------------------------------
# Patch 2: dataset feature loading (.npy 768-d -> .pt SMAP 512-d).
# ---------------------------------------------------------------------------
import vtimellm.train.dataset as _ds  # noqa: E402


def _load_smap_feature(feat_folder: str, scene_id: str) -> torch.Tensor:
    """Load the 3D-AVS SMAP feature for a single scene.

    The extractor saves dict(output_smap=Tensor[1, n_views, smap_dim], n_views=int).
    We squeeze the leading batch dim and return a (n_views, smap_dim) float tensor.
    """
    path = Path(feat_folder) / f"{scene_id}.pt"
    blob = torch.load(path, map_location="cpu")
    feat = blob["output_smap"]
    if feat.dim() == 3:
        feat = feat.squeeze(0)
    return feat.float()


def _patched_getitem(self, i):
    source = copy.deepcopy(self.list_data_dict[i])

    data_type = "video"
    if "<image>" in source["conversations"][0]["value"]:
        source["conversations"][0]["value"] = source["conversations"][0]["value"].replace(
            "<image>", "<video>"
        )
        data_type = "image"

    if "meta" in source:
        def convert(duration, x):
            x = x / duration * 100
            x = str(min(round(x), 99))
            if len(x) == 1:
                x = "0" + x
            return x

        replace_set = []
        for k, v in source["meta"]["token"].items():
            replace_set.append((k, convert(source["meta"]["duration"], v)))
        for l in range(len(source["conversations"])):
            for x1, x2 in replace_set:
                source["conversations"][l]["value"] = (
                    source["conversations"][l]["value"].replace(x1, x2)
                )

    # The default dummy fallback should match the SMAP feature shape so that
    # the projector input_dim assertion holds even if loading fails.
    n_views_default = 12
    smap_dim = _DEFAULT_MM_INPUT_DIM
    image = torch.zeros((n_views_default, smap_dim), dtype=torch.float32)

    try:
        scene_id = source.get("scene_id") or source.get("sample_token")
        image = _load_smap_feature(self.data_args.feat_folder, scene_id)
    except Exception as e:
        print(f"[smap-load-fail] {scene_id}: {e}")
        return random.choice(self)

    if getattr(self.tokenizer, "name", None) == "GLMTokenizer":
        data_dict = _ds.preprocess_glm([source["conversations"]], self.tokenizer)
    else:
        data_dict = _ds.preprocess([source["conversations"]], self.tokenizer, has_image=True)

    if isinstance(i, int):
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

    data_dict["image"] = image

    if source.get("is_time_grounding", False):
        data_dict["is_time_grounding"] = True
        data_dict["start_frame"] = source.get("start_frame", 0)
        data_dict["end_frame"] = source.get("end_frame", 0)
    else:
        data_dict["is_time_grounding"] = False
        data_dict["start_frame"] = -1
        data_dict["end_frame"] = -1

    return data_dict


_ds.LazySupervisedDataset.__getitem__ = _patched_getitem


# ---------------------------------------------------------------------------
# Patch 3: extend ModelArguments with --mm_input_dim so HfArgumentParser picks
# it up from the CLI. We do this by injecting the field into the existing
# dataclass before train() reads sys.argv.
# ---------------------------------------------------------------------------
import vtimellm.train.train as _train  # noqa: E402

# Patch 4: replace the upstream special-token registration. The B4DL codebase
# unconditionally adds <meta> / <4DLiDAR> / <tg>; we are *not* using B4DL's
# stage-2 dataset so registering those would only confuse the embedding
# layer and bloat the vocab. We override to register only the bbox span
# markers we need.
_orig_resize = _train.smart_tokenizer_and_embedding_resize

_OUR_SPECIAL_TOKENS = ["<|box_start|>", "<|box_end|>"]


def _resize_with_box(special_tokens_dict, tokenizer, model):
    """Drop the B4DL-specific tokens; register only ours."""
    return _orig_resize(
        {"additional_special_tokens": list(_OUR_SPECIAL_TOKENS)},
        tokenizer,
        model,
    )


_train.smart_tokenizer_and_embedding_resize = _resize_with_box


# transformers HfArgumentParser builds parsers from dataclass fields at parse
# time, so we need to monkey-patch the dataclass itself.
import dataclasses as _dc  # noqa: E402

_orig_fields = _dc.fields(_train.ModelArguments)
if not any(f.name == "mm_input_dim" for f in _orig_fields):
    new_fields = list(_orig_fields) + [
        _dc.field(  # type: ignore[arg-type]
            default=_DEFAULT_MM_INPUT_DIM,
            metadata={"help": "Input feature dim for the mm projector."},
        )
    ]
    # We can't easily mutate a frozen-style dataclass in-place across modules;
    # instead, dynamically subclass and replace.
    @_dc.dataclass
    class ModelArguments(_train.ModelArguments):  # type: ignore[misc]
        mm_input_dim: int = _dc.field(
            default=_DEFAULT_MM_INPUT_DIM,
            metadata={"help": "Input feature dim for the mm projector."},
        )

    _train.ModelArguments = ModelArguments


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------
def main():
    _train.train()


if __name__ == "__main__":
    main()

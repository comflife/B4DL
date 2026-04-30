"""Qwen3 + 3D-AVS SMAP LiDAR feature multimodal training package.

Replaces the Vicuna/VTimeLLM-based pipeline with a self-contained Qwen3-0.6B
wrapper. Components:

  model.MMQwen      Qwen3ForCausalLM subclass with a Linear projector that
                    converts SMAP feature tokens (12, 512) into hidden-dim
                    embeddings spliced at the position of <image> in input_ids.

  data.SMAPDataset  Loads stage1_combined.json, applies Qwen ChatML template,
                    masks user-side tokens with -100, returns paired LiDAR
                    feature.

  data.Collator     Right-pads variable-length sequences and keeps the per-
                    sample image feature as a list (so MMQwen.forward can
                    splice each one independently).

The image placeholder token is "<image>" (kept identical to the upstream
B4DL data so we don't have to rewrite the JSON). It is registered as an
additional special token at training time so it always tokenises to a
single id, regardless of subword behaviour.
"""

from .data import IMAGE_PLACEHOLDER, BOX_START, BOX_END, Collator, SMAPDataset
from .model import MMQwen, MMQwenConfig
from . import quantizer

__all__ = [
    "MMQwen",
    "MMQwenConfig",
    "SMAPDataset",
    "Collator",
    "IMAGE_PLACEHOLDER",
    "BOX_START",
    "BOX_END",
    "quantizer",
]

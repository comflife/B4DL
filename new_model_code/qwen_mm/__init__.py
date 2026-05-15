"""Qwen3.5-9B + joint VoxelNeXt LiDAR encoder multimodal training package.

Components:

  model.MMQwen          Qwen3_5ForCausalLM subclass with a Linear projector
                        that maps VoxelNeXt voxel-query tokens (K, 128)
                        plus their (K, 3) xyz into hidden-dim embeddings,
                        spliced at the position of <image> in input_ids.
                        Holds the pcdet VoxelNeXt encoder (frozen) inside
                        — call MMQwen.init_voxelnext(...) after loading.

  data.LiDARMMDataset   Loads ChatML conversations + raw 10-sweep LiDAR
                        per nuScenes sample_token via the nuScenes API.
                        The encoder runs inside the model, so the dataset
                        needs neither a feat_folder nor pre-extracted .pt
                        blobs.

  data.Collator         Right-pads variable-length text. Keeps the per-
                        sample point cloud as a Python list so MMQwen's
                        joint forward can encode each independently.

The image placeholder token is "<image>", registered as an additional
special token at training time so it always tokenises to a single id.
"""

from .data import (
    IMAGE_PLACEHOLDER,
    BOX_START,
    BOX_END,
    Collator,
    LiDARMMDataset,
    SMAPDataset,
)
from .model import MMQwen, MMQwenConfig
from . import quantizer_999

__all__ = [
    "MMQwen",
    "MMQwenConfig",
    "LiDARMMDataset",
    "SMAPDataset",  # backwards-compat alias
    "Collator",
    "IMAGE_PLACEHOLDER",
    "BOX_START",
    "BOX_END",
    "quantizer_999",
]

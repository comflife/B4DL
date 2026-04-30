"""Dataset for joint LidarCLIP + VTimeLLM training on Nu-Grounding (3dtesting).

Returns raw point clouds (variable-length (N, 4) tensors) instead of
pre-extracted features. Conversation tokenization reuses preprocess from
the existing dataset.py.
"""
import copy
import json
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset

from vtimellm.constants import IGNORE_INDEX
from vtimellm.train.dataset import preprocess


NUSCENES_INTENSITY_MAX = 255.0


@dataclass
class JointDataArguments:
    data_path: str = field(default=None)
    nuscenes_root: str = field(default="/home/byounggun/B4DL/nuscenes")
    pc_index_path: str = field(
        default="/home/byounggun/B4DL/3dtesting_dataset/sample_token_to_lidar.json")
    lazy_preprocess: bool = False
    max_points: int = field(default=40000,
                            metadata={"help": "Cap points per cloud (random downsample if exceeded)."})


def load_lidar_pcd_bin(path: str, max_points: int) -> torch.Tensor:
    """Read nuScenes LIDAR_TOP .pcd.bin -> tensor (N, 4) in lidar frame.
    Columns: x, y, z, intensity (normalized to [0, 1]).
    """
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]
    pts[:, 3] = pts[:, 3] / NUSCENES_INTENSITY_MAX
    if max_points > 0 and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    return torch.from_numpy(pts.copy())


class JointLidarDataset(Dataset):
    """Loads nu-grounding records + raw point clouds."""

    def __init__(self,
                 data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: JointDataArguments):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        with open(data_path, "r") as f:
            self.records = json.load(f)
        with open(data_args.pc_index_path, "r") as f:
            self.token_to_path = json.load(f)
        self.nuscenes_root = data_args.nuscenes_root

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        source = copy.deepcopy(self.records[i])

        if "<image>" in source["conversations"][0]["value"]:
            source["conversations"][0]["value"] = source["conversations"][0]["value"].replace("<image>", "<video>")

        sample_token = source["scene_id"]
        rel = self.token_to_path.get(sample_token)
        if rel is None:
            return random.choice(self)
        pc_path = os.path.join(self.nuscenes_root, rel)
        try:
            pc = load_lidar_pcd_bin(pc_path, self.data_args.max_points)
        except Exception as e:
            print(f"[load fail] {pc_path}: {e}")
            return random.choice(self)

        data_dict = preprocess([source["conversations"]], self.tokenizer, has_image=True)
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                         labels=data_dict["labels"][0])
        data_dict["point_cloud"] = pc
        return data_dict


@dataclass
class JointDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [inst["input_ids"] for inst in instances]
        labels = [inst["labels"] for inst in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            point_clouds=[inst["point_cloud"] for inst in instances],
        )
        return batch


def make_joint_data_module(tokenizer, data_args):
    train_ds = JointLidarDataset(data_args.data_path, tokenizer, data_args)
    collator = JointDataCollator(tokenizer=tokenizer)
    return dict(train_dataset=train_ds, eval_dataset=None, data_collator=collator)

"""SFT dataset + collator for Qwen3.5-9B + joint VoxelNeXt pipeline.

The dataset loads:
  - One ChatML conversation from the combined / nuGrounding JSON.
  - The 10-sweep accumulated LiDAR point cloud for that sample, **as raw
    points** (not pre-extracted features). The point cloud is handed to
    `MMQwen` which runs a frozen VoxelNeXt encoder internally.

Each conversation has exactly one "<image>" placeholder in the user turn,
which `MMQwen._splice` replaces with the K voxel-query tokens VoxelNeXt
emits.

For SFT we apply Qwen's ChatML template (`apply_chat_template`) and mask
out user-side tokens with -100 so only the assistant response contributes
to the cross-entropy loss.
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

IMAGE_PLACEHOLDER = "<image>"
BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"
SPECIAL_TOKENS = [IMAGE_PLACEHOLDER, BOX_START, BOX_END]


# ---------------------------------------------------------------------------
# 10-sweep accumulator. Mirrors extract_voxelnext_features.py so the
# encoder sees the exact same input distribution as during pretraining.
# ---------------------------------------------------------------------------
def _transform_matrix(translation, rotation_quat) -> np.ndarray:
    from pyquaternion import Quaternion

    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = Quaternion(rotation_quat).rotation_matrix
    M[:3, 3] = np.asarray(translation, dtype=np.float64)
    return M


def load_keyframe_with_sweeps(
    nusc, sample_token: str, n_sweeps: int = 10
) -> Optional[np.ndarray]:
    """Return points (N, 5) in keyframe ego frame as
    [x, y, z, intensity, time_delta_seconds]. Time delta is 0 at the
    keyframe and negative for earlier sweeps."""
    from nuscenes.utils.data_classes import LidarPointCloud

    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]
    sd_rec = nusc.get("sample_data", sd_token)

    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])

    ego_key_to_global = _transform_matrix(pose_rec["translation"], pose_rec["rotation"])
    global_to_ego_key = np.linalg.inv(ego_key_to_global)

    keyframe_ts = sd_rec["timestamp"]

    pts_list = []
    cur_token = sd_token
    swept = 0
    while cur_token != "" and swept < n_sweeps:
        cur_sd = nusc.get("sample_data", cur_token)
        path = os.path.join(nusc.dataroot, cur_sd["filename"])
        pc = LidarPointCloud.from_file(path)
        xyz = pc.points[:3, :].T.astype(np.float32)
        intensity = pc.points[3, :].astype(np.float32) / 255.0

        cs_i = nusc.get("calibrated_sensor", cur_sd["calibrated_sensor_token"])
        pose_i = nusc.get("ego_pose", cur_sd["ego_pose_token"])
        lidar_i_to_ego_i = _transform_matrix(cs_i["translation"], cs_i["rotation"])
        ego_i_to_global = _transform_matrix(pose_i["translation"], pose_i["rotation"])
        lidar_i_to_ego_key = global_to_ego_key @ ego_i_to_global @ lidar_i_to_ego_i

        xyz_h = np.concatenate(
            [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1
        )
        xyz_in_key = (lidar_i_to_ego_key @ xyz_h.T).T[:, :3].astype(np.float32)

        time_delta = (cur_sd["timestamp"] - keyframe_ts) * 1e-6
        time_col = np.full((xyz_in_key.shape[0], 1), time_delta, dtype=np.float32)
        intensity_col = intensity.reshape(-1, 1)

        pts_list.append(
            np.concatenate([xyz_in_key, intensity_col, time_col], axis=1)
        )

        cur_token = cur_sd["prev"]
        swept += 1

    if not pts_list:
        return None
    return np.concatenate(pts_list, axis=0)


# ---------------------------------------------------------------------------
# Dataset.
# ---------------------------------------------------------------------------
class LiDARMMDataset(Dataset):
    """One SFT sample = one nuCaption / nuGrounding conversation + its
    matching raw 10-sweep LiDAR point cloud.

    The dataset stays single-process safe (no NuScenes object cached
    across workers); `_get_nusc()` lazily instantiates the nuScenes API
    inside each dataloader worker on first access.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        nuscenes_root: str,
        nuscenes_version: str = "v1.0-trainval",
        n_sweeps: int = 10,
        max_length: int = 2048,
        task_filter: Optional[str] = None,
        template_filter: Optional[str] = None,
    ):
        with open(data_path) as f:
            self.data = json.load(f)
        if task_filter:
            allowed = {x.strip() for x in task_filter.split(",") if x.strip()}
            self.data = [x for x in self.data if x.get("task") in allowed]
        if template_filter:
            allowed = {x.strip() for x in template_filter.split(",") if x.strip()}
            self.data = [x for x in self.data if x.get("template_type") in allowed]
        if not self.data:
            raise RuntimeError(
                f"No samples left after task_filter={task_filter!r} "
                f"template_filter={template_filter!r}"
            )
        self.tokenizer = tokenizer
        self.nuscenes_root = nuscenes_root
        self.nuscenes_version = nuscenes_version
        self.n_sweeps = int(n_sweeps)
        self.max_length = max_length

        self._nusc = None  # lazy per-worker

        ids = tokenizer.encode(IMAGE_PLACEHOLDER, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"'<image>' must tokenise to a single id; got {ids}. "
                "Did you forget to add it as an additional_special_token?"
            )
        self.image_token_id = ids[0]
        self.box_start_id = tokenizer.convert_tokens_to_ids(BOX_START)
        self.box_end_id = tokenizer.convert_tokens_to_ids(BOX_END)

    # --- nuScenes API (lazy + per-worker) ---------------------------------
    def _get_nusc(self):
        if self._nusc is None:
            from nuscenes.nuscenes import NuScenes

            self._nusc = NuScenes(
                version=self.nuscenes_version,
                dataroot=self.nuscenes_root,
                verbose=False,
            )
        return self._nusc

    # --- text path --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    def _build_inputs(self, conversations: List[Dict[str, str]]):
        messages = []
        for turn in conversations:
            role = "user" if turn["from"].lower() == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})

        # Qwen3.5 enables "thinking" by default; we want concise answers.
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
            ids = self.tokenizer(
                full_text,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            ).input_ids[0]
            return ids, ids.clone()

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

    # --- main ------------------------------------------------------------
    def __getitem__(self, i: int):
        sample = copy.deepcopy(self.data[i])
        scene_id = sample.get("sample_token") or sample.get("scene_id")

        # Load raw 10-sweep LiDAR. On any error (missing sample, missing
        # sweep file, transform issue) we resample so the Trainer keeps
        # ticking instead of raising.
        try:
            nusc = self._get_nusc()
            pts_np = load_keyframe_with_sweeps(nusc, scene_id, self.n_sweeps)
            if pts_np is None or pts_np.shape[0] == 0:
                return self[random.randint(0, len(self) - 1)]
            points = torch.from_numpy(pts_np)  # (N, 5) float32
        except Exception:
            return self[random.randint(0, len(self) - 1)]

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
            "points": points,
            "task": sample.get("task"),
            "template_type": sample.get("template_type"),
        }


# ---------------------------------------------------------------------------
# Backwards-compat alias so any old `from qwen_mm.data import SMAPDataset`
# importer keeps working without code changes.
# ---------------------------------------------------------------------------
SMAPDataset = LiDARMMDataset


class Collator:
    """Right-pads input_ids / labels / attention_mask. `points` stays as a
    Python list because per-sample LiDAR point counts vary."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        ids = [b["input_ids"] for b in batch]
        lbls = [b["labels"] for b in batch]
        masks = [b["attention_mask"] for b in batch]
        points = [b["points"] for b in batch]

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
            "points": points,
        }

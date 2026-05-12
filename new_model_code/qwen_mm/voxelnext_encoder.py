"""Frozen VoxelNeXt encoder for joint training inside MMQwen.

The dataset hands us raw 10-sweep LiDAR points per sample (already loaded
in keyframe ego frame). This encoder:

  1. Voxelizes via spconv VoxelGenerator (matches the official VoxelNeXt
     nuScenes config: voxel_size 0.075m, range [-54,54]xy / [-5,3]z).
  2. Runs the full VoxelNeXt forward (MeanVFE -> backbone -> dense_head)
     under torch.no_grad() so no gradients flow through the encoder.
  3. Picks top-K voxels per sample by max-class hm score, returns a list
     of dicts {feat, xyz, mask} -- the same layout MMQwen._project_images
     already accepts for VoxelNeXt-style features.

pcdet imports are deferred to __init__ so simply importing qwen_mm does
not require pcdet to be built. Build it once with
`pip install -e .` from $VOXELNEXT_ROOT.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


def _add_voxelnext_to_path(voxelnext_root: Path) -> None:
    p = str(voxelnext_root)
    if p not in sys.path:
        sys.path.insert(0, p)


class _DummyFeatureEncoder:
    def __init__(self, num_point_features: int = 5):
        self.num_point_features = num_point_features


class _DummyDataset:
    def __init__(self, dataset_cfg, point_cloud_range, voxel_size, class_names):
        self.dataset_cfg = dataset_cfg
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        self.voxel_size = np.asarray(voxel_size, dtype=np.float32)
        self.class_names = class_names
        self.depth_downsample_factor = None
        self.point_feature_encoder = _DummyFeatureEncoder(num_point_features=5)
        spans = self.point_cloud_range[3:] - self.point_cloud_range[:3]
        self.grid_size = np.round(spans / self.voxel_size).astype(np.int64)

    def __len__(self):
        return 1


class VoxelNeXtEncoder(nn.Module):
    """Frozen pcdet VoxelNeXt -> per-sample top-K voxel queries."""

    FEAT_DIM = 128
    FEATURE_MAP_STRIDE = 8

    def __init__(
        self,
        voxelnext_root: str,
        ckpt_path: str,
        cfg_file: Optional[str] = None,
        top_k: int = 256,
        freeze: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        self.voxelnext_root = Path(voxelnext_root)
        self.ckpt_path = ckpt_path
        self.top_k = int(top_k)
        self.freeze = bool(freeze)
        self._target_device = torch.device(device)

        if cfg_file is None:
            cfg_file = (
                self.voxelnext_root
                / "tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
            )
        self.cfg_file = Path(cfg_file)

        _add_voxelnext_to_path(self.voxelnext_root)

        from easydict import EasyDict  # noqa: F401  (validates env)
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.models import build_network
        from pcdet.utils import common_utils

        # _BASE_CONFIG_ entries in the yaml are relative to VoxelNeXt/tools.
        prev_cwd = os.getcwd()
        os.chdir(str(self.voxelnext_root / "tools"))
        try:
            cfg_from_yaml_file(str(self.cfg_file), cfg)
        finally:
            os.chdir(prev_cwd)

        self._cfg = cfg
        self.point_cloud_range = list(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)

        voxel_size = max_pts = max_vox = None
        for proc in cfg.DATA_CONFIG.DATA_PROCESSOR:
            if proc.NAME == "transform_points_to_voxels":
                voxel_size = proc.VOXEL_SIZE
                max_pts = proc.MAX_POINTS_PER_VOXEL
                max_vox = proc.MAX_NUMBER_OF_VOXELS["test"]
        assert voxel_size is not None, "VoxelNeXt cfg has no voxelization stage"
        self.voxel_size = list(voxel_size)
        self._max_pts = int(max_pts)
        self._max_vox = int(max_vox)

        dataset = _DummyDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            point_cloud_range=self.point_cloud_range,
            voxel_size=self.voxel_size,
            class_names=cfg.CLASS_NAMES,
        )
        logger = common_utils.create_logger()
        self.model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=dataset,
        )
        self.model.load_params_from_file(
            filename=self.ckpt_path, logger=logger, to_cpu=True
        )
        self.model.to(self._target_device).eval()

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        # spconv / pcdet internals are not safe under bf16; keep the encoder
        # in fp32 regardless of what dtype the parent MMQwen lives in.
        self.model = self.model.to(torch.float32)

        # Voxel generator is built lazily on first forward (needs cuda dev).
        self._voxel_gen = None

    # ------------------------------------------------------------------
    def to(self, *args, **kwargs):
        """Keep the pcdet sub-model permanently in fp32 (spconv is not bf16-safe)."""
        # Run the normal Module.to() first so buffers / params we own move.
        super().to(*args, **kwargs)
        # Force the pcdet model back to fp32 regardless of parent dtype.
        if hasattr(self, "model") and self.model is not None:
            self.model = self.model.to(torch.float32)
        return self

    # ------------------------------------------------------------------
    def _current_device(self) -> torch.device:
        """Where pcdet's submodule actually lives RIGHT NOW.

        Trainer/DDP moves the parent MMQwen (and us as a submodule) onto
        its local-rank GPU long after `init_voxelnext()` ran, so we can't
        cache a device at construction time. Read it off a parameter.
        """
        return next(self.model.parameters()).device

    def _ensure_voxel_gen(self, device: torch.device):
        # Rebuild when the device changed (e.g. first call on CPU during
        # init checks, then on cuda:N once DDP moved us).
        if (
            self._voxel_gen is not None
            and getattr(self, "_voxel_gen_device", None) == device
        ):
            return self._voxel_gen
        from spconv.pytorch.utils import PointToVoxel

        self._voxel_gen = PointToVoxel(
            vsize_xyz=self.voxel_size,
            coors_range_xyz=self.point_cloud_range,
            num_point_features=5,
            max_num_voxels=self._max_vox,
            max_num_points_per_voxel=self._max_pts,
            device=device,
        )
        self._voxel_gen_device = device
        return self._voxel_gen

    @staticmethod
    def _crop_to_range(
        points: torch.Tensor, point_cloud_range: Sequence[float]
    ) -> torch.Tensor:
        pmin = points.new_tensor(point_cloud_range[:3])
        pmax = points.new_tensor(point_cloud_range[3:])
        keep = ((points[:, :3] >= pmin) & (points[:, :3] < pmax)).all(dim=1)
        return points[keep]

    # ------------------------------------------------------------------
    def _encode_one(self, points: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        """One-sample VoxelNeXt forward + top-K. points: (N, 5)."""
        device = points.device
        if points.shape[0] == 0:
            return None

        points = self._crop_to_range(points, self.point_cloud_range)
        if points.shape[0] == 0:
            return None

        voxel_gen = self._ensure_voxel_gen(device)
        voxels, coords, num_pts = voxel_gen(points.float())  # (V,P,5),(V,3),(V,)

        # OpenPCDet wants coords as [batch_idx, z, y, x].
        batch_col = torch.zeros((coords.shape[0], 1), dtype=coords.dtype, device=device)
        coords_b = torch.cat([batch_col, coords], dim=1).float()

        # MeanVFE inline: sum points / num_pts (clamped to 1).
        feats_sum = voxels.sum(dim=1)
        occ = num_pts.clamp(min=1).float().unsqueeze(-1)
        voxel_features = (feats_sum / occ).float()

        batch_dict = {
            "batch_size": 1,
            "voxels": voxels,
            "voxel_coords": coords_b,
            "voxel_num_points": num_pts,
            "voxel_features": voxel_features,
        }
        for cur_module in self.model.module_list:
            batch_dict = cur_module(batch_dict)

        sp = batch_dict["encoded_spconv_tensor"]
        feats_all = sp.features                                # (N, 128)
        indices_all = sp.indices                               # (N, 3) [b, y, x]

        head_outs = self.model.dense_head.forward_ret_dict["pred_dicts"]
        hm_list = [d["hm"] for d in head_outs]
        cls_offsets, cum = [], 0
        for d in head_outs:
            cls_offsets.append(cum)
            cum += d["hm"].shape[-1]
        hm_concat = torch.cat(hm_list, dim=-1)
        hm_sig = torch.sigmoid(hm_concat)
        score_all, cls_all = hm_sig.max(dim=-1)

        K = min(self.top_k, score_all.shape[0])
        if K == 0:
            return None
        topv, topi = score_all.topk(K)
        feats_k = feats_all[topi]
        coords_k = indices_all[topi]
        cls_k = cls_all[topi]

        yi = coords_k[:, 1].float()
        xi = coords_k[:, 2].float()
        x_real = (xi + 0.5) * self.FEATURE_MAP_STRIDE * self.voxel_size[0] + self.point_cloud_range[0]
        y_real = (yi + 0.5) * self.FEATURE_MAP_STRIDE * self.voxel_size[1] + self.point_cloud_range[1]

        # z from the head whose argmax class won this voxel.
        z_pred = torch.zeros(K, device=device)
        for k_idx in range(K):
            c = int(cls_k[k_idx].item())
            h_idx = 0
            for hi in range(len(cls_offsets)):
                if hi + 1 < len(cls_offsets) and cls_offsets[hi + 1] <= c:
                    h_idx = hi + 1
                else:
                    break
            z_pred[k_idx] = head_outs[h_idx]["center_z"][topi[k_idx], 0]

        xyz_k = torch.stack([x_real, y_real, z_pred], dim=1)

        feat_pad = torch.zeros(self.top_k, self.FEAT_DIM, device=device)
        xyz_pad = torch.zeros(self.top_k, 3, device=device)
        mask_pad = torch.zeros(self.top_k, dtype=torch.bool, device=device)
        feat_pad[:K] = feats_k
        xyz_pad[:K] = xyz_k
        mask_pad[:K] = True

        return {"feat": feat_pad, "xyz": xyz_pad, "mask": mask_pad}

    # ------------------------------------------------------------------
    def forward(
        self, points_list: List[torch.Tensor]
    ) -> List[Optional[Dict[str, torch.Tensor]]]:
        """Per-sample VoxelNeXt forward. Frozen by default (no grad)."""
        device = self._current_device()
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        was_training = self.model.training
        if self.freeze:
            self.model.eval()
        out = []
        with ctx:
            for pts in points_list:
                if pts is None:
                    out.append(None)
                    continue
                out.append(self._encode_one(pts.to(device)))
        if self.freeze and was_training:
            self.model.train()
        return out

"""Per-scene VoxelNeXt token extraction for the Q3D Qwen3 pipeline.

Replaces the SMAP encoder (12 view-pooled vectors) with **K=256 sparse
voxel queries**, each carrying its own (x, y, z) position and a learned
128-dim feature. The motivation is the diagnosis in `experiment.md`:
SMAP collapses spatial information into 12 polar tokens, which gives the
LLM nothing to attend to when localising from text — VoxelNeXt's sparse
voxel-query design preserves "where there are objects" all the way into
the token sequence.

Pipeline per sample_token:

  1. Accumulate 10 LiDAR sweeps in the keyframe ego frame, attach an
     intensity channel + per-point time-since-keyframe (seconds).
  2. Voxelize at VoxelNeXt's nuScenes config (voxel_size 0.075 m,
     range [-54, 54] xy / [-5, 3] z) using spconv's VoxelGenerator.
  3. Forward through MeanVFE → VoxelResBackBone8xVoxelNeXt → VoxelNeXtHead
     to populate batch_dict['encoded_spconv_tensor'] and the per-voxel
     hm/center predictions inside the head's forward_ret_dict.
  4. Score each surviving voxel by max(hm) across the 6 class-group heads
     (sigmoid-applied for stable [0,1] scores) and take **top-K=256**.
  5. For each kept voxel save:
        feat   (128,)    raw backbone features
        xyz    (3,)      real-world centre (x, y from BEV, z from
                          predicted center_z + voxel mid-z)
        score  scalar    sigmoid(max-class hm logit)
        cls    int8      argmax class id over all heads
  6. Right-pad to K=256 with zeros, plus a (256,) bool `mask`.
  7. Dump as `.pt` blob under --save_dir.

Output blob layout (matches what `qwen_mm.data.SMAPDataset` will load):

    {
      "feat":   torch.float16 (256, 128),
      "xyz":    torch.float16 (256, 3),     # metres in keyframe ego frame
      "score":  torch.float16 (256,),
      "cls":    torch.int8    (256,),
      "mask":   torch.bool    (256,),       # True where a real voxel
    }

This script is meant to be run from the dedicated `voxelnext` conda
env (spconv-cu121 + pcdet from /data1/byounggun/voxelnext_work/VoxelNeXt).
The training env does not need to import any of this.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make pcdet importable. We use the JIA-Lab-research/VoxelNeXt clone.
# ---------------------------------------------------------------------------
VOXELNEXT_ROOT = Path("/data1/byounggun/voxelnext_work/VoxelNeXt")
sys.path.insert(0, str(VOXELNEXT_ROOT))

from easydict import EasyDict  # noqa: E402
from pcdet.config import cfg, cfg_from_yaml_file  # noqa: E402
from pcdet.models import build_network  # noqa: E402
from pcdet.utils import common_utils  # noqa: E402

from nuscenes.nuscenes import NuScenes  # noqa: E402
from nuscenes.utils.data_classes import LidarPointCloud  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402


CFG_FILE = VOXELNEXT_ROOT / "tools/cfgs/nuscenes_models/cbgs_voxel0075_voxelnext.yaml"
NUM_CLASSES = 10  # nuScenes detection
TOP_K = 256
FEAT_DIM = 128


# ---------------------------------------------------------------------------
# 10-sweep loader (mirrors extract_smap_features.py).
# ---------------------------------------------------------------------------
def transform_matrix(translation, rotation_quat) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = Quaternion(rotation_quat).rotation_matrix
    M[:3, 3] = np.asarray(translation, dtype=np.float64)
    return M


def load_keyframe_with_sweeps(
    nusc: NuScenes, sample_token: str, n_sweeps: int = 10
):
    """Return points (N, 5) in keyframe ego frame:
        [x, y, z, intensity, time_delta_seconds].
    Time delta is 0 at keyframe, negative for earlier sweeps."""
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]
    sd_rec = nusc.get("sample_data", sd_token)

    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])

    lidar_to_ego_key = transform_matrix(cs_rec["translation"], cs_rec["rotation"])
    ego_key_to_global = transform_matrix(pose_rec["translation"], pose_rec["rotation"])
    global_to_ego_key = np.linalg.inv(ego_key_to_global)

    keyframe_ts = sd_rec["timestamp"]  # microseconds

    pts_list = []
    cur_token = sd_token
    swept = 0
    while cur_token != "" and swept < n_sweeps:
        cur_sd = nusc.get("sample_data", cur_token)
        path = os.path.join(nusc.dataroot, cur_sd["filename"])
        pc = LidarPointCloud.from_file(path)  # (4, N) -> [x, y, z, intensity]
        xyz = pc.points[:3, :].T.astype(np.float32)
        intensity = pc.points[3, :].astype(np.float32) / 255.0  # nuScenes intensity in [0, 255]

        cs_i = nusc.get("calibrated_sensor", cur_sd["calibrated_sensor_token"])
        pose_i = nusc.get("ego_pose", cur_sd["ego_pose_token"])
        lidar_i_to_ego_i = transform_matrix(cs_i["translation"], cs_i["rotation"])
        ego_i_to_global = transform_matrix(pose_i["translation"], pose_i["rotation"])
        lidar_i_to_ego_key = global_to_ego_key @ ego_i_to_global @ lidar_i_to_ego_i

        xyz_h = np.concatenate(
            [xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1
        )
        xyz_in_key = (lidar_i_to_ego_key @ xyz_h.T).T[:, :3].astype(np.float32)

        time_delta = (cur_sd["timestamp"] - keyframe_ts) * 1e-6  # seconds
        time_col = np.full((xyz_in_key.shape[0], 1), time_delta, dtype=np.float32)
        intensity = intensity.reshape(-1, 1)

        pts_list.append(np.concatenate([xyz_in_key, intensity, time_col], axis=1))

        cur_token = cur_sd["prev"]
        swept += 1

    if not pts_list:
        return None
    return np.concatenate(pts_list, axis=0)


# ---------------------------------------------------------------------------
# Voxelization (spconv VoxelGenerator).
# ---------------------------------------------------------------------------
def build_voxel_generator(point_cloud_range, voxel_size, max_num_points, max_voxels):
    """Return a spconv VoxelGenerator that mirrors VoxelNeXt's data processor."""
    from spconv.pytorch.utils import PointToVoxel

    return PointToVoxel(
        vsize_xyz=list(voxel_size),
        coors_range_xyz=list(point_cloud_range),
        num_point_features=5,
        max_num_voxels=max_voxels,
        max_num_points_per_voxel=max_num_points,
        device=torch.device("cuda"),
    )


def voxelize(points: np.ndarray, voxel_gen) -> dict:
    """points: (N, 5) -> dict with voxel_features (V, P, 5) and
    voxel_coords (V, 3) where coords are [zi, yi, xi]."""
    pts_t = torch.from_numpy(points).float().cuda()
    voxels, coords, num_points_per_voxel = voxel_gen(pts_t)
    return {
        "voxels": voxels,           # (V, P, 5)
        "coords": coords,           # (V, 3)  [z, y, x]
        "num_points": num_points_per_voxel,  # (V,)
    }


# ---------------------------------------------------------------------------
# Build VoxelNeXt model.
# ---------------------------------------------------------------------------
class _DummyFeatureEncoder:
    """Just enough of pcdet.PointFeatureEncoder for build_networks.

    nuScenes 10-sweep input has 5 channels: x, y, z, intensity, time.
    The detector only reads `num_point_features` to size the VFE."""
    def __init__(self, num_point_features: int = 5):
        self.num_point_features = num_point_features


class _DummyDataset:
    """Stand-in for the OpenPCDet dataset object that `build_network` uses
    only for grid_size / point_cloud_range / voxel_size / class_names /
    point_feature_encoder."""
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


def build_voxelnext(ckpt_path: str, device: str = "cuda"):
    # cfg_from_yaml_file follows _BASE_CONFIG_ entries with relative paths
    # rooted at VoxelNeXt/tools/. We chdir there for the load and restore
    # cwd afterwards.
    prev_cwd = os.getcwd()
    os.chdir(str(VOXELNEXT_ROOT / "tools"))
    try:
        cfg_from_yaml_file(str(CFG_FILE), cfg)
    finally:
        os.chdir(prev_cwd)
    pc_range = cfg.DATA_CONFIG.POINT_CLOUD_RANGE
    voxel_size = None
    for proc in cfg.DATA_CONFIG.DATA_PROCESSOR:
        if proc.NAME == "transform_points_to_voxels":
            voxel_size = proc.VOXEL_SIZE
            max_pts = proc.MAX_POINTS_PER_VOXEL
            max_vox = proc.MAX_NUMBER_OF_VOXELS["test"]
    assert voxel_size is not None

    dataset = _DummyDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        point_cloud_range=pc_range,
        voxel_size=voxel_size,
        class_names=cfg.CLASS_NAMES,
    )

    logger = common_utils.create_logger()
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
    model.to(device).eval()

    return model, pc_range, voxel_size, max_pts, max_vox


# ---------------------------------------------------------------------------
# Per-token feature extraction.
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_one(
    nusc: NuScenes,
    sample_token: str,
    model,
    voxel_gen,
    pc_range,
    voxel_size,
    n_sweeps: int = 10,
    top_k: int = TOP_K,
):
    points = load_keyframe_with_sweeps(nusc, sample_token, n_sweeps=n_sweeps)
    if points is None:
        return None

    # Crop to point-cloud range — VoxelNeXt's first data processor does this
    # for the keyframe, and we replicate it here.
    pmin = np.array(pc_range[:3], dtype=np.float32)
    pmax = np.array(pc_range[3:], dtype=np.float32)
    keep = np.all((points[:, :3] >= pmin) & (points[:, :3] < pmax), axis=1)
    points = points[keep]
    if points.shape[0] == 0:
        return None

    voxel_dict = voxelize(points, voxel_gen)
    voxels = voxel_dict["voxels"]            # (V, P, 5)
    coords = voxel_dict["coords"]            # (V, 3) [z, y, x]
    num_pts = voxel_dict["num_points"]       # (V,)

    # OpenPCDet expects coords laid out as [batch_idx, z, y, x].
    batch_col = torch.zeros((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
    coords_b = torch.cat([batch_col, coords], dim=1)  # (V, 4)

    # MeanVFE: (V, P, 5) + (V,) -> (V, 5) by averaging over occupied points.
    feats_sum = voxels.sum(dim=1)
    occ = num_pts.clamp(min=1).float().unsqueeze(-1)
    voxel_features = (feats_sum / occ).float()

    batch_dict = {
        "batch_size": 1,
        "voxels": voxels,
        "voxel_coords": coords_b.float(),     # OpenPCDet stores as float
        "voxel_num_points": num_pts,
        "voxel_features": voxel_features,
    }

    # Forward through model module list (VFE handles voxel_features
    # already, so we still pipe through for consistency).
    for cur_module in model.module_list:
        batch_dict = cur_module(batch_dict)

    sp = batch_dict["encoded_spconv_tensor"]              # spconv tensor at stride 8
    feats_all = sp.features                               # (N, 128)
    indices_all = sp.indices                              # (N, 3) [batch, y, x]
    # Tip: VoxelNeXt's bev_out concats x_conv4/5/6 which can pull the
    # *unique* set of indices to a single layer; we trust their shape.

    # Per-voxel scores: max-class hm logit across all class-group heads.
    head_outs = model.dense_head.forward_ret_dict["pred_dicts"]
    hm_list = [d["hm"] for d in head_outs]                 # each (N, num_cls)
    cls_offsets = []
    cum = 0
    for d in head_outs:
        cls_offsets.append(cum)
        cum += d["hm"].shape[-1]
    hm_concat = torch.cat(hm_list, dim=-1)                 # (N, num_class_total)
    hm_sig = torch.sigmoid(hm_concat)
    score_all, cls_all = hm_sig.max(dim=-1)                # (N,) (N,)

    # top-K (clip if fewer voxels survived).
    K = min(top_k, score_all.shape[0])
    topv, topi = score_all.topk(K)
    feats_k = feats_all[topi]                              # (K, 128)
    coords_k = indices_all[topi]                           # (K, 3) [batch, y, x]
    score_k = topv
    cls_k = cls_all[topi]

    # spatial idx -> real-world xy (centre of the BEV cell).
    feature_map_stride = 8
    yi = coords_k[:, 1].float()
    xi = coords_k[:, 2].float()
    x_real = (xi + 0.5) * feature_map_stride * voxel_size[0] + pc_range[0]
    y_real = (yi + 0.5) * feature_map_stride * voxel_size[1] + pc_range[1]

    # z: use the head's predicted center_z to give the LLM a real-z signal.
    # Pick z from the head whose argmax class won this voxel.
    # cls_k indexes into the concatenated class space; map back to (head_idx, local_cls).
    z_pred = torch.zeros(K, device=feats_k.device)
    for k_idx in range(K):
        c = int(cls_k[k_idx].item())
        # find which head this class belongs to
        h_idx = 0
        for hi in range(len(cls_offsets)):
            if hi + 1 < len(cls_offsets) and cls_offsets[hi + 1] <= c:
                h_idx = hi + 1
            else:
                break
        z_pred[k_idx] = head_outs[h_idx]["center_z"][topi[k_idx], 0]

    xyz_k = torch.stack([x_real, y_real, z_pred], dim=1)   # (K, 3)

    # Pad to top_k.
    real_mask = torch.zeros(top_k, dtype=torch.bool, device=feats_k.device)
    real_mask[:K] = True

    feat_pad = torch.zeros(top_k, FEAT_DIM, device=feats_k.device)
    xyz_pad = torch.zeros(top_k, 3, device=feats_k.device)
    score_pad = torch.zeros(top_k, device=feats_k.device)
    cls_pad = torch.zeros(top_k, dtype=torch.long, device=feats_k.device)
    feat_pad[:K] = feats_k
    xyz_pad[:K] = xyz_k
    score_pad[:K] = score_k
    cls_pad[:K] = cls_k

    return {
        "feat": feat_pad.cpu().to(torch.float16),
        "xyz": xyz_pad.cpu().to(torch.float16),
        "score": score_pad.cpu().to(torch.float16),
        "cls": cls_pad.cpu().to(torch.int8),
        "mask": real_mask.cpu(),
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes_root", default="/data/nuscenes")
    ap.add_argument("--nuscenes_version", default="v1.0-trainval")
    ap.add_argument(
        "--token_list",
        default="/home/byounggun/B4DL/new_model_code/sample_tokens_union.json",
    )
    ap.add_argument(
        "--ckpt",
        default="/data1/byounggun/voxelnext_work/ckpt/voxelnext_nuscenes_kernel1.pth",
        help="Pretrained nuScenes VoxelNeXt checkpoint",
    )
    ap.add_argument(
        "--save_dir",
        default="/data1/byounggun/3davs_b4dl/features/voxelnext_q256",
    )
    ap.add_argument("--top_k", type=int, default=TOP_K)
    ap.add_argument("--n_sweeps", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=-1, help="for debugging")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    return ap.parse_args()


def main():
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    with open(args.token_list) as f:
        tokens = json.load(f)
    if args.end > 0:
        tokens = tokens[args.start : args.end]
    elif args.start > 0:
        tokens = tokens[args.start :]
    if args.limit > 0:
        tokens = tokens[: args.limit]
    print(f"[info] {len(tokens)} sample_tokens to process -> {args.save_dir}")

    print(f"[info] loading nuScenes {args.nuscenes_version} from {args.nuscenes_root}")
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_root, verbose=False)

    print(f"[info] building VoxelNeXt from {args.ckpt}")
    model, pc_range, voxel_size, max_pts, max_vox = build_voxelnext(
        args.ckpt, device=args.device
    )

    voxel_gen = build_voxel_generator(
        point_cloud_range=pc_range,
        voxel_size=voxel_size,
        max_num_points=max_pts,
        max_voxels=max_vox,
    )

    t0 = time.time()
    n_done = n_skip = n_fail = 0
    for i, tok in enumerate(tokens):
        out_path = Path(args.save_dir) / f"{tok}.pt"
        if args.skip_existing and out_path.exists():
            n_skip += 1
            continue
        try:
            blob = extract_one(
                nusc,
                tok,
                model,
                voxel_gen,
                pc_range,
                voxel_size,
                n_sweeps=args.n_sweeps,
                top_k=args.top_k,
            )
            if blob is None:
                n_fail += 1
                continue
            torch.save(blob, out_path)
            n_done += 1
        except Exception as e:
            print(f"[fail] {tok}: {type(e).__name__}: {e}")
            n_fail += 1
            continue

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (n_done + n_skip) / max(1.0, elapsed)
            print(
                f"  [{i+1}/{len(tokens)}] done={n_done} skip={n_skip} fail={n_fail} "
                f"rate={rate:.2f}/s eta={ (len(tokens)-i-1)/max(rate,1e-6)/60:.1f}min"
            )
    print(f"[done] done={n_done} skip={n_skip} fail={n_fail} total_time={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

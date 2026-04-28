"""
Extract per-scene SMAP features from the 3D-AVS point captioner for use as
LiDAR tokens fed into the MLLM.

For each nuScenes sample_token we:
  1. Accumulate 10 LiDAR sweeps in the keyframe ego frame (matches how
     OpenScene/3D-AVS prepares input).
  2. Voxelize at voxel_size=0.05 (config value) using the 3D-AVS Voxelizer.
  3. Build a Minkowski SparseTensor.
  4. mask = boolean of "voxel originated from the keyframe sweep".
  5. camera_id_mask = polar-angle 12-sector partition (lidar mode), filtered to
     keyframe voxels.
  6. Forward through DisNet (MinkUNet18A + SMAP attention pool).
  7. Save output_smap (1, n_views, smap_dim) to <save_dir>/<sample_token>.pt.

The script is designed to be self-contained: it constructs the cfg in code and
imports the model definitions from the cloned 3D-AVS repo so we don't depend on
their data-preprocessing pipeline (which is not yet released).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make the cloned 3D-AVS repo importable.
THIS_DIR = Path(__file__).resolve().parent
AVS_ROOT = Path("/data1/byounggun/3davs_b4dl/3D-AVS")
sys.path.insert(0, str(AVS_ROOT))

from MinkowskiEngine import SparseTensor  # noqa: E402

from dataset.voxelizer import Voxelizer  # noqa: E402
from models.disnet import DisNet  # noqa: E402
from util.config import CfgNode  # noqa: E402

from nuscenes.nuscenes import NuScenes  # noqa: E402
from nuscenes.utils.data_classes import LidarPointCloud  # noqa: E402
from pyquaternion import Quaternion  # noqa: E402


# ---------------------------------------------------------------------------
# Config + model construction
# ---------------------------------------------------------------------------
def build_cfg():
    """Mirrors config/nuscenes/ours_openseg_smap_pretrained.yaml."""
    return CfgNode(
        dict(
            data_root="data/nuscenes_openscene/nuscenes_3d",  # only used for name match
            feature_2d_extractor="openseg",
            voxel_size=0.05,
            arch_3d="MinkUNet18A_smap",
            attn_dim=512,
            smap_dim=512,
            method="smap",
        )
    )


def build_model(cfg, ckpt_path: str, device: str = "cuda"):
    model = DisNet(cfg=cfg)
    sd = torch.load(ckpt_path, map_location="cpu")
    state = {k.replace("module.", ""): v for k, v in sd["state_dict"].items()}
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        print(f"[warn] missing={missing} unexpected={unexpected}")
    model = model.to(device).eval()
    return model


# ---------------------------------------------------------------------------
# nuScenes 10-sweep loader
# ---------------------------------------------------------------------------
def load_keyframe_with_sweeps(nusc: NuScenes, sample_token: str, n_sweeps: int = 10):
    """Return xyz array (N, 3) in keyframe ego frame, plus boolean mask
    marking which points originated from the keyframe sweep."""
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"]["LIDAR_TOP"]
    sd_rec = nusc.get("sample_data", sd_token)

    # Pose at keyframe time (used as the reference frame).
    cs_rec = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
    pose_rec = nusc.get("ego_pose", sd_rec["ego_pose_token"])

    # Reference transforms: lidar -> ego (key) and ego (key) -> global.
    lidar_to_ego_key = transform_matrix(cs_rec["translation"], cs_rec["rotation"])
    ego_key_to_global = transform_matrix(pose_rec["translation"], pose_rec["rotation"])

    # Inverse of ego_key (we will bring sweeps into ego_key frame).
    global_to_ego_key = np.linalg.inv(ego_key_to_global)

    points_list = []
    is_key_list = []

    cur_token = sd_token
    swept = 0
    while cur_token != "" and swept < n_sweeps:
        cur_sd = nusc.get("sample_data", cur_token)
        path = os.path.join(nusc.dataroot, cur_sd["filename"])
        pc = LidarPointCloud.from_file(path)  # (4, N) -> [x,y,z,intensity]
        xyz = pc.points[:3, :].T.astype(np.float32)  # (N, 3)

        # Per-sweep transforms.
        cs_i = nusc.get("calibrated_sensor", cur_sd["calibrated_sensor_token"])
        pose_i = nusc.get("ego_pose", cur_sd["ego_pose_token"])
        lidar_i_to_ego_i = transform_matrix(cs_i["translation"], cs_i["rotation"])
        ego_i_to_global = transform_matrix(pose_i["translation"], pose_i["rotation"])

        # lidar_i -> ego_key  =  inv(ego_key->global) @ ego_i->global @ lidar_i->ego_i
        lidar_i_to_ego_key = global_to_ego_key @ ego_i_to_global @ lidar_i_to_ego_i

        xyz_h = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
        xyz_in_key = (lidar_i_to_ego_key @ xyz_h.T).T[:, :3]

        points_list.append(xyz_in_key)
        is_key_list.append(np.full(xyz_in_key.shape[0], swept == 0, dtype=bool))

        cur_token = cur_sd["prev"]
        swept += 1

    if not points_list:
        return None, None

    coords = np.concatenate(points_list, axis=0)
    is_key = np.concatenate(is_key_list, axis=0)
    return coords, is_key


def transform_matrix(translation, rotation_quat):
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = Quaternion(rotation_quat).rotation_matrix
    M[:3, 3] = np.asarray(translation, dtype=np.float64)
    return M


# ---------------------------------------------------------------------------
# Polar 12-sector camera_id_mask (lidar-only mode)
# ---------------------------------------------------------------------------
def polar_view_mask(coords_xyz: np.ndarray, n_views: int = 12) -> np.ndarray:
    """coords_xyz: (N, 3) float; returns (N, n_views) bool."""
    rho_phi_z = np.stack(
        [
            np.sqrt(coords_xyz[:, 0] ** 2 + coords_xyz[:, 1] ** 2),
            np.arctan2(coords_xyz[:, 1], coords_xyz[:, 0]),
            coords_xyz[:, 2],
        ],
        axis=1,
    )
    step = 2 * np.pi / n_views
    start = -np.pi
    mask = np.zeros((coords_xyz.shape[0], n_views), dtype=bool)
    for i in range(n_views):
        a = start + i * step
        mask[:, i] = (rho_phi_z[:, 1] >= a) & (rho_phi_z[:, 1] < a + step)
    return mask


# ---------------------------------------------------------------------------
# Per-token feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_one(
    nusc: NuScenes,
    sample_token: str,
    model,
    voxelizer: Voxelizer,
    n_views: int,
    n_sweeps: int,
    device: str,
):
    coords, is_key = load_keyframe_with_sweeps(nusc, sample_token, n_sweeps=n_sweeps)
    if coords is None:
        return None

    # Voxelize. We pass per-point integer ids encoded in `labels` so we can
    # recover voxel->key mapping after quantization.
    is_key_int = is_key.astype(np.int32)  # 0/1
    feats_in = np.zeros_like(coords, dtype=np.float32)  # placeholder color
    locs_vox, _, lbl_vox, _, vox_ind = voxelizer.voxelize(
        coords, feats_in, is_key_int, return_ind=True
    )
    # vox_ind = indices into original points kept (1 per voxel).
    is_key_vox = is_key[np.asarray(vox_ind)]  # (N_vox,)
    voxel_float_coords = coords[np.asarray(vox_ind)]  # (N_vox, 3) in meters

    # Build SparseTensor.
    coords_t = torch.from_numpy(locs_vox).int()
    coords_t = torch.cat(
        [torch.zeros(coords_t.shape[0], 1, dtype=torch.int), coords_t], dim=1
    )  # batch idx 0
    feats_t = torch.ones(coords_t.shape[0], 3, dtype=torch.float32)
    sinput = SparseTensor(feats_t.to(device), coords_t.to(device))

    # Masks.
    mask_t = torch.from_numpy(is_key_vox).to(device)  # (N_vox,)

    cam_mask_full = polar_view_mask(voxel_float_coords, n_views=n_views)  # (N_vox, V)
    cam_mask_key = cam_mask_full[is_key_vox]  # (N_keyframe_vox, V)
    cam_mask_t = torch.from_numpy(cam_mask_key).to(device)

    out = model({"sinput": sinput, "mask": mask_t, "camera_id_mask": cam_mask_t})
    feat = out["output_smap"].detach().cpu()  # (1, V, smap_dim)
    return feat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nuscenes_root", default="/data/nuscenes")
    ap.add_argument("--nuscenes_version", default="v1.0-trainval")
    ap.add_argument(
        "--token_list",
        default=str(THIS_DIR / "sample_tokens_union.json"),
        help="JSON list of sample_tokens to extract.",
    )
    ap.add_argument(
        "--ckpt",
        default="/data1/byounggun/3davs_b4dl/ckpt/smap_model_epoch_20.pth.tar",
    )
    ap.add_argument(
        "--save_dir",
        default="/data1/byounggun/3davs_b4dl/features/smap_lidar12",
    )
    ap.add_argument("--n_views", type=int, default=12)
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

    # Load tokens.
    with open(args.token_list) as f:
        tokens = json.load(f)
    if args.end > 0:
        tokens = tokens[args.start : args.end]
    elif args.start > 0:
        tokens = tokens[args.start :]
    if args.limit > 0:
        tokens = tokens[: args.limit]
    print(f"[info] {len(tokens)} sample_tokens to process")

    # nuScenes.
    print(f"[info] loading nuScenes {args.nuscenes_version} from {args.nuscenes_root}")
    nusc = NuScenes(version=args.nuscenes_version, dataroot=args.nuscenes_root, verbose=False)

    # Model.
    cfg = build_cfg()
    model = build_model(cfg, args.ckpt, device=args.device)
    voxelizer = Voxelizer(
        voxel_size=cfg.voxel_size,
        clip_bound=None,
        use_augmentation=False,
        scale_augmentation_bound=None,
        rotation_augmentation_bound=None,
        translation_augmentation_ratio_bound=None,
        ignore_label=255,
    )

    t0 = time.time()
    n_done = 0
    n_skip = 0
    n_fail = 0
    for i, tok in enumerate(tokens):
        out_path = Path(args.save_dir) / f"{tok}.pt"
        if args.skip_existing and out_path.exists():
            n_skip += 1
            continue
        try:
            feat = extract_one(
                nusc,
                tok,
                model,
                voxelizer,
                n_views=args.n_views,
                n_sweeps=args.n_sweeps,
                device=args.device,
            )
            if feat is None:
                n_fail += 1
                continue
            torch.save({"output_smap": feat.half(), "n_views": args.n_views}, out_path)
            n_done += 1
        except Exception as e:
            print(f"[fail] {tok}: {e}")
            n_fail += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (n_done + 1) / max(elapsed, 1e-6)
            print(
                f"[{i + 1}/{len(tokens)}] done={n_done} skip={n_skip} "
                f"fail={n_fail} elapsed={elapsed:.1f}s rate={rate:.2f}/s"
            )

    print(f"[final] done={n_done} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()

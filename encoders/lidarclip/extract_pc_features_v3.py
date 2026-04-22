"""
LiDAR Feature Extractor v3 — Simplified & Robust

Pipeline:
  1. Build sample_token -> filename map from nuScenes sample_data.json (canonical source)
  2. Build filename -> sample_token reverse map for loader output conversion
  3. Check file existence for all required tokens (Phase 1)
  4. Load model, run data loader (trainval + test), save features by sample_token (Phase 2)
  5. Verify all required tokens are present (Phase 3)
  6. Save Stage 1 (.npy per sample_token) and Stage 2 (.npy per scene_id, concatenated)
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project paths
sys.path.insert(0, "/home/byounggun/B4DL/encoders/lidarclip")
sys.path.insert(0, "/home/byounggun/B4DL/mllm")
sys.path.insert(0, "/home/byounggun/B4DL")

import clip
from train import LidarClip
from lidarclip.loader import build_loader as build_dataonly_loader
from lidarclip.model.sst import LidarEncoderSST

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_required_stage1_tokens(json_path):
    """Extract unique sample_tokens from Stage 1 JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    tokens = set()
    for item in data:
        # item["scene_id"] is actually sample_token for Stage 1
        tokens.add(item["scene_id"])
    return tokens


def get_required_stage2_scenes(json_path):
    """Extract unique scene_ids from Stage 2 JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)
    scenes = set()
    for item in data:
        scenes.add(item["scene_id"])
    return scenes


def build_sample_data_maps(sample_data_json_path):
    """
    Build two maps from nuScenes sample_data.json:
      - token_to_path: sample_token -> relative filename (LIDAR_TOP only)
      - path_to_token: relative filename -> sample_token
    """
    with open(sample_data_json_path, "r") as f:
        sample_data = json.load(f)
    
    token_to_path = {}
    path_to_token = {}
    
    for entry in sample_data:
        filename = entry.get("filename", "")
        if "LIDAR_TOP" in filename:
            token = entry["sample_token"]
            token_to_path[token] = filename
            path_to_token[filename] = token
    
    return token_to_path, path_to_token


def build_scene_frame_map(scene_metadata_path, path_to_token):
    """
    Build scene_id -> list of (frame_idx, sample_token) from scene_metadata.json.
    Uses path_to_token to resolve each PATH_xxx to a sample_token.
    """
    with open(scene_metadata_path, "r") as f:
        scenes = json.load(f)
    
    scene_frame_map = {}
    for scene_info in scenes:
        scene_id = scene_info.get("scene_id")
        if not scene_id:
            continue
        frames = scene_info.get("paths", {}).get("PATH_LIDAR_TOP", {})
        num_frames = scene_info.get("num_frames", 0)
        
        frame_tokens = []
        for i in range(num_frames):
            frame_key = f"PATH_{i:03d}"
            rel_path = frames.get(frame_key)
            if not rel_path:
                continue
            token = path_to_token.get(rel_path)
            if token:
                frame_tokens.append((i, token))
        
        scene_frame_map[scene_id] = frame_tokens
    
    return scene_frame_map


def build_loader(datadir, clip_preprocess, batch_size=8, num_workers=4, split="trainval", dataset_name="nuscenes"):
    """Build nuScenes data loader."""
    return build_dataonly_loader(
        datadir,
        clip_preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        split=split,
        dataset_name="with_path",
    )


def create_clean_directory(path):
    """Create directory, remove existing contents if any."""
    if os.path.exists(path):
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
    else:
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # =====================================================================
    # PHASE 0: Build canonical maps from nuScenes sample_data.json
    # =====================================================================
    print("=" * 60)
    print("PHASE 0: Building canonical maps from sample_data.json")
    print("=" * 60)
    
    token_to_path, path_to_token = build_sample_data_maps(args.sample_data_json)
    print(f"[sample_data] Built maps: {len(token_to_path)} LIDAR_TOP entries")
    
    # Stage 2: scene -> ordered list of sample_tokens
    scene_frame_map = {}
    if args.scene_metadata_path and os.path.exists(args.scene_metadata_path):
        scene_frame_map = build_scene_frame_map(args.scene_metadata_path, path_to_token)
        print(f"[scene_metadata] Built frame maps for {len(scene_frame_map)} scenes")
    
    # =====================================================================
    # PHASE 1: Determine required tokens & check file existence
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Checking required files exist on disk")
    print("=" * 60)
    
    required_stage1_tokens = set()
    if args.stage1_train_json and os.path.exists(args.stage1_train_json):
        required_stage1_tokens.update(get_required_stage1_tokens(args.stage1_train_json))
    if args.stage1_val_json and os.path.exists(args.stage1_val_json):
        required_stage1_tokens.update(get_required_stage1_tokens(args.stage1_val_json))
    
    required_stage2_scenes = set()
    if args.stage2_train_json and os.path.exists(args.stage2_train_json):
        required_stage2_scenes.update(get_required_stage2_scenes(args.stage2_train_json))
    
    print(f"[Stage 1] Required unique sample_tokens: {len(required_stage1_tokens)}")
    print(f"[Stage 2] Required unique scenes: {len(required_stage2_scenes)}")
    
    # Check Stage 1 file existence using canonical token_to_path
    stage1_missing_files = []
    for token in tqdm(sorted(required_stage1_tokens), desc="[Check] Stage 1 file existence"):
        if token not in token_to_path:
            stage1_missing_files.append((token, "not_in_sample_data"))
            continue
        full_path = os.path.join(args.data_path, token_to_path[token])
        if not os.path.exists(full_path):
            stage1_missing_files.append((token, full_path))
    
    # Check Stage 2 file existence using scene_frame_map
    stage2_missing_files = []
    for scene_id in tqdm(sorted(required_stage2_scenes), desc="[Check] Stage 2 file existence"):
        if scene_id not in scene_frame_map:
            stage2_missing_files.append((scene_id, "scene_not_in_metadata"))
            continue
        frame_tokens = scene_frame_map[scene_id]
        if not frame_tokens:
            stage2_missing_files.append((scene_id, "no_frames_mapped"))
            continue
        # Check at least first frame exists
        _, first_token = frame_tokens[0]
        full_path = os.path.join(args.data_path, token_to_path[first_token])
        if not os.path.exists(full_path):
            stage2_missing_files.append((scene_id, full_path))
    
    all_ok = True
    if stage1_missing_files:
        print(f"\n[Stage 1] FAILED: {len(stage1_missing_files)} files missing")
        with open("./stage1_missing_files.txt", "w") as f:
            for token, path in stage1_missing_files:
                f.write(f"{token}\t{path}\n")
        all_ok = False
    else:
        print(f"[Stage 1] PASSED: All {len(required_stage1_tokens)} samples have files ✅")
    
    if stage2_missing_files:
        print(f"\n[Stage 2] FAILED: {len(stage2_missing_files)} scenes missing files")
        with open("./stage2_missing_files.txt", "w") as f:
            for scene_id, path in stage2_missing_files:
                f.write(f"{scene_id}\t{path}\n")
        all_ok = False
    else:
        print(f"[Stage 2] PASSED: All {len(required_stage2_scenes)} scenes have files ✅")
    
    if not all_ok:
        print("\n" + "=" * 60)
        print("ABORTED: Fix missing files and re-run.")
        print("=" * 60)
        return
    
    # =====================================================================
    # PHASE 2: Load model and extract features
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Loading model and extracting features")
    print("=" * 60)
    
    checkpoint_path = args.checkpoint
    clip_version = args.clip_version
    clip_model, clip_preprocess = clip.load(clip_version)
    lidar_encoder = LidarEncoderSST(
        "lidarclip/model/sst_encoder_only_config.py", clip_model.visual.output_dim
    )
    model = LidarClip.load_from_checkpoint(
        checkpoint_path,
        lidar_encoder=lidar_encoder,
        clip_model=clip_model,
        batch_size=1,
        epoch_size=1,
    )
    model = model.to("cuda")
    model.eval()
    
    clip_preprocess = clip_preprocess  # from clip.load above
    
    lidar_dict = {}  # sample_token -> feature tensor
    splits_to_load = [s.strip() for s in args.split.split(",")]
    
    for split in splits_to_load:
        print(f"\n[Loader] Loading split: {split}")
        loader = build_loader(
            args.data_path,
            clip_preprocess,
            batch_size=args.batch_size,
            num_workers=4,
            split=split,
            dataset_name=args.dataset_name,
        )
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Extracting [{split}]"):
                _, point_clouds, pc_paths = batch[:3]
                point_clouds = [pc.to("cuda") for pc in point_clouds]
                lidar_features, _ = model.lidar_encoder(point_clouds)
                
                for lidar_feat, pc_path in zip(lidar_features, pc_paths):
                    # Convert absolute path to relative path, then to sample_token
                    rel_path = os.path.relpath(pc_path, args.data_path)
                    token = path_to_token.get(rel_path)
                    if token:
                        lidar_dict[token] = lidar_feat.unsqueeze(0)
    
    print(f"\n[Loader] Total unique sample tokens extracted: {len(lidar_dict)}")
    
    # =====================================================================
    # PHASE 3: Verify coverage
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Verifying coverage")
    print("=" * 60)
    
    stage1_missing_ld = [t for t in required_stage1_tokens if t not in lidar_dict]
    stage2_missing_ld = [s for s in required_stage2_scenes if s not in scene_frame_map or not scene_frame_map[s]]
    
    if stage1_missing_ld:
        print(f"[Stage 1] FAILED: {len(stage1_missing_ld)} samples not loaded by data loader")
        with open("./stage1_missing_loader.txt", "w") as f:
            for m in stage1_missing_ld:
                f.write(m + "\n")
        all_ok = False
    else:
        print(f"[Stage 1] PASSED: All {len(required_stage1_tokens)} samples loaded ✅")
    
    if stage2_missing_ld:
        print(f"[Stage 2] FAILED: {len(stage2_missing_ld)} scenes not mapped")
        with open("./stage2_missing_loader.txt", "w") as f:
            for m in stage2_missing_ld:
                f.write(m + "\n")
        all_ok = False
    else:
        print(f"[Stage 2] PASSED: All {len(required_stage2_scenes)} scenes mapped ✅")
    
    if not all_ok:
        print("\n" + "=" * 60)
        print("ABORTED: Data loader did not load all required samples.")
        print("Try adding --split trainval,test")
        print("=" * 60)
        return
    
    print("\n" + "=" * 60)
    print("ALL VALIDATIONS PASSED. Starting saving...")
    print("=" * 60 + "\n")
    
    # =====================================================================
    # PHASE 4: Save features
    # =====================================================================
    
    # ------------------- Stage 1: per sample_token -------------------
    if required_stage1_tokens:
        create_clean_directory(args.stage1_save_dir)
        for token in tqdm(sorted(required_stage1_tokens), desc="Saving Stage 1"):
            feature = lidar_dict[token]
            save_path = os.path.join(args.stage1_save_dir, f"{token}.npy")
            np.save(save_path, feature.cpu().detach().numpy())
        print(f"[Stage 1] Saved {len(required_stage1_tokens)} features to {args.stage1_save_dir} ✅")
    
    # ------------------- Stage 2: per scene_id, concatenated -------------------
    if required_stage2_scenes:
        create_clean_directory(args.stage2_save_dir)
        for scene_id in tqdm(sorted(required_stage2_scenes), desc="Saving Stage 2"):
            frame_tokens = scene_frame_map[scene_id]
            features = [lidar_dict[token] for _, token in frame_tokens]
            concat = torch.cat(features, dim=0)
            save_path = os.path.join(args.stage2_save_dir, f"{scene_id}.npy")
            np.save(save_path, concat.cpu().detach().numpy())
        print(f"[Stage 2] Saved {len(required_stage2_scenes)} scenes to {args.stage2_save_dir} ✅")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract LiDAR features for B4DL Stage 1 & 2")
    parser.add_argument("--data-path", type=str, default="/home/byounggun/B4DL/nuscenes")
    parser.add_argument("--dataset-name", type=str, default="nuscenes")
    parser.add_argument("--split", type=str, default="trainval,test", help="Comma-separated splits")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default="./ckpt/lidarclip_mm/epochepoch=14.ckpt")
    parser.add_argument("--clip-version", type=str, default="ViT-L/14")
    
    # nuScenes metadata (canonical source)
    parser.add_argument("--sample-data-json", type=str, default="/home/byounggun/B4DL/nuscenes/v1.0-trainval/sample_data.json")
    parser.add_argument("--scene-metadata-path", type=str, default="/home/byounggun/B4DL/b4dl_dataset/metadata/scene_metadata.json")
    
    # Training data (source of truth for what we need)
    parser.add_argument("--stage1-train-json", type=str, default="/home/byounggun/B4DL/lidarllm_only_dataset/stage1_train_converted.json")
    parser.add_argument("--stage1-val-json", type=str, default="/home/byounggun/B4DL/lidarllm_only_dataset/stage1_val_converted.json")
    parser.add_argument("--stage2-train-json", type=str, default="/home/byounggun/B4DL/b4dl_dataset/stage2_combined_meta_v2.json")
    
    # Output dirs
    parser.add_argument("--stage1-save-dir", type=str, default="/home/byounggun/B4DL/lidarclip/stage1_features/")
    parser.add_argument("--stage2-save-dir", type=str, default="/home/byounggun/B4DL/lidarclip/stage2_features/")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

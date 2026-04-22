"""
Extract LiDAR features for Stage 1 and Stage 2 training data.
Ensures 100% coverage of required samples by using training JSONs as the source of truth.
"""
from train import LidarClip
import argparse
import os
import json
import numpy as np
import shutil

from tqdm import tqdm
import torch

import clip

from lidarclip.anno_loader import build_anno_loader
from lidarclip.loader import build_loader as build_dataonly_loader
from lidarclip.model.sst import LidarEncoderSST


def create_clean_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)


def load_model(args):
    assert torch.cuda.is_available()
    
    clip_model, clip_preprocess = clip.load(args.clip_version)
    lidar_encoder = LidarEncoderSST(
        "lidarclip/model/sst_encoder_only_config.py", clip_model.visual.output_dim
    )
    model = LidarClip.load_from_checkpoint(
        args.checkpoint,
        lidar_encoder=lidar_encoder,
        clip_model=clip_model,
        batch_size=1,
        epoch_size=1,
    )
    model.to("cuda")
    return model, clip_preprocess


def build_sample_data_map(sample_data_json_path):
    """Build sample_token -> lidar filename mapping from nuScenes sample_data.json."""
    print(f"Loading sample_data.json from {sample_data_json_path} ...")
    with open(sample_data_json_path, "r") as f:
        data = json.load(f)
    
    mapping = {}
    for entry in data:
        filename = entry.get("filename", "")
        if "LIDAR_TOP" in filename:
            mapping[entry["sample_token"]] = filename
    
    print(f"[sample_data] Found {len(mapping)} LiDAR_TOP entries")
    return mapping


def build_seqmeta_map(frame_json_path):
    """Build sample_token -> PATH_LIDAR_TOP mapping from sequence_metadata.json."""
    with open(frame_json_path, "r") as f:
        data = json.load(f)
    
    mapping = {}
    for d in data:
        for frame in d.get("frames", []):
            token = frame.get("sample_token")
            path = frame.get("PATH_LIDAR_TOP")
            if token and path:
                mapping[token] = path
    
    print(f"[sequence_metadata] Found {len(mapping)} sample_token entries")
    return mapping


def build_scene_map(scene_metadata_path):
    """Build scene_id -> scene metadata mapping."""
    with open(scene_metadata_path, "r") as f:
        data = json.load(f)
    
    mapping = {}
    for scene in data:
        mapping[scene["scene_id"]] = scene
    
    print(f"[scene_metadata] Found {len(mapping)} scenes")
    return mapping


def get_required_stage1_tokens(train_json_path):
    """Get all unique sample_tokens (scene_ids) from Stage 1 training data."""
    with open(train_json_path, "r") as f:
        data = json.load(f)
    
    tokens = set()
    for item in data:
        tokens.add(item["scene_id"])  # scene_id == sample_token
    
    print(f"[Stage 1] Required unique sample_tokens: {len(tokens)}")
    return tokens


def get_required_stage2_scenes(train_json_path):
    """Get all unique scene_ids from Stage 2 training data."""
    with open(train_json_path, "r") as f:
        data = json.load(f)
    
    scenes = set()
    for item in data:
        scenes.add(item["scene_id"])
    
    print(f"[Stage 2] Required unique scenes: {len(scenes)}")
    return scenes


def validate_coverage(args, seqmeta_map, sample_data_map, scene_map, lidar_dict):
    """
    Validate that all required samples can be matched before actual extraction.
    lidar_dict is keyed by sample_token.
    Returns (stage1_missing, stage2_missing).
    """
    stage1_missing = []
    stage2_missing = []
    
    # Stage 1 validation: check by sample_token directly (train + val)
    all_required_tokens = set()
    if args.stage1_train_json and os.path.exists(args.stage1_train_json):
        all_required_tokens.update(get_required_stage1_tokens(args.stage1_train_json))
    if args.stage1_val_json and os.path.exists(args.stage1_val_json):
        all_required_tokens.update(get_required_stage1_tokens(args.stage1_val_json))
    
    for token in tqdm(sorted(all_required_tokens), desc="[Validate] Stage 1"):
        if token not in lidar_dict:
            stage1_missing.append(token)
    
    # Stage 2 validation: check scene coverage (simplified - Phase 1 already checked file existence)
    if args.stage2_train_json and os.path.exists(args.stage2_train_json):
        required_scenes = get_required_stage2_scenes(args.stage2_train_json)
        for scene_id in tqdm(required_scenes, desc="[Validate] Stage 2"):
            if scene_id not in scene_map:
                stage2_missing.append(scene_id)
    
    return stage1_missing, stage2_missing


def validate_file_existence(args, seqmeta_map, sample_data_map, scene_map):
    """
    Validate that all required LiDAR files exist on disk BEFORE model loading.
    Returns (stage1_ok, stage1_missing, stage2_ok, stage2_missing).
    """
    stage1_missing = []
    stage2_missing = []
    
    # Stage 1 validation: check file existence on disk (train + val)
    all_required_tokens = set()
    if args.stage1_train_json and os.path.exists(args.stage1_train_json):
        all_required_tokens.update(get_required_stage1_tokens(args.stage1_train_json))
    if args.stage1_val_json and os.path.exists(args.stage1_val_json):
        all_required_tokens.update(get_required_stage1_tokens(args.stage1_val_json))
    
    for token in tqdm(sorted(all_required_tokens), desc="[Validate] Stage 1 file existence"):
        lidar_path = None
        if token in seqmeta_map:
            lidar_path = os.path.join(args.data_path, seqmeta_map[token])
        elif token in sample_data_map:
            lidar_path = os.path.join(args.data_path, sample_data_map[token])
        
        if not lidar_path or not os.path.exists(lidar_path):
            stage1_missing.append((token, lidar_path))
    
    # Stage 2 validation: check file existence on disk
    if args.stage2_train_json and os.path.exists(args.stage2_train_json):
        required_scenes = get_required_stage2_scenes(args.stage2_train_json)
        for scene_id in tqdm(required_scenes, desc="[Validate] Stage 2 file existence"):
            if scene_id not in scene_map:
                stage2_missing.append((scene_id, "scene_not_in_metadata"))
                continue
            
            scene = scene_map[scene_id]
            num_frames = scene.get("num_frames", 0)
            frames = scene.get("paths", {}).get("PATH_LIDAR_TOP", {})
            
            for i in range(num_frames):
                frame_key = f"PATH_{i:03d}"
                frame_path = frames.get(frame_key)
                if not frame_path:
                    stage2_missing.append((scene_id, f"missing_{frame_key}"))
                    break
                full_path = os.path.join(args.data_path, frame_path)
                if not os.path.exists(full_path):
                    stage2_missing.append((scene_id, full_path))
                    break
    
    return stage1_missing, stage2_missing


def main(args):
    # ========== PHASE 1: Build mappings and validate file existence ==========
    print("="*60)
    print("PHASE 1: Building mappings and validating file existence")
    print("="*60)
    
    seqmeta_map = build_seqmeta_map(args.frame_json_path)
    sample_data_map = build_sample_data_map(args.sample_data_json) if args.sample_data_json else {}
    scene_map = build_scene_map(args.scene_metadata_path) if args.scene_metadata_path else {}
    
    stage1_missing, stage2_missing = validate_file_existence(
        args, seqmeta_map, sample_data_map, scene_map
    )
    
    has_stage1_train = args.stage1_train_json and os.path.exists(args.stage1_train_json)
    has_stage1_val = args.stage1_val_json and os.path.exists(args.stage1_val_json)
    has_stage1 = has_stage1_train or has_stage1_val
    has_stage2 = args.stage2_train_json and os.path.exists(args.stage2_train_json)
    
    all_ok = True
    
    if has_stage1:
        all_required_tokens = set()
        if has_stage1_train:
            all_required_tokens.update(get_required_stage1_tokens(args.stage1_train_json))
        if has_stage1_val:
            all_required_tokens.update(get_required_stage1_tokens(args.stage1_val_json))
        total_stage1 = len(all_required_tokens)
        
        if stage1_missing:
            print(f"\n[Stage 1] FAILED: {len(stage1_missing)}/{total_stage1} samples missing files on disk")
            print(f"[Stage 1] Missing list saved to: ./stage1_missing.txt")
            with open("./stage1_missing.txt", "w") as f:
                for token, path in stage1_missing:
                    f.write(f"{token}\t{path}\n")
            all_ok = False
        else:
            print(f"[Stage 1] PASSED: All {total_stage1} samples have files on disk ✅")
    
    if has_stage2:
        with open(args.stage2_train_json, "r") as f:
            total_stage2 = len(set(item["scene_id"] for item in json.load(f)))
        if stage2_missing:
            print(f"\n[Stage 2] FAILED: {len(stage2_missing)}/{total_stage2} scenes missing files on disk")
            print(f"[Stage 2] Missing list saved to: ./stage2_missing.txt")
            with open("./stage2_missing.txt", "w") as f:
                for scene_id, path in stage2_missing:
                    f.write(f"{scene_id}\t{path}\n")
            all_ok = False
        else:
            print(f"[Stage 2] PASSED: All {total_stage2} scenes have files on disk ✅")
    
    if not all_ok:
        print("\n" + "="*60)
        print("EXTRACTION ABORTED. Please fix missing files and re-run.")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print("ALL FILE VALIDATIONS PASSED.")
    print("="*60 + "\n")
    
    # ========== PHASE 2: Load model and extract features ==========
    print("="*60)
    print("PHASE 2: Loading model and extracting features")
    print("="*60)
    
    model, clip_preprocess = load_model(args)
    build_loader = build_anno_loader if args.use_anno_loader else build_dataonly_loader
    
    # Build filename -> sample_token mapping from sample_data.json
    filename_to_token = {}
    if args.sample_data_json and os.path.exists(args.sample_data_json):
        with open(args.sample_data_json, "r") as f:
            sample_data = json.load(f)
        for entry in sample_data:
            if "LIDAR_TOP" in entry.get("filename", ""):
                filename_to_token[entry["filename"]] = entry["sample_token"]
    
    lidar_dict = {}
    splits_to_load = args.split.split(",")
    
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
            for batch in tqdm(loader, desc=f"Extracting LiDAR features ({split})"):
                _, point_clouds, pc_path = batch[:3]
                point_clouds = [pc.to("cuda") for pc in point_clouds]
                lidar_features, _ = model.lidar_encoder(point_clouds)
                for lidar_feat, lidar_path in zip(lidar_features, pc_path):
                    # Convert absolute path to relative path
                    rel_path = os.path.relpath(lidar_path, args.data_path)
                    token = filename_to_token.get(rel_path)
                    if token:
                        lidar_dict[token] = lidar_feat.unsqueeze(0)
    
    print(f"\n[Loader] Total unique sample tokens extracted: {len(lidar_dict)}")
    
    # ========== PHASE 3: Validate loader coverage ==========
    print("\n" + "="*60)
    print("PHASE 3: Validating loader coverage")
    print("="*60)
    
    stage1_missing_ld, stage2_missing_ld = validate_coverage(
        args, seqmeta_map, sample_data_map, scene_map, lidar_dict
    )
    
    if has_stage1:
        if stage1_missing_ld:
            print(f"[Stage 1] FAILED: {len(stage1_missing_ld)} samples not loaded by data loader")
            print(f"[Stage 1] Missing list saved to: ./stage1_missing_loader.txt")
            with open("./stage1_missing_loader.txt", "w") as f:
                for m in stage1_missing_ld:
                    f.write(m + "\n")
            all_ok = False
        else:
            print(f"[Stage 1] PASSED: All samples loaded by data loader ✅")
    
    if has_stage2:
        if stage2_missing_ld:
            print(f"[Stage 2] FAILED: {len(stage2_missing_ld)} scenes not loaded by data loader")
            print(f"[Stage 2] Missing list saved to: ./stage2_missing_loader.txt")
            with open("./stage2_missing_loader.txt", "w") as f:
                for m in stage2_missing_ld:
                    f.write(m + "\n")
            all_ok = False
        else:
            print(f"[Stage 2] PASSED: All scenes loaded by data loader ✅")
    
    if not all_ok:
        print("\n" + "="*60)
        print("EXTRACTION ABORTED. Data loader did not load all required files.")
        print("Try changing --split or check if files exist in the dataset.")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print("ALL VALIDATIONS PASSED. Starting saving...")
    print("="*60 + "\n")
    
    # ==================== STAGE 1 ====================
    if has_stage1:
        # Collect all required Stage 1 tokens (train + val)
        all_required_tokens = set()
        if has_stage1_train:
            all_required_tokens.update(get_required_stage1_tokens(args.stage1_train_json))
        if has_stage1_val:
            all_required_tokens.update(get_required_stage1_tokens(args.stage1_val_json))
        all_required_tokens = sorted(all_required_tokens)
        
        create_clean_directory(args.stage1_save_dir)
        
        for token in tqdm(all_required_tokens, desc="Saving Stage 1 features"):
            feature = lidar_dict[token]
            save_path = os.path.join(args.stage1_save_dir, f"{token}.npy")
            np.save(save_path, feature.cpu().detach().numpy())
        
        print(f"[Stage 1] Saved: {len(all_required_tokens)}/{len(all_required_tokens)} ✅")
    
    # ==================== STAGE 2 ====================
    if has_stage2:
        required_scenes = get_required_stage2_scenes(args.stage2_train_json)
        create_clean_directory(args.stage2_save_dir)
        
        # Build reverse mapping: relative_path -> sample_token
        path_to_token = {v: k for k, v in sample_data_map.items()}
        
        for scene_id in tqdm(required_scenes, desc="Saving Stage 2 features"):
            scene = scene_map[scene_id]
            num_frames = scene.get("num_frames", 0)
            frames = scene.get("paths", {}).get("PATH_LIDAR_TOP", {})
            
            feature_list = []
            for i in range(num_frames):
                frame_key = f"PATH_{i:03d}"
                frame_path = frames[frame_key]
                
                # Find sample_token for this path
                token = path_to_token.get(frame_path)
                if not token:
                    # Try with full path
                    full_path = os.path.join(args.data_path, frame_path)
                    rel_path = os.path.relpath(full_path, args.data_path)
                    token = path_to_token.get(rel_path)
                
                feature = lidar_dict[token]
                feature_list.append(feature)
            
            concat_feature = torch.cat(feature_list, dim=0)
            save_path = os.path.join(args.stage2_save_dir, f"{scene_id}.npy")
            np.save(save_path, concat_feature.cpu().detach().numpy())
        
        print(f"[Stage 2] Saved: {len(required_scenes)}/{len(required_scenes)} ✅")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./ckpt/lidarclip_mm/epochepoch=14.ckpt")
    parser.add_argument("--clip-version", type=str, default="ViT-L/14")
    parser.add_argument("--data-path", type=str, default="/home/byounggun/B4DL/nuscenes/")
    parser.add_argument("--split", type=str, default="trainval")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--use-anno-loader", action="store_true")
    parser.add_argument("--dataset-name", type=str, default="with_path", choices=["once", "nuscenes", "with_path"])
    
    # Metadata paths
    parser.add_argument("--frame-json-path", type=str, default="./annotations/sequence_metadata.json")
    parser.add_argument("--scene-metadata-path", type=str, default="/home/byounggun/B4DL/b4dl_dataset/metadata/scene_metadata.json")
    parser.add_argument("--sample-data-json", type=str, default="/home/byounggun/B4DL/nuscenes/v1.0-trainval/sample_data.json")
    
    # Training data paths (source of truth)
    parser.add_argument("--stage1-train-json", type=str, default="/home/byounggun/B4DL/lidarllm_only_dataset/stage1_train_converted.json")
    parser.add_argument("--stage1-val-json", type=str, default="/home/byounggun/B4DL/lidarllm_only_dataset/stage1_val_converted.json")
    parser.add_argument("--stage2-train-json", type=str, default="/home/byounggun/B4DL/b4dl_dataset/stage2_combined_meta_v2.json")
    
    # Output dirs
    parser.add_argument("--stage1-save-dir", type=str, default="/home/byounggun/B4DL/lidarclip/stage1_features/")
    parser.add_argument("--stage2-save-dir", type=str, default="/home/byounggun/B4DL/lidarclip/stage2_features/")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

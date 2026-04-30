"""Evaluate the JOINT (LidarCLIP + VTimeLLM) stage2 LoRA on Nu-Grounding val.

Loads: base Vicuna + stage1 mm_projector + stage2 LoRA + trained LidarEncoderSST.
Reads raw .pcd.bin files at runtime (no pre-extracted features).
"""
import os
# torch<1.10/1.12 compat shims, must run before transformers/accelerate.
import torch
if not hasattr(torch.cuda, "is_bf16_supported"):
    torch.cuda.is_bf16_supported = lambda *a, **k: (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8)
if not hasattr(torch.backends, "mps"):
    class _M:
        @staticmethod
        def is_available(): return False
        @staticmethod
        def is_built(): return False
    torch.backends.mps = _M()

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
import distutils.version  # noqa: F401

import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(root_dir)
sys.path.append("/home/byounggun/B4DL/encoders/lidarclip")

import argparse
import json
import numpy as np
from tqdm import tqdm

from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.inference import inference
from lidarclip.model.sst import LidarEncoderSST


NUSCENES_INTENSITY_MAX = 255.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", default="/home/byounggun/B4DL/base_model/vicuna-v1-5-7b")
    parser.add_argument("--pretrain_mm_mlp_adapter",
                        default="/home/byounggun/B4DL/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2",
                        default="/home/byounggun/B4DL/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-3dtesting-joint")
    parser.add_argument("--lidar_sst_config",
                        default="/home/byounggun/B4DL/encoders/lidarclip/lidarclip/model/sst_encoder_only_config.py")
    parser.add_argument("--lidar_ckpt",
                        default=None,
                        help="Path to lidar_encoder.bin from training (default: <stage2>/lidar_encoder.bin)")
    parser.add_argument("--data_path", default="/home/byounggun/B4DL/3dtesting_dataset/val.json")
    parser.add_argument("--pc_index_path", default="/home/byounggun/B4DL/3dtesting_dataset/sample_token_to_lidar.json")
    parser.add_argument("--nuscenes_root", default="/home/byounggun/B4DL/nuscenes")
    parser.add_argument("--max_points", type=int, default=40000)
    parser.add_argument("--log_path",
                        default="./vtimellm/eval/log/3dtesting_joint_val.jsonl")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_pcd(path, max_points):
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :4]
    pts[:, 3] = pts[:, 3] / NUSCENES_INTENSITY_MAX
    if max_points > 0 and pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    return torch.from_numpy(pts.copy())


def load_done_keys(log_path):
    if not os.path.isfile(log_path):
        return set()
    keys = set()
    with open(log_path) as f:
        for line in f:
            try:
                row = json.loads(line)
                keys.add((row.get("scene_id"), row.get("conv_idx")))
            except Exception:
                continue
    return keys


def main():
    args = parse_args()
    disable_torch_init()

    tokenizer, model, _ = load_pretrained_model(args, args.stage2, None)
    model = model.cuda().to(torch.float16).eval()

    # Load joint-trained LidarEncoderSST
    lidar_ckpt = args.lidar_ckpt or os.path.join(args.stage2, "lidar_encoder.bin")
    print(f"[lidar] loading: {lidar_ckpt}")
    encoder = LidarEncoderSST(args.lidar_sst_config, clip_embedding_dim=768)
    sd = torch.load(lidar_ckpt, map_location="cpu")
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    print(f"[lidar] loaded. missing={len(missing)} unexpected={len(unexpected)}")
    encoder = encoder.cuda().to(torch.float32).eval()

    with open(args.data_path) as f:
        data = json.load(f)
    with open(args.pc_index_path) as f:
        token_to_path = json.load(f)
    if args.max_samples > 0:
        data = data[: args.max_samples]

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    done = load_done_keys(args.log_path) if args.resume else set()

    feat_id, feat = None, None
    written = 0
    fout = open(args.log_path, "a")
    try:
        for conv_idx, item in enumerate(tqdm(data, desc="3dtesting-joint eval")):
            scene_id = item["scene_id"]
            sample_token = item.get("sample_token", scene_id)
            convs = item["conversations"]
            human = next((c for c in convs if c["from"] == "human"), None)
            gpt = next((c for c in convs if c["from"] == "gpt"), None)
            if human is None or gpt is None:
                continue
            key = (scene_id, conv_idx)
            if key in done:
                continue

            # Re-encode if scene changed
            if scene_id != feat_id:
                rel = token_to_path.get(sample_token)
                if rel is None:
                    continue
                pc_path = os.path.join(args.nuscenes_root, rel)
                if not os.path.isfile(pc_path):
                    continue
                pc = load_pcd(pc_path, args.max_points).cuda().float()
                with torch.no_grad():
                    feat_768, _ = encoder([pc])  # (1, 768)
                feat = feat_768.to(model.dtype)  # (1, 768) — single-frame "image"
                feat_id = scene_id

            with torch.no_grad():
                pred = inference(model, feat, human["value"], tokenizer)

            row = {
                "scene_id": scene_id,
                "sample_token": sample_token,
                "conv_idx": conv_idx,
                "view": item.get("view"),
                "template_type": item.get("template_type"),
                "question": human["value"],
                "gt_answer": gpt["value"],
                "pred_answer": pred,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
    finally:
        fout.close()

    print(f"\nWrote {written} -> {args.log_path}")


if __name__ == "__main__":
    main()

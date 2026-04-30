"""Evaluate the feature-cached stage2_3dtesting LoRA on Nu-Grounding val.

Loads: base Vicuna + stage1 mm_projector + stage2 LoRA.
Uses pre-extracted LiDAR features from --feat_folder.
"""
import os
# torch 1.9 + setuptools>=60 compat. Must run before transformers import.
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")
import distutils.version  # noqa: F401

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

import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(root_dir)

import argparse
import json
import numpy as np
from tqdm import tqdm

from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.inference import inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_base", default="/home/byounggun/B4DL/base_model/vicuna-v1-5-7b")
    parser.add_argument("--pretrain_mm_mlp_adapter",
                        default="/home/byounggun/B4DL/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2",
                        default="/home/byounggun/B4DL/mllm/checkpoints/vtimellm-vicuna-v1-5-7b-3dtesting")
    parser.add_argument("--data_path", default="/home/byounggun/B4DL/3dtesting_dataset/val.json")
    parser.add_argument("--feat_folder", default="/home/byounggun/B4DL/lidarclip/stage1_features")
    parser.add_argument("--log_path",
                        default="./vtimellm/eval/log/3dtesting_val.jsonl")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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

    with open(args.data_path) as f:
        data = json.load(f)
    if args.max_samples > 0:
        data = data[: args.max_samples]

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    done = load_done_keys(args.log_path) if args.resume else set()

    feat_id, feat = None, None
    written = 0
    fout = open(args.log_path, "a")
    try:
        for conv_idx, item in enumerate(tqdm(data, desc="3dtesting eval")):
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

            if scene_id != feat_id:
                fp = os.path.join(args.feat_folder, f"{scene_id}.npy")
                if not os.path.isfile(fp):
                    continue
                feat = torch.from_numpy(np.load(fp)).cuda().to(model.dtype)
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

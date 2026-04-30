import os
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(root_dir)

import argparse
import json
import random
import torch
import numpy as np
from tqdm import tqdm

from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.inference import inference


def parse_args():
    parser = argparse.ArgumentParser(description="Stage1 (mm_projector only) eval on LiDAR-LLM val set")
    parser.add_argument("--model_base", type=str, default="./base_model/vicuna-v1-5-7b")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str,
                        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--data_path", type=str,
                        default="./lidarllm_only_dataset/stage1_val_converted.json")
    parser.add_argument("--feat_folder", type=str,
                        default="./lidarclip/stage1_features")
    parser.add_argument("--log_path", type=str,
                        default="./vtimellm/eval/log/stage1_val.jsonl")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Cap total conversations to evaluate (-1 = all).")
    parser.add_argument("--max_per_scene", type=int, default=-1,
                        help="Cap conversations per scene_id (-1 = all).")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle data before applying max_samples.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="Skip entries already present in log_path.")
    return parser.parse_args()


def load_done_keys(log_path):
    if not os.path.isfile(log_path):
        return set()
    keys = set()
    with open(log_path, "r") as f:
        for line in f:
            try:
                row = json.loads(line)
                keys.add((row.get("scene_id"), row.get("conv_idx")))
            except Exception:
                continue
    return keys


def main():
    args = parse_args()
    random.seed(args.seed)

    disable_torch_init()
    tokenizer, model, _ = load_pretrained_model(args, stage2=None, stage3=None)
    model = model.cuda()
    model.to(torch.float16)
    model.eval()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    if args.shuffle:
        random.shuffle(data)

    if args.max_per_scene > 0:
        seen = {}
        filtered = []
        for item in data:
            sid = item.get("scene_id")
            seen[sid] = seen.get(sid, 0) + 1
            if seen[sid] <= args.max_per_scene:
                filtered.append(item)
        data = filtered

    if args.max_samples > 0:
        data = data[: args.max_samples]

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    done = load_done_keys(args.log_path) if args.resume else set()

    feat_cache_id = None
    feat_cache = None

    written = 0
    skipped_no_feat = 0
    fout = open(args.log_path, "a")

    try:
        for conv_idx, item in enumerate(tqdm(data, desc="Stage1 eval")):
            scene_id = item.get("scene_id")
            sample_token = item.get("sample_token")
            convs = item.get("conversations", [])
            if len(convs) < 2:
                continue
            human_turn = next((c for c in convs if c.get("from") == "human"), None)
            gpt_turn = next((c for c in convs if c.get("from") == "gpt"), None)
            if human_turn is None or gpt_turn is None:
                continue

            key = (scene_id, conv_idx)
            if key in done:
                continue

            question = human_turn["value"]
            gt_answer = gpt_turn["value"]

            if scene_id != feat_cache_id:
                feat_path = os.path.join(args.feat_folder, f"{scene_id}.npy")
                if not os.path.isfile(feat_path):
                    skipped_no_feat += 1
                    feat_cache_id = None
                    feat_cache = None
                    continue
                feat_cache = torch.from_numpy(np.load(feat_path)).cuda().to(model.dtype)
                feat_cache_id = scene_id

            with torch.no_grad():
                pred_answer = inference(model, feat_cache, question, tokenizer)

            row = {
                "scene_id": scene_id,
                "sample_token": sample_token,
                "conv_idx": conv_idx,
                "question": question,
                "gt_answer": gt_answer,
                "pred_answer": pred_answer,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
    finally:
        fout.close()

    print(f"\n=== Stage1 Eval Summary ===")
    print(f"Wrote: {written}")
    print(f"Skipped (no feature): {skipped_no_feat}")
    print(f"Log: {args.log_path}")


if __name__ == "__main__":
    main()

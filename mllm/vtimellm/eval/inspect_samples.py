import os
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(root_dir)

import argparse
import torch
import json
import numpy as np
from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.inference import inference


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect 10 samples")
    parser.add_argument("--model_base", type=str, default="./base_model/vicuna-v1-5-7b")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str,
                        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str,
                        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage2/checkpoint-9266")
    parser.add_argument("--stage3", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--feat_folder", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="./inspect_10samples.txt")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    return args


def main(args):
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    model.to(torch.float16)
    model.eval()

    with open(args.data_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = list(data.values())

    # Take first N samples
    samples = data[:args.num_samples]

    lines = []
    lines.append("=" * 60)
    lines.append(f"Model: {args.stage2}")
    lines.append(f"Data: {args.data_path}")
    lines.append(f"Samples: {len(samples)}")
    lines.append("=" * 60)
    lines.append("")

    for idx, item in enumerate(samples, 1):
        scene_id = item.get('scene_id')
        question = item.get('question', '')
        gt_answer = item.get('answer', '')

        feat_path = os.path.join(args.feat_folder, f"{scene_id}.npy")
        if not os.path.isfile(feat_path):
            pred_answer = "[FEATURE NOT FOUND]"
        else:
            features = torch.from_numpy(np.load(feat_path)).cuda().to(model.dtype)
            query = f"<4DLiDAR>\n{question}"
            with torch.no_grad():
                pred_answer = inference(model, features, query, tokenizer)

        lines.append(f"--- Sample {idx} | Scene: {scene_id} ---")
        lines.append(f"Q: {question}")
        lines.append(f"GT: {gt_answer}")
        lines.append(f"Pred: {pred_answer}")
        lines.append("")

    with open(args.out_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"Saved to {args.out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

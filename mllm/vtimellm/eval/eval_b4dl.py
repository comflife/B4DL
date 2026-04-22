import os
import sys
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(root_dir)

import re
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.inference import inference


def parse_args():
    parser = argparse.ArgumentParser(description="B4DL Evaluation")
    parser.add_argument("--model_base", type=str, default="./base_model/vicuna-v1-5-7b")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str,
                        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str,
                        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default=None,
                        help="Optional stage3 checkpoint")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to B4DL test JSON (e.g., b4dl_dataset/test/temporal_understanding.json)")
    parser.add_argument("--feat_folder", type=str, required=True,
                        help="Path to LiDAR feature .npy files")
    parser.add_argument("--log_path", type=str, default='vtimellm/eval/log/b4dl_eval.jsonl')
    parser.add_argument("--task", type=str, default='all',
                        choices=['all', 'grounding', 'non_grounding'])
    args = parser.parse_args()
    return args


def parse_interval_from_text(text):
    """Parse 'from frame XXX to frame YYY' from text. Returns (start, end) or None."""
    match = re.search(r'from frame (\d+) to frame (\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Fallback: single frame like "FRAME 003."
    single = re.search(r'frame (\d+)', text, re.IGNORECASE)
    if single:
        f = int(single.group(1))
        return f, f
    return None


def compute_iou(pred_interval, gt_interval):
    """Compute IoU between two frame intervals."""
    if pred_interval is None or gt_interval is None:
        return 0.0
    pred_s, pred_e = pred_interval
    gt_s, gt_e = gt_interval
    intersection = max(0, min(pred_e, gt_e) - max(pred_s, gt_s))
    union = max(pred_e, gt_e) - min(pred_s, gt_s)
    if union == 0:
        return 0.0
    return intersection / union


def write_log(log_path, scene_id, question, gt_answer, pred_answer, pred_interval, gt_interval, iou):
    log = {
        'scene_id': scene_id,
        'question': question,
        'gt_answer': gt_answer,
        'pred_answer': pred_answer,
    }
    if gt_interval is not None:
        log['gt_interval'] = list(gt_interval)
        log['pred_interval'] = list(pred_interval) if pred_interval else None
        log['iou'] = iou
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a') as f:
        f.write(json.dumps(log) + '\n')


def main(args):
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    model = model.cuda()
    model.to(torch.float16)
    model.eval()

    with open(args.data_path, 'r') as f:
        data = json.load(f)

    # B4DL test data is a list of dicts
    if isinstance(data, dict):
        data = list(data.values())

    total = 0
    tg_total = 0
    tg_ious = []

    for item in tqdm(data, desc="Evaluating"):
        scene_id = item.get('scene_id')
        question = item.get('question', '')
        gt_answer = item.get('answer', '')

        # Load LiDAR feature
        feat_path = os.path.join(args.feat_folder, f"{scene_id}.npy")
        if not os.path.isfile(feat_path):
            print(f"[Warning] Feature not found: {feat_path}")
            continue

        features = torch.from_numpy(np.load(feat_path)).cuda().to(model.dtype)

        # Prepare query with <4DLiDAR> token (same as training)
        query = f"<4DLiDAR>\n{question}"

        # Inference
        with torch.no_grad():
            pred_answer = inference(model, features, query, tokenizer)

        # Determine if this is a time grounding sample
        gt_interval = parse_interval_from_text(gt_answer)
        pred_interval = parse_interval_from_text(pred_answer)

        is_tg = gt_interval is not None

        # Filter by task type if specified
        if args.task == 'grounding' and not is_tg:
            continue
        if args.task == 'non_grounding' and is_tg:
            continue

        iou = 0.0
        if is_tg:
            iou = compute_iou(pred_interval, gt_interval)
            tg_ious.append(iou)
            tg_total += 1

        write_log(args.log_path, scene_id, question, gt_answer, pred_answer,
                  pred_interval, gt_interval, iou)
        total += 1

    # Print summary
    print(f"\n=== Evaluation Summary ===")
    print(f"Total evaluated: {total}")
    if tg_total > 0:
        print(f"Time Grounding samples: {tg_total}")
        print(f"mIoU: {sum(tg_ious) / tg_total * 100:.2f}")
        for thresh in [0.3, 0.5, 0.7]:
            recall = sum(1 for iou in tg_ious if iou >= thresh) / tg_total
            print(f"R1@{thresh}: {recall * 100:.2f}")
    else:
        print("No time grounding samples found in this data.")


if __name__ == "__main__":
    args = parse_args()
    main(args)

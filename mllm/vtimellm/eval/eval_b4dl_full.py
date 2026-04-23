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
from collections import defaultdict

from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.inference import inference


def parse_args():
    parser = argparse.ArgumentParser(description="B4DL Full Evaluation")
    parser.add_argument("--model_base", type=str, default="./base_model/vicuna-v1-5-7b")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str,
                        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str,
                        default="./checkpoints/vtimellm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--feat_folder", type=str, required=True)
    parser.add_argument("--log_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples for quick test")
    args = parser.parse_args()
    return args


# ============ Time Grounding Metrics ============
def parse_interval_from_text(text):
    match = re.search(r'from frame (\d+) to frame (\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    single = re.search(r'frame (\d+)', text, re.IGNORECASE)
    if single:
        f = int(single.group(1))
        return f, f
    return None


def compute_iou(pred_interval, gt_interval):
    if pred_interval is None or gt_interval is None:
        return 0.0
    pred_s, pred_e = pred_interval
    gt_s, gt_e = gt_interval
    intersection = max(0, min(pred_e, gt_e) - max(pred_s, gt_s))
    union = max(pred_e, gt_e) - min(pred_s, gt_s)
    if union == 0:
        return 0.0
    return intersection / union


# ============ Caption/QA Metrics ============
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        import string
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_accuracy(gts, preds):
    correct = sum(1 for g, p in zip(gts, preds) if normalize_answer(g) == normalize_answer(p))
    return correct / len(gts) * 100 if gts else 0.0


def compute_caption_metrics(gts_dict, res_dict):
    """gts_dict and res_dict: {id: [text]}"""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    results = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts_dict, res_dict)
        if isinstance(method, list):
            for m, s in zip(method, score):
                results[m] = s * 100
        else:
            results[method] = score * 100
    return results


def detect_task(data_path):
    basename = os.path.basename(data_path).lower()
    if 'temporal_understanding' in basename:
        return 'temporal'
    elif 'binary' in basename or 'existence' in basename:
        return 'accuracy'
    elif 'description' in basename or 'comprehensive' in basename:
        return 'caption'
    else:
        return 'unknown'


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

    if args.max_samples:
        data = data[:args.max_samples]

    task = detect_task(args.data_path)
    print(f"Detected task: {task}")
    print(f"Total samples: {len(data)}")

    # Accumulators
    all_gts = []
    all_preds = []
    tg_ious = []
    tg_total = 0
    total_evaluated = 0
    feature_missing = 0

    # For caption metrics
    gts_dict = {}
    res_dict = {}

    for idx, item in enumerate(tqdm(data, desc="Evaluating")):
        scene_id = item.get('scene_id')
        question = item.get('question', '')
        gt_answer = item.get('answer', '')

        feat_path = os.path.join(args.feat_folder, f"{scene_id}.npy")
        if not os.path.isfile(feat_path):
            feature_missing += 1
            continue

        features = torch.from_numpy(np.load(feat_path)).cuda().to(model.dtype)
        query = f"<4DLiDAR>\n{question}"

        with torch.no_grad():
            # Debug: check for nan/inf in features
            if torch.isnan(features).any() or torch.isinf(features).any():
                print(f"[Warning] Feature contains nan/inf for scene {scene_id}. Skipping.")
                feature_missing += 1
                continue
            try:
                pred_answer = inference(model, features, query, tokenizer)
            except RuntimeError as e:
                print(f"[Error] Inference failed at sample {idx}, scene {scene_id}: {e}")
                print(f"  Question: {question}")
                pred_answer = ""

        all_gts.append(gt_answer)
        all_preds.append(pred_answer)
        total_evaluated += 1

        if task == 'temporal':
            gt_interval = parse_interval_from_text(gt_answer)
            pred_interval = parse_interval_from_text(pred_answer)
            if gt_interval is not None:
                iou = compute_iou(pred_interval, gt_interval)
                tg_ious.append(iou)
                tg_total += 1
        elif task == 'caption':
            gts_dict[idx] = [gt_answer]
            res_dict[idx] = [pred_answer]

        # Log individual results
        if args.log_path:
            log = {
                'scene_id': scene_id,
                'question': question,
                'gt_answer': gt_answer,
                'pred_answer': pred_answer,
            }
            if task == 'temporal':
                gt_interval = parse_interval_from_text(gt_answer)
                pred_interval = parse_interval_from_text(pred_answer)
                if gt_interval:
                    log['gt_interval'] = list(gt_interval)
                    log['pred_interval'] = list(pred_interval) if pred_interval else None
                    log['iou'] = compute_iou(pred_interval, gt_interval)
            os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
            with open(args.log_path, 'a') as f:
                f.write(json.dumps(log) + '\n')

    # ============ Print Results ============
    print(f"\n{'='*60}")
    print(f"B4DL Evaluation Results: {os.path.basename(args.data_path)}")
    print(f"{'='*60}")
    print(f"Total evaluated: {total_evaluated} / {len(data)}")
    if feature_missing:
        print(f"Feature missing: {feature_missing}")

    if task == 'temporal':
        print(f"Time Grounding samples: {tg_total}")
        if tg_total > 0:
            print(f"mIoU: {sum(tg_ious) / tg_total * 100:.2f}")
            for thresh in [0.3, 0.5, 0.7]:
                recall = sum(1 for iou in tg_ious if iou >= thresh) / tg_total
                print(f"R1@{thresh}: {recall * 100:.2f}")
    elif task == 'accuracy':
        acc = compute_accuracy(all_gts, all_preds)
        print(f"Accuracy: {acc:.2f}%")
    elif task == 'caption':
        if gts_dict:
            metrics = compute_caption_metrics(gts_dict, res_dict)
            for k, v in metrics.items():
                print(f"{k}: {v:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

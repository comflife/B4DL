import json
import argparse
import re
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Compute B4DL paper metrics from eval logs")
    parser.add_argument("--temporal_log", type=str, default=None, help="temporal_understanding eval log")
    parser.add_argument("--binary_log", type=str, default=None, help="binary eval log")
    parser.add_argument("--existence_log", type=str, default=None, help="existence eval log")
    parser.add_argument("--description_log", type=str, default=None, help="description eval log")
    parser.add_argument("--comprehensive_log", type=str, default=None, help="comprehensive eval log")
    parser.add_argument("--model_name", type=str, default="Model")
    args = parser.parse_args()
    return args


def parse_interval(text):
    match = re.search(r'from frame (\d+) to frame (\d+)', text, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    single = re.search(r'frame (\d+)', text, re.IGNORECASE)
    if single:
        f = int(single.group(1))
        return f, f
    return None


def compute_iou(pred, gt):
    if pred is None or gt is None:
        return 0.0
    ps, pe = pred
    gs, ge = gt
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0


def normalize(s):
    import string
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(c for c in s if c not in string.punctuation)
    return ' '.join(s.split())


def load_log(path):
    if not path or not os.path.exists(path):
        return []
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def compute_temporal_metrics(data):
    tg_ious = []
    for item in data:
        gt_int = item.get('gt_interval')
        pred_int = item.get('pred_interval')
        if gt_int is not None:
            iou = compute_iou(pred_int, gt_int)
            tg_ious.append(iou)
    if not tg_ious:
        return {}
    miou = sum(tg_ious) / len(tg_ious)
    recalls = {}
    for th in [0.3, 0.5, 0.7]:
        recalls[f'R1@{th}'] = sum(1 for i in tg_ious if i >= th) / len(tg_ious)
    return {'mIoU': miou, **recalls, 'n': len(tg_ious)}


def compute_accuracy(data):
    correct = sum(1 for item in data if normalize(item['gt_answer']) == normalize(item['pred_answer']))
    return correct / len(data) if data else 0.0


def compute_caption_metrics(data):
    """Compute BLEU-4, ROUGE-L, METEOR, BERTScore from log data."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.meteor.meteor import Meteor
    from bert_score import score as bert_score

    gts_dict = {}
    res_dict = {}
    all_gts = []
    all_preds = []

    for idx, item in enumerate(data):
        gts_dict[idx] = [item['gt_answer']]
        res_dict[idx] = [item['pred_answer']]
        all_gts.append(item['gt_answer'])
        all_preds.append(item['pred_answer'])

    results = {}

    # BLEU-4
    try:
        bleu_scorer = Bleu(4)
        bleu_scores, _ = bleu_scorer.compute_score(gts_dict, res_dict)
        results['B@4'] = bleu_scores[3]  # Bleu_4
    except Exception as e:
        print(f"BLEU error: {e}")
        results['B@4'] = 0.0

    # ROUGE-L
    try:
        rouge_scorer = Rouge()
        rouge_score, _ = rouge_scorer.compute_score(gts_dict, res_dict)
        results['ROUGE-L'] = rouge_score
    except Exception as e:
        print(f"ROUGE error: {e}")
        results['ROUGE-L'] = 0.0

    # METEOR via nltk (no Java needed)
    try:
        from nltk.translate.meteor_score import meteor_score
        from nltk import word_tokenize
        meteor_scores = []
        for idx in gts_dict:
            ref = word_tokenize(gts_dict[idx][0])
            hyp = word_tokenize(res_dict[idx][0])
            meteor_scores.append(meteor_score([ref], hyp))
        results['METEOR'] = sum(meteor_scores) / len(meteor_scores)
    except Exception as e:
        print(f"METEOR error: {e}")
        results['METEOR'] = None

    # BERTScore (requires torch.frombuffer, not available in PyTorch 1.13)
    print("BERTScore skipped: PyTorch 1.13 lacks torch.frombuffer")
    results['BERTScore'] = None

    return results


def main(args):
    # Load all logs
    temporal_data = load_log(args.temporal_log)
    binary_data = load_log(args.binary_log)
    existence_data = load_log(args.existence_log)
    description_data = load_log(args.description_log)
    comprehensive_data = load_log(args.comprehensive_log)

    # Simple Tasks
    binary_acc = compute_accuracy(binary_data) if binary_data else 0.0
    existence_acc = compute_accuracy(existence_data) if existence_data else 0.0
    accuracy = (binary_acc + existence_acc) / 2 if (binary_data or existence_data) else 0.0

    temporal_metrics = compute_temporal_metrics(temporal_data)
    miou = temporal_metrics.get('mIoU', 0.0)

    # Complex Tasks
    caption_data = description_data + comprehensive_data
    caption_metrics = compute_caption_metrics(caption_data) if caption_data else {}

    # Print table
    print(f"\n{'='*80}")
    print(f"B4DL Evaluation Results: {args.model_name}")
    print(f"{'='*80}")
    print(f"{'Task':<30} {'Metric':<15} {'Value':<10}")
    print(f"{'-'*80}")
    print(f"{'Simple Tasks':<30} {'Accuracy':<15} {accuracy:.3f}")
    print(f"{'Simple Tasks':<30} {'mIoU':<15} {miou:.3f}")
    print(f"{'Complex Tasks':<30} {'B@4':<15} {caption_metrics.get('B@4', 0.0):.3f}")
    print(f"{'Complex Tasks':<30} {'ROUGE-L':<15} {caption_metrics.get('ROUGE-L', 0.0):.3f}")
    meteor_val = caption_metrics.get('METEOR')
    bert_val = caption_metrics.get('BERTScore')
    print(f"{'Complex Tasks':<30} {'METEOR':<15} {meteor_val:.3f}" if meteor_val is not None else f"{'Complex Tasks':<30} {'METEOR':<15} N/A")
    print(f"{'Complex Tasks':<30} {'BERTScore':<15} {bert_val:.3f}" if bert_val is not None else f"{'Complex Tasks':<30} {'BERTScore':<15} N/A")
    print(f"{'='*80}")

    # Print in paper table format
    print(f"\nPaper Table Format:")
    b4 = caption_metrics.get('B@4', 0.0)
    rouge = caption_metrics.get('ROUGE-L', 0.0)
    meteor = caption_metrics.get('METEOR')
    bert = caption_metrics.get('BERTScore')
    meteor_str = f"{meteor:.3f}" if meteor is not None else "N/A"
    bert_str = f"{bert:.3f}" if bert is not None else "N/A"
    print(f"{args.model_name} | {accuracy:.3f} | {miou:.3f} | {b4:.3f} | {rouge:.3f} | {meteor_str} | {bert_str}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

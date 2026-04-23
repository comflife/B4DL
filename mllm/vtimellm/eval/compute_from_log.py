import json
import argparse
import re
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Compute metrics from eval log JSONL")
    parser.add_argument("--log_path", type=str, required=True)
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


def main(args):
    data = []
    with open(args.log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Total samples: {len(data)}")

    # Time Grounding metrics
    tg_ious = []
    for item in data:
        gt_int = item.get('gt_interval')
        pred_int = item.get('pred_interval')
        if gt_int is not None:
            iou = compute_iou(pred_int, gt_int)
            tg_ious.append(iou)

    if tg_ious:
        print(f"\n=== Time Grounding ===")
        print(f"Samples: {len(tg_ious)}")
        print(f"mIoU: {sum(tg_ious)/len(tg_ious)*100:.2f}")
        for th in [0.3, 0.5, 0.7]:
            rec = sum(1 for i in tg_ious if i >= th) / len(tg_ious)
            print(f"R1@{th}: {rec*100:.2f}")

    # Split TG vs non-TG
    tg_data = [item for item in data if item.get('gt_interval') is not None]
    non_tg_data = [item for item in data if item.get('gt_interval') is None]

    print(f"\n=== Split ===")
    print(f"TG samples: {len(tg_data)}")
    print(f"Non-TG samples: {len(non_tg_data)}")

    # Non-TG Accuracy (exact match)
    if non_tg_data:
        correct = sum(1 for item in non_tg_data if normalize(item['gt_answer']) == normalize(item['pred_answer']))
        print(f"\n=== Non-TG Accuracy ===")
        print(f"Accuracy: {correct/len(non_tg_data)*100:.2f}% ({correct}/{len(non_tg_data)})")

    # Yes/No Accuracy (within non-TG)
    yn_gts = []
    yn_preds = []
    for item in non_tg_data:
        gt_norm = normalize(item['gt_answer'])
        pred_norm = normalize(item['pred_answer'])
        if gt_norm in ('yes', 'no'):
            yn_gts.append(gt_norm)
            yn_preds.append(pred_norm)
    if yn_gts:
        yn_correct = sum(1 for g, p in zip(yn_gts, yn_preds) if g == p)
        print(f"Yes/No Accuracy: {yn_correct/len(yn_gts)*100:.2f}% ({yn_correct}/{len(yn_gts)})")


if __name__ == "__main__":
    args = parse_args()
    main(args)

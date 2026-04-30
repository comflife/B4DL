"""Quick-look evaluator for a Q3D-trained Qwen3 + SMAP checkpoint.

Loads the SFT model, runs greedy generation on a held-out slice of the
quantised val set, and reports parse rate, BEV-IoU, centre L2, heading,
class-match (for det_object), and per-template breakdowns.

Intended as a smoke check between stage 1b and GRPO — *not* a full
benchmark. Use --max_samples to keep runtime sane on a single GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

NEW_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NEW_CODE_DIR))

from qwen_mm import MMQwen, IMAGE_PLACEHOLDER  # noqa: E402
from qwen_mm.quantizer import parse_quantized_boxes  # noqa: E402
from rewards_lidar import (  # noqa: E402
    bev_iou,
    box_center,
    parse_boxes,
    parse_class,
    NUSC_CLASSES,
)

# Supercategory grouping for ACC-5 (LiDAR-LLM Table 2 style):
#   vehicle      -> car, truck, bus, trailer, construction_vehicle
#   two_wheeler  -> bicycle, motorcycle
#   pedestrian   -> pedestrian
#   barrier      -> barrier
#   traffic_cone -> traffic_cone
SUPER5 = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "trailer": "vehicle",
    "construction_vehicle": "vehicle",
    "bicycle": "two_wheeler",
    "motorcycle": "two_wheeler",
    "pedestrian": "pedestrian",
    "barrier": "barrier",
    "traffic_cone": "traffic_cone",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_dir", required=True,
                    help="Stage-1b checkpoint dir (output_dir from stage1b_full.sh)")
    ap.add_argument("--lora_adapter", default=None,
                    help="Optional GRPO LoRA adapter dir (e.g., qwen_stage2_grpo_q3d/step_100). "
                    "Loaded on top of the SFT base via peft.")
    ap.add_argument("--data_path", default="/data1/byounggun/3davs_b4dl/data/3dtesting_val_q3d.json")
    ap.add_argument("--feat_folder", default="/data1/byounggun/3davs_b4dl/features/smap_lidar12")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--out_jsonl", default=None,
                    help="Optional path to dump per-sample (prompt, gt, pred) records.")
    return ap.parse_args()


def load_model(sft_dir: str, lora_adapter: str = None):
    tok = AutoTokenizer.from_pretrained(sft_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = MMQwen.from_pretrained(sft_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
    if lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_adapter)
        print(f"[eval] loaded LoRA adapter from {lora_adapter}")
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    img_id = tok.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if img_id == tok.unk_token_id:
        raise RuntimeError("'<image>' missing from tokenizer — wrong checkpoint?")
    # peft wraps the underlying MMQwen as `model.base_model.model`.
    try:
        from peft import PeftModel
        if isinstance(model, PeftModel):
            model.base_model.model.image_token_id = img_id
        else:
            model.image_token_id = img_id
    except ImportError:
        model.image_token_id = img_id
    return model, tok


def build_prompt(tok, conversations):
    """Apply Qwen ChatML on the user turn(s) only and request generation."""
    messages = []
    for turn in conversations[:-1]:
        role = "user" if turn["from"].lower() == "human" else "assistant"
        messages.append({"role": role, "content": turn["value"]})
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    return enc.input_ids, enc.attention_mask


def safe_load_feat(feat_dir: Path, scene_id: str):
    p = feat_dir / f"{scene_id}.pt"
    if not p.exists():
        return None
    blob = torch.load(p, map_location="cpu")
    feat = blob["output_smap"]
    if feat.dim() == 3:
        feat = feat.squeeze(0)
    return feat.float()


def per_pair_metrics(preds, gts):
    """Best-match centre L2 + IoU. Returns (matched_pairs, missed, extra)."""
    if not preds or not gts:
        return [], len(gts), len(preds)
    P, G = len(preds), len(gts)
    iou = np.zeros((P, G), dtype=np.float32)
    cdist = np.zeros((P, G), dtype=np.float32)
    for i in range(P):
        for j in range(G):
            iou[i, j] = bev_iou(preds[i], gts[j])
            cdist[i, j] = float(np.linalg.norm(box_center(preds[i]) - box_center(gts[j])))
    # Greedy by IoU desc, then centre asc.
    used_p, used_g = set(), set()
    pairs = []
    score = iou - 1e-3 * cdist  # tie-break with proximity
    while True:
        i, j = np.unravel_index(np.argmax(score), score.shape)
        if score[i, j] <= -1e8:
            break
        if i in used_p or j in used_g:
            score[i, j] = -1e9
            continue
        pairs.append((i, j, float(iou[i, j]), float(cdist[i, j])))
        used_p.add(i); used_g.add(j)
        score[i, :] = -1e9; score[:, j] = -1e9
        if len(pairs) == min(P, G):
            break
    return pairs, max(0, G - len(pairs)), max(0, P - len(pairs))


def main():
    args = parse_args()
    feat_dir = Path(args.feat_folder)
    print(f"[eval] sft_dir = {args.sft_dir}")
    print(f"[eval] data    = {args.data_path}")

    model, tok = load_model(args.sft_dir, args.lora_adapter)
    data = json.load(open(args.data_path))
    print(f"[eval] val n   = {len(data)} (using first {args.max_samples})")

    metrics_by_tmpl = defaultdict(lambda: {
        "n": 0, "parsed": 0, "ious": [], "cdists": [],
        "missed": 0, "extra": 0,
        "class_ok": 0, "class_total": 0,           # ACC-10 (exact 10-class)
        "class_super_ok": 0, "class_super_total": 0,  # ACC-5 (supercategory)
    })
    # Per-class BEV-IoU buckets — gt-class is parsed from the GT text so we
    # can report a Car-only mIoU comparable to LiDAR-LLM Table 2.
    iou_by_class = defaultdict(list)
    # Per-template-per-class so we can isolate visual-grounding-style Car
    # mIoU on det_area (real localisation) from det_object (echo).
    iou_by_tmpl_class = defaultdict(lambda: defaultdict(list))

    out_records = []
    skipped = 0
    for i, sample in enumerate(data[: args.max_samples]):
        scene_id = sample.get("scene_id") or sample.get("sample_token")
        feat = safe_load_feat(feat_dir, scene_id)
        if feat is None:
            skipped += 1
            continue

        first = sample["conversations"][0]["value"]
        if IMAGE_PLACEHOLDER not in first:
            sample["conversations"][0]["value"] = IMAGE_PLACEHOLDER + "\n" + first

        gt_text = sample["conversations"][-1]["value"]
        tmpl = sample.get("template_type") or "?"
        m = metrics_by_tmpl[tmpl]
        m["n"] += 1

        input_ids, attn = build_prompt(tok, sample["conversations"])
        device = next(model.parameters()).device
        with torch.no_grad():
            gen = model.generate(
                input_ids=input_ids.to(device),
                attention_mask=attn.to(device),
                images=[feat.to(device)],
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        pred_text = tok.decode(gen[0], skip_special_tokens=False)
        # When using inputs_embeds path, generate returns only the new tokens.
        # Strip any prompt copy if present.
        if "<|im_start|>assistant" in pred_text:
            pred_text = pred_text.split("<|im_start|>assistant", 1)[-1]

        preds = parse_boxes(pred_text)
        gts = parse_boxes(gt_text)
        pairs, missed, extra = per_pair_metrics(preds, gts)

        if preds:
            m["parsed"] += 1
        m["missed"] += missed
        m["extra"] += extra
        for _, _, iou, cd in pairs:
            m["ious"].append(iou); m["cdists"].append(cd)

        # Class match (top-1 across NUSC_CLASSES) and supercategory match.
        # Compute on every sample where GT has a class word — this includes
        # det_area like "The 2 cars are at ..." not just det_object.
        gc = parse_class(gt_text); pc = parse_class(pred_text)
        if gc:
            m["class_total"] += 1
            m["class_super_total"] += 1
            if pc == gc:
                m["class_ok"] += 1
            if pc and SUPER5.get(pc) == SUPER5.get(gc):
                m["class_super_ok"] += 1
            # Bucket each matched IoU by GT class so we can report a Car
            # mIoU directly comparable to the LiDAR-LLM paper's number.
            for _, _, iou, _ in pairs:
                iou_by_class[gc].append(iou)
                iou_by_tmpl_class[tmpl][gc].append(iou)

        if args.out_jsonl:
            out_records.append({
                "scene_id": scene_id,
                "template_type": tmpl,
                "gt": gt_text,
                "pred": pred_text,
                "n_pred": len(preds),
                "n_gt": len(gts),
                "matches": [{"iou": p[2], "cdist": p[3]} for p in pairs],
            })

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1} samples (skipped {skipped})")

    # ---- summary ------------------------------------------------------
    print("\n=== Q3D eval summary ===")
    print(f"skipped (missing feat) : {skipped}")
    for tmpl, m in metrics_by_tmpl.items():
        n = m["n"]
        if n == 0:
            continue
        ious = m["ious"]; cdists = m["cdists"]
        line = (
            f"[{tmpl:>10}] n={n:4d}  parse_rate={m['parsed']/n:.2%}  "
            f"matched={len(ious):4d}  iou_mean={(np.mean(ious) if ious else 0):.3f}  "
            f"iou_p50={(np.median(ious) if ious else 0):.3f}  "
            f"cd_mean={(np.mean(cdists) if cdists else 0):.2f}m  "
            f"missed={m['missed']}  extra={m['extra']}  "
            f"acc10={m['class_ok']/max(1, m['class_total']):.2%}  "
            f"acc5={m['class_super_ok']/max(1, m['class_super_total']):.2%}"
        )
        print(line)

    # ---- LiDAR-LLM-style aggregate metrics ----------------------------
    # Combined ACC-10 / ACC-5 across all templates that had a class word.
    all_class_ok = sum(m["class_ok"] for m in metrics_by_tmpl.values())
    all_class_total = sum(m["class_total"] for m in metrics_by_tmpl.values())
    all_super_ok = sum(m["class_super_ok"] for m in metrics_by_tmpl.values())
    all_super_total = sum(m["class_super_total"] for m in metrics_by_tmpl.values())
    all_ious = [iou for m in metrics_by_tmpl.values() for iou in m["ious"]]

    print("\n=== LiDAR-LLM comparable metrics ===")
    print(
        f"ACC-10 (exact)        : {all_class_ok/max(1, all_class_total):.2%}  "
        f"({all_class_ok}/{all_class_total})"
    )
    print(
        f"ACC-5  (supercategory): {all_super_ok/max(1, all_super_total):.2%}  "
        f"({all_super_ok}/{all_super_total})"
    )
    if all_ious:
        print(
            f"BEV mIoU (all matched): {np.mean(all_ious):.3f}  "
            f"(matched={len(all_ious)})"
        )
    print("BEV mIoU per class:")
    for cls in NUSC_CLASSES:
        ious_c = iou_by_class.get(cls, [])
        if ious_c:
            print(
                f"  {cls:>20s}: mean={np.mean(ious_c):.3f}  "
                f"p50={np.median(ious_c):.3f}  n={len(ious_c)}"
            )
    # det_area-only Car mIoU is the cleanest analogue of LiDAR-LLM's
    # Visual-Grounding number (the model has to localise from text, not
    # just echo the input box). We report both for transparency.
    car_all = iou_by_class.get("car", [])
    car_da = iou_by_tmpl_class.get("det_area", {}).get("car", [])
    car_do = iou_by_tmpl_class.get("det_object", {}).get("car", [])
    print("\nCar BEV mIoU breakdown (LiDAR-LLM Visual Grounding analogue):")
    if car_all:
        print(f"  all     : mean={np.mean(car_all)*100:5.2f}  n={len(car_all)}")
    if car_do:
        print(f"  det_obj : mean={np.mean(car_do)*100:5.2f}  n={len(car_do)}  (echo task)")
    if car_da:
        print(f"  det_area: mean={np.mean(car_da)*100:5.2f}  n={len(car_da)}  (true VG; LiDAR-LLM 14.3)")
    elif "car" in iou_by_class:
        print("  det_area: no det_area Car samples in this slice")

    if args.out_jsonl:
        Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_jsonl, "w") as f:
            for r in out_records:
                f.write(json.dumps(r) + "\n")
        print(f"[eval] wrote {len(out_records)} records to {args.out_jsonl}")


if __name__ == "__main__":
    main()

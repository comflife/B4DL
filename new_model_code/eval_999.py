"""Evaluator for a 999-bin Qwen3.5 + joint VoxelNeXt checkpoint.

Loads the SFT model, attaches the frozen VoxelNeXt encoder, runs greedy
generation on a held-out slice of the quantized val set, and reports
parse rate, BEV-IoU, centre L2, heading, class-match (for det_object),
and per-template breakdowns.

By default this is a smoke check between stage 1b and GRPO. Use
--max_samples 0 for a full validation pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer

NEW_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NEW_CODE_DIR))

from qwen_mm import MMQwen, IMAGE_PLACEHOLDER  # noqa: E402
from qwen_mm.data import load_keyframe_with_sweeps  # noqa: E402
from rewards_lidar import (  # noqa: E402
    bev_iou as axis_aligned_bev_iou,
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
    _data_root = os.environ.get("DATA_ROOT", "/data1/byounggun/3davs_b4dl")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sft_dir",
        "--model_path",
        dest="sft_dir",
        required=True,
        help="Stage checkpoint dir with model + tokenizer.",
    )
    ap.add_argument("--lora_adapter", default=None,
                    help="Optional GRPO LoRA adapter dir (e.g., qwen_stage2_grpo_vxnxt/step_100). "
                    "Loaded on top of the SFT base via peft.")
    ap.add_argument("--data_path", default=f"{_data_root}/data/3dtesting_val_999.json")
    ap.add_argument(
        "--nuscenes_root",
        default=os.environ.get("NUSCENES_ROOT", f"{_data_root}/nuscenes"),
    )
    ap.add_argument("--nuscenes_version", default="v1.0-trainval")
    ap.add_argument("--n_sweeps", type=int, default=10)
    ap.add_argument(
        "--voxelnext_root",
        default=os.environ.get(
            "VOXELNEXT_ROOT", f"{_data_root}/voxelnext_work/VoxelNeXt"
        ),
    )
    ap.add_argument(
        "--voxelnext_ckpt",
        default=os.environ.get(
            "VOXELNEXT_CKPT",
            f"{_data_root}/voxelnext_work/ckpt/voxelnext_nuscenes_kernel1.pth",
        ),
    )
    ap.add_argument("--voxelnext_top_k", type=int, default=256)
    ap.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Number of validation samples to evaluate. Use 0 or negative for all samples.",
    )
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument(
        "--bev_iou_mode",
        choices=("rotated", "axis_aligned"),
        default="rotated",
        help="BEV IoU for reporting. 'rotated' uses decoded center/size/yaw; "
        "'axis_aligned' matches the GRPO reward implementation.",
    )
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--out_jsonl", default=None,
                    help="Optional path to dump per-sample (prompt, gt, pred) records.")
    return ap.parse_args()


def load_model(
    sft_dir: str,
    lora_adapter: str = None,
    voxelnext_root: str = None,
    voxelnext_ckpt: str = None,
    voxelnext_top_k: int = 256,
):
    tok = AutoTokenizer.from_pretrained(sft_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = MMQwen.from_pretrained(sft_dir, dtype=torch.bfloat16, attn_implementation="sdpa")
    if torch.cuda.is_available():
        model.cuda()
    if voxelnext_root and voxelnext_ckpt:
        print(f"[eval] init VoxelNeXt: {voxelnext_ckpt} top_k={voxelnext_top_k}")
        model.init_voxelnext(
            voxelnext_root=voxelnext_root,
            ckpt_path=voxelnext_ckpt,
            top_k=voxelnext_top_k,
            freeze=True,
        )
    if lora_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_adapter)
        print(f"[eval] loaded LoRA adapter from {lora_adapter}")
    model.eval()
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


def safe_load_points(nusc, scene_id: str, n_sweeps: int = 10):
    try:
        pts_np = load_keyframe_with_sweeps(nusc, scene_id, n_sweeps)
    except Exception:
        return None
    if pts_np is None or pts_np.shape[0] == 0:
        return None
    return torch.from_numpy(pts_np)


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _poly_area(poly) -> float:
    if len(poly) < 3:
        return 0.0
    pts = np.asarray(poly, dtype=np.float64)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _rotated_bev_corners(box: np.ndarray):
    cx = float((box[0] + box[1]) * 0.5)
    cy = float((box[2] + box[3]) * 0.5)
    w = max(0.0, float(box[1] - box[0]))
    l = max(0.0, float(box[3] - box[2]))
    yaw = float(box[6])
    c, s = math.cos(yaw), math.sin(yaw)
    local = [(-w * 0.5, -l * 0.5), (w * 0.5, -l * 0.5),
             (w * 0.5, l * 0.5), (-w * 0.5, l * 0.5)]
    return [
        np.array([cx + x * c - y * s, cy + x * s + y * c], dtype=np.float64)
        for x, y in local
    ]


def _line_intersection(p1, p2, q1, q2):
    r = p2 - p1
    s = q2 - q1
    denom = _cross(r, s)
    if abs(denom) < 1e-9:
        return p2
    t = _cross(q1 - p1, s) / denom
    return p1 + t * r


def _clip_polygon(subject, clipper):
    """Sutherland-Hodgman clipping for two CCW convex polygons."""
    out = list(subject)
    for i in range(len(clipper)):
        cp1 = clipper[i]
        cp2 = clipper[(i + 1) % len(clipper)]
        inp = out
        out = []
        if not inp:
            break

        def inside(p):
            return _cross(cp2 - cp1, p - cp1) >= -1e-9

        s = inp[-1]
        for e in inp:
            if inside(e):
                if not inside(s):
                    out.append(_line_intersection(s, e, cp1, cp2))
                out.append(e)
            elif inside(s):
                out.append(_line_intersection(s, e, cp1, cp2))
            s = e
    return out


def rotated_bev_iou(a: np.ndarray, b: np.ndarray) -> float:
    area_a = max(0.0, float(a[1] - a[0])) * max(0.0, float(a[3] - a[2]))
    area_b = max(0.0, float(b[1] - b[0])) * max(0.0, float(b[3] - b[2]))
    if area_a <= 1e-9 or area_b <= 1e-9:
        return 0.0
    pa = _rotated_bev_corners(a)
    pb = _rotated_bev_corners(b)
    inter = _poly_area(_clip_polygon(pa, pb))
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return float(max(0.0, min(1.0, inter / union)))


def compute_bev_iou(a: np.ndarray, b: np.ndarray, mode: str) -> float:
    if mode == "axis_aligned":
        return axis_aligned_bev_iou(a, b)
    return rotated_bev_iou(a, b)


def per_pair_metrics(preds, gts, bev_iou_mode: str = "rotated"):
    """Best-match centre L2 + IoU. Returns (matched_pairs, missed, extra)."""
    if not preds or not gts:
        return [], len(gts), len(preds)
    P, G = len(preds), len(gts)
    iou = np.zeros((P, G), dtype=np.float32)
    cdist = np.zeros((P, G), dtype=np.float32)
    for i in range(P):
        for j in range(G):
            iou[i, j] = compute_bev_iou(preds[i], gts[j], bev_iou_mode)
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
    if args.output_dir and not args.out_jsonl:
        args.out_jsonl = str(Path(args.output_dir) / "predictions.jsonl")
    print(f"[eval] sft_dir = {args.sft_dir}")
    print(f"[eval] data    = {args.data_path}")
    print(f"[eval] nuscenes_root = {args.nuscenes_root}")
    print(f"[eval] bev_iou_mode = {args.bev_iou_mode}")

    model, tok = load_model(
        args.sft_dir,
        args.lora_adapter,
        voxelnext_root=args.voxelnext_root,
        voxelnext_ckpt=args.voxelnext_ckpt,
        voxelnext_top_k=args.voxelnext_top_k,
    )
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(
        version=args.nuscenes_version, dataroot=args.nuscenes_root, verbose=False
    )
    data = json.load(open(args.data_path))
    eval_data = data if args.max_samples <= 0 else data[: args.max_samples]
    print(f"[eval] val n   = {len(data)} (using {len(eval_data)})")

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
    table6_by_class = defaultdict(lambda: {"ok": 0, "total": 0, "super_ok": 0})
    table7_iou_by_class = defaultdict(list)

    out_records = []
    skipped = 0
    for i, sample in enumerate(eval_data):
        scene_id = sample.get("sample_token") or sample.get("scene_id")
        points = safe_load_points(nusc, scene_id, args.n_sweeps)
        if points is None:
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
                points=[points.to(device)],
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
        pairs, missed, extra = per_pair_metrics(preds, gts, args.bev_iou_mode)

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

            if tmpl == "det_object":
                table6_by_class[gc]["total"] += 1
                if pc == gc:
                    table6_by_class[gc]["ok"] += 1
                if pc and SUPER5.get(pc) == SUPER5.get(gc):
                    table6_by_class[gc]["super_ok"] += 1

            if tmpl == "det_area" and gts:
                matched_by_gt = {gt_idx: iou for _, gt_idx, iou, _ in pairs}
                for gt_idx in range(len(gts)):
                    table7_iou_by_class[gc].append(float(matched_by_gt.get(gt_idx, 0.0)))

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
    print("\n=== 999-bin eval summary ===")
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
    # Combined exact / ACC-5 across all templates that had a class word.
    all_class_ok = sum(m["class_ok"] for m in metrics_by_tmpl.values())
    all_class_total = sum(m["class_total"] for m in metrics_by_tmpl.values())
    all_super_ok = sum(m["class_super_ok"] for m in metrics_by_tmpl.values())
    all_super_total = sum(m["class_super_total"] for m in metrics_by_tmpl.values())
    all_ious = [iou for m in metrics_by_tmpl.values() for iou in m["ious"]]

    print("\n=== LiDAR-LLM-style aggregate metrics ===")
    print(
        f"ACC-{len(NUSC_CLASSES):02d} (exact, all templates): "
        f"{all_class_ok/max(1, all_class_total):.2%}  "
        f"({all_class_ok}/{all_class_total})"
    )
    print(
        f"ACC-5  (supercategory): {all_super_ok/max(1, all_super_total):.2%}  "
        f"({all_super_ok}/{all_super_total})"
    )
    if all_ious:
        print(
            f"BEV mIoU ({args.bev_iou_mode}, all matched): {np.mean(all_ious):.3f}  "
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

    # ---- LiDAR-LLM Table 6 ------------------------------------------------
    # Grounded Captioning: prompt gives a bbox; answer should name the object.
    print("\n=== LiDAR-LLM Table 6: Grounded Captioning (det_object) ===")
    print("Protocol: bbox is given in the prompt; score is top-1 class accuracy.")
    table6_total = sum(v["total"] for v in table6_by_class.values())
    table6_ok = sum(v["ok"] for v in table6_by_class.values())
    table6_super_ok = sum(v["super_ok"] for v in table6_by_class.values())
    if len(NUSC_CLASSES) == 19:
        print(f"ACC-19: {table6_ok/max(1, table6_total):.2%}  ({table6_ok}/{table6_total})")
    else:
        print(
            "ACC-19: N/A "
            f"(current eval label space is nuScenes {len(NUSC_CLASSES)} classes)"
        )
        print(
            f"ACC-{len(NUSC_CLASSES):02d}: {table6_ok/max(1, table6_total):.2%}  "
            f"({table6_ok}/{table6_total})"
        )
    print(
        f"ACC-5 : {table6_super_ok/max(1, table6_total):.2%}  "
        f"({table6_super_ok}/{table6_total})"
    )
    print("Per-class top-1 accuracy:")
    for cls in NUSC_CLASSES:
        stat = table6_by_class.get(cls)
        if stat and stat["total"]:
            print(
                f"  {cls:>22s}: {stat['ok']/stat['total']:.2%}  "
                f"({stat['ok']}/{stat['total']})"
            )

    # ---- LiDAR-LLM Table 7 (Appendix A.2) reproduction -------------------
    # 5-category Visual-Grounding mIoU. Only det_area samples count as a
    # true VG eval — det_object echoes the input box and inflates numbers.
    print("\n=== LiDAR-LLM Table 7: Visual Grounding (det_area only, 5-class BEV mIoU) ===")
    print(f"Protocol: predicted boxes are matched to GT boxes; missed GT boxes count as IoU=0. BEV IoU mode={args.bev_iou_mode}.")
    print("    (paper: Car 11.94  Pedestrian 9.05  Bus 11.23  Truck 8.09  Construction_vehicle 9.40)")
    LLM_TABLE7 = ["car", "pedestrian", "bus", "truck", "construction_vehicle"]
    means = []
    for cls in LLM_TABLE7:
        ious_c = table7_iou_by_class.get(cls, [])
        if ious_c:
            m = float(np.mean(ious_c)) * 100
            means.append(m)
            print(f"  {cls:>22s}: mIoU={m:5.2f}  n_gt={len(ious_c)}")
        else:
            print(f"  {cls:>22s}: (no det_area samples)")
    if means:
        print(f"  {'mean(5-class)':>22s}: {float(np.mean(means)):5.2f}")

    if args.out_jsonl:
        Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_jsonl, "w") as f:
            for r in out_records:
                f.write(json.dumps(r) + "\n")
        print(f"[eval] wrote {len(out_records)} records to {args.out_jsonl}")


if __name__ == "__main__":
    main()

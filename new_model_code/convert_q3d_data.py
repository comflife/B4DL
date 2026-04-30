"""Convert the existing combined Stage-1 JSON into Q3D-quantised form.

Input JSON entries look like (after build_combined_stage1.py):

    {
      "scene_id": ...,
      "task": "nucaption" | "nugrounding",
      "template_type": "det_object" | "det_area" | None,
      "conversations": [
        {"from": "human", "value": "<image>\n... <|box_start|>[1.2,3.4,...]<|box_end|> ..."},
        {"from": "gpt",   "value": "... <|box_start|>[...]<|box_end|> ..."},
      ],
    }

We rewrite every `<|box_start|>[xmin,xmax,ymin,ymax,zmin,zmax,yaw]<|box_end|>`
span into

    `<|box_start|><coord_a><coord_b><coord_c><coord_d><coord_e><coord_f><coord_g><coord_h><|box_end|>`

and leave the rest of the prose untouched. nuCaption rows are passed through
unchanged (no boxes there).

Run-once script. We also dump a small JSONL of conversion stats so we can
sanity-check the empirical quantisation error.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

NEW_CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NEW_CODE_DIR))

from qwen_mm.quantizer import (  # noqa: E402
    encode_box_indices,
    decode_box_indices,
    encode_box_to_text,
)

BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"

# Match a wrapped 7-tuple: <|box_start|>[..., ..., ..., ..., ..., ..., ...]<|box_end|>
_WRAPPED_BBOX_RE = re.compile(
    re.escape(BOX_START)
    + r"\s*\["
    + r"\s*(-?\d+\.?\d*)\s*,"  # xmin
    + r"\s*(-?\d+\.?\d*)\s*,"  # xmax
    + r"\s*(-?\d+\.?\d*)\s*,"  # ymin
    + r"\s*(-?\d+\.?\d*)\s*,"  # ymax
    + r"\s*(-?\d+\.?\d*)\s*,"  # zmin
    + r"\s*(-?\d+\.?\d*)\s*,"  # zmax
    + r"\s*(-?\d+\.?\d*)\s*"   # yaw
    + r"\]\s*"
    + re.escape(BOX_END)
)


def _convert_match(m: re.Match) -> str:
    box7 = [float(g) for g in m.groups()]
    return encode_box_to_text(box7, BOX_START, BOX_END)


def convert_value(text: str, stats: dict) -> str:
    """Replace all wrapped 7-tuples in `text` with their quantised form."""
    if not text or BOX_START not in text:
        return text
    n_before = stats["boxes"]
    out = _WRAPPED_BBOX_RE.sub(_convert_match, text)
    # Track per-axis quantisation error for monitoring.
    for m in _WRAPPED_BBOX_RE.finditer(text):
        box7 = [float(g) for g in m.groups()]
        idxs = encode_box_indices(box7)
        dec = decode_box_indices(idxs)
        cx_o = 0.5 * (box7[0] + box7[1]); cy_o = 0.5 * (box7[2] + box7[3]); cz_o = 0.5 * (box7[4] + box7[5])
        w_o = box7[1] - box7[0]; l_o = box7[3] - box7[2]; h_o = box7[5] - box7[4]
        cx_d = 0.5 * (dec[0] + dec[1]); cy_d = 0.5 * (dec[2] + dec[3]); cz_d = 0.5 * (dec[4] + dec[5])
        w_d = dec[1] - dec[0]; l_d = dec[3] - dec[2]; h_d = dec[5] - dec[4]
        yaw_d = dec[6]
        d2 = math.hypot(cx_o, cy_o)
        bucket = "near" if d2 < 20 else ("mid" if d2 < 50 else "far")
        stats["err_centre_" + bucket].append(math.hypot(cx_d - cx_o, cy_d - cy_o))
        stats["err_size"].append(abs(w_d - w_o) + abs(l_d - l_o) + abs(h_d - h_o))
        diff = box7[6] - yaw_d
        diff = math.atan2(math.sin(diff), math.cos(diff))
        stats["err_yaw"].append(abs(diff))
        stats["boxes"] += 1
    if stats["boxes"] > n_before:
        stats["spans_replaced"] += stats["boxes"] - n_before
    return out


def transform_sample(sample: dict, stats: dict) -> dict:
    for turn in sample.get("conversations", []):
        v = turn.get("value")
        if v:
            turn["value"] = convert_value(v, stats)
    return sample


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_path",
        default="/data1/byounggun/3davs_b4dl/data/stage1_combined.json",
    )
    ap.add_argument(
        "--out_path",
        default="/data1/byounggun/3davs_b4dl/data/stage1_combined_q3d.json",
    )
    return ap.parse_args()


def _summary(arr):
    if not arr:
        return "n=0"
    import statistics
    return f"n={len(arr)} mean={statistics.fmean(arr):.4f} p50={statistics.median(arr):.4f} p95={sorted(arr)[int(0.95*len(arr))-1]:.4f}"


def main():
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {in_path}")
    data = json.load(open(in_path))
    print(f"[load] {len(data)} samples")

    stats = {
        "boxes": 0,
        "spans_replaced": 0,
        "err_centre_near": [],
        "err_centre_mid": [],
        "err_centre_far": [],
        "err_size": [],
        "err_yaw": [],
    }

    for i, x in enumerate(data):
        transform_sample(x, stats)
        if (i + 1) % 50000 == 0:
            print(f"  ... {i+1} samples, boxes={stats['boxes']}")

    print(f"[done] total samples = {len(data)}")
    print(f"[done] total boxes   = {stats['boxes']}")
    print("[err]  centre near  ", _summary(stats["err_centre_near"]))
    print("[err]  centre mid   ", _summary(stats["err_centre_mid"]))
    print("[err]  centre far   ", _summary(stats["err_centre_far"]))
    print("[err]  size  (sum3) ", _summary(stats["err_size"]))
    print("[err]  yaw   (rad)  ", _summary(stats["err_yaw"]))

    print(f"[write] {out_path}")
    with open(out_path, "w") as f:
        json.dump(data, f)
    sz = out_path.stat().st_size / (1024 * 1024)
    print(f"[done] wrote {sz:.1f} MB")


if __name__ == "__main__":
    main()

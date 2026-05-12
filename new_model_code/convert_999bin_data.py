"""Convert raw-metres JSON into 0-999 linear-quantised token form.

Input:  stage1_combined.json (or 3dtesting_train.json / val.json)
Output: stage1_combined_999.json

Replaces every <|box_start|>[xmin,xmax,ymin,ymax,zmin,zmax,yaw]<|box_end|>
with   <|box_start|><cx><cy><cz><w><l><h><yaw><|box_end|>
where each <i> is a 0-999 coordinate token.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Avoid importing qwen_mm package (pulls in transformers, torch, etc.)
QUANTIZER_PATH = Path(__file__).resolve().parent / "qwen_mm" / "quantizer_999.py"
_spec = __import__("importlib.util").util.spec_from_file_location("quantizer_999", QUANTIZER_PATH)
_quantizer = __import__("importlib.util").util.module_from_spec(_spec)
_spec.loader.exec_module(_quantizer)
encode_box_to_text = _quantizer.encode_box_to_text

BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"

# Match a wrapped 7-tuple: <|box_start|>[..., ..., ..., ..., ..., ..., ...]<|box_end|>
_WRAPPED_BBOX_RE = re.compile(
    re.escape(BOX_START)
    + r"\s*\["
    + r"\s*(-?\d+\.?\d*)\s*,"
    + r"\s*(-?\d+\.?\d*)\s*,"
    + r"\s*(-?\d+\.?\d*)\s*,"
    + r"\s*(-?\d+\.?\d*)\s*,"
    + r"\s*(-?\d+\.?\d*)\s*,"
    + r"\s*(-?\d+\.?\d*)\s*,"
    + r"\s*(-?\d+\.?\d*)\s*"
    + r"\]\s*"
    + re.escape(BOX_END)
)


def _convert_match(m: re.Match) -> str:
    box7 = [float(g) for g in m.groups()]
    return encode_box_to_text(box7)


def convert_value(text: str) -> str:
    if not text or BOX_START not in text:
        return text
    return _WRAPPED_BBOX_RE.sub(_convert_match, text)


def transform_sample(sample: dict) -> dict:
    for turn in sample.get("conversations", []):
        v = turn.get("value")
        if v:
            turn["value"] = convert_value(v)
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {in_path}")
    data = json.load(open(in_path))
    print(f"[load] {len(data)} samples")

    out = [transform_sample(s) for s in data]

    print(f"[write] {out_path}")
    with open(out_path, "w") as f:
        json.dump(out, f)
    sz = out_path.stat().st_size / (1024 * 1024)
    print(f"[done] wrote {sz:.1f} MB")


if __name__ == "__main__":
    main()

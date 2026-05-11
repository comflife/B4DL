"""Build sample_tokens_union.json — the union of all nuScenes sample_tokens
referenced by the train + GRPO + eval JSONs. extract_voxelnext_features.py
only extracts features for tokens in this list, so we don't waste time on
samples that never appear in training.

Defaults read from $DATA_ROOT and $REPO_ROOT (set via ~/.bashrc.b4dl).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def collect_tokens(json_path: Path) -> set[str]:
    if not json_path.exists():
        print(f"[warn] missing: {json_path}", file=sys.stderr)
        return set()
    with open(json_path) as f:
        data = json.load(f)
    out = set()
    for e in data:
        tok = e.get("sample_token") or e.get("scene_id")
        if tok:
            out.add(tok)
    print(f"  {json_path.name}: {len(data)} entries, {len(out)} unique tokens")
    return out


def main():
    data_root = os.environ.get("DATA_ROOT", "/data1/byounggun/3davs_b4dl")
    repo_root = os.environ.get(
        "REPO_ROOT",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            f"{data_root}/data/stage1_combined_q3d.json",
            f"{data_root}/data/3dtesting_train_q3d.json",
            f"{data_root}/data/3dtesting_val_q3d.json",
        ],
        help="JSON files to scan for sample_token/scene_id",
    )
    ap.add_argument(
        "--out",
        default=f"{repo_root}/new_model_code/sample_tokens_union.json",
    )
    args = ap.parse_args()

    union: set[str] = set()
    for p in args.inputs:
        union |= collect_tokens(Path(p))

    sorted_tokens = sorted(union)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(sorted_tokens, f)
    print(f"\ntotal unique tokens: {len(sorted_tokens)} -> {args.out}")


if __name__ == "__main__":
    main()

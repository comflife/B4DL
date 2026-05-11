"""Convert Senqiao/LiDAR-LLM-Nu-Caption raw JSON to the ChatML format that
build_combined_stage1.py expects. Mirrors REPRODUCE.md §2.2.

Input fields per entry:
  split, sample_token, question, answer, answer_lidar

Output fields per entry:
  scene_id, sample_token, split, conversations=[{human, gpt}]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def to_chatml(entry: dict) -> dict:
    return {
        "scene_id": entry["sample_token"],
        "sample_token": entry["sample_token"],
        "split": entry["split"],
        "conversations": [
            {"from": "human", "value": f"<image>\n{entry['question']}"},
            {"from": "gpt", "value": entry["answer"]},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="dir holding the raw Senqiao/LiDAR-LLM-Nu-Caption {train,val}.json",
    )
    ap.add_argument(
        "--out_dir",
        default=os.path.dirname(os.path.abspath(__file__)),
    )
    args = ap.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        src_path = src / f"{split}.json"
        if not src_path.exists():
            print(f"[skip] missing {src_path}")
            continue
        data = json.load(open(src_path))
        converted = [to_chatml(e) for e in data]
        out_path = out / f"stage1_{split}_converted.json"
        with open(out_path, "w") as f:
            json.dump(converted, f)
        print(f"[done] {len(converted):>7} -> {out_path}")


if __name__ == "__main__":
    main()

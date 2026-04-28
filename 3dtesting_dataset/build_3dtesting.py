"""Download LiDAR-LLM-Nu-Grounding from HuggingFace and convert to vtimellm stage2 format.

Outputs:
    3dtesting_dataset/raw/LiDAR-LLM-Nu-Grounding-train.json   (raw download)
    3dtesting_dataset/raw/LiDAR-LLM-Nu-Grounding-val.json     (raw download)
    3dtesting_dataset/train.json                              (converted, ready for train.py)
    3dtesting_dataset/val.json                                (converted)

Each converted record:
    {
        "scene_id": "<sample_token>",
        "sample_token": "<sample_token>",
        "split": "train" | "val",
        "view": "front" | ...,
        "template_type": "det_object" | "det_area",
        "conversations": [
            {"from": "human", "value": "<image>\n[<view> view] <question>"},
            {"from": "gpt",   "value": "<answer>"}
        ]
    }

scene_id is the nuScenes sample_token; features are loaded from
    {feat_folder}/{sample_token}.npy   (shape (1, 768)).

Run from anywhere:
    python 3dtesting_dataset/build_3dtesting.py \
        --feat_folder /home/byounggun/B4DL/lidarclip/stage1_features
"""

import argparse
import json
import os
import sys
import urllib.request

HF_BASE = "https://huggingface.co/datasets/Senqiao/LiDAR-LLM-Nu-Grounding/resolve/main"
FILES = {
    "train": "LiDAR-LLM-Nu-Grounding-train.json",
    "val": "LiDAR-LLM-Nu-Grounding-val.json",
}


def download(url, dst):
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"[skip] {dst} already exists ({os.path.getsize(dst)/1e6:.1f} MB)")
        return
    print(f"[download] {url} -> {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"

    def _hook(blocks, bs, total):
        if total <= 0:
            return
        done = blocks * bs
        pct = min(100, done * 100 // total)
        sys.stdout.write(f"\r  {pct:3d}%  {done/1e6:.1f}/{total/1e6:.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, _hook)
    sys.stdout.write("\n")
    os.replace(tmp, dst)


def convert(records, split, feat_dir):
    """Convert raw nu-grounding records to vtimellm conversation format.
    Filters out samples whose feature .npy is missing.
    """
    out = []
    skipped_no_feat = 0
    for r in records:
        sample_token = r["sample_token"]
        if feat_dir is not None:
            if not os.path.isfile(os.path.join(feat_dir, f"{sample_token}.npy")):
                skipped_no_feat += 1
                continue

        view = r.get("view", "")
        question = r["question"].strip()
        answer = r["answer"].strip()
        view_tag = f"[{view} view] " if view else ""

        out.append({
            "scene_id": sample_token,
            "sample_token": sample_token,
            "split": split,
            "view": view,
            "template_type": r.get("template_type", ""),
            "conversations": [
                {"from": "human", "value": f"<image>\n{view_tag}{question}"},
                {"from": "gpt",   "value": answer},
            ],
        })
    return out, skipped_no_feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--feat_folder", default="/home/byounggun/B4DL/lidarclip/stage1_features",
                        help="If set, drop records whose feature .npy is missing. Set to '' to skip filter.")
    args = parser.parse_args()

    raw_dir = os.path.join(args.out_dir, "raw")
    feat_dir = args.feat_folder if args.feat_folder else None

    for split, fname in FILES.items():
        raw_path = os.path.join(raw_dir, fname)
        download(f"{HF_BASE}/{fname}", raw_path)

        with open(raw_path, "r") as f:
            records = json.load(f)
        print(f"[{split}] loaded {len(records):,} raw records")

        converted, skipped = convert(records, split, feat_dir)
        print(f"[{split}] kept {len(converted):,} (skipped {skipped:,} missing-feature)")

        out_path = os.path.join(args.out_dir, f"{split}.json")
        with open(out_path, "w") as f:
            json.dump(converted, f, ensure_ascii=False)
        print(f"[{split}] wrote -> {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()

"""Build sample_token -> LiDAR_TOP filename index from nuScenes metadata.

Reads {nuscenes_root}/v1.0-trainval/sample_data.json (and v1.0-test) and
emits a single JSON mapping every sample_token whose sensor is LIDAR_TOP
to its relative file path.

Output: 3dtesting_dataset/sample_token_to_lidar.json

Usage:
    python 3dtesting_dataset/build_pc_index.py
"""
import argparse
import json
import os


def collect(json_path):
    with open(json_path, "r") as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        if "LIDAR_TOP" in r.get("filename", "") and r.get("is_key_frame", True):
            out[r["sample_token"]] = r["filename"]
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nuscenes_root", default="/home/byounggun/B4DL/nuscenes")
    parser.add_argument("--out", default="/home/byounggun/B4DL/3dtesting_dataset/sample_token_to_lidar.json")
    args = parser.parse_args()

    merged = {}
    for split_dir in ["v1.0-trainval", "v1.0-test"]:
        sd_path = os.path.join(args.nuscenes_root, split_dir, "sample_data.json")
        if os.path.isfile(sd_path):
            part = collect(sd_path)
            merged.update(part)
            print(f"[{split_dir}] {len(part):,} keyframes")

    with open(args.out, "w") as f:
        json.dump(merged, f)
    print(f"Total: {len(merged):,} sample_tokens -> {args.out}")


if __name__ == "__main__":
    main()

"""Combine nuCaption + nuGrounding for Stage-1 SFT, and wrap every 7-DOF bbox
with explicit <|box_start|> / <|box_end|> span markers (Rex-Omni style, but
keeping coordinates as plain text rather than quantising into 1k-token vocab —
chosen to avoid a massive tokenizer surgery while still giving the model a
crisp learning signal for "this is a box").

Wrapping happens on the *answer side* for nuGrounding samples (det_area and
det_object), and also on the *question side* for det_object queries that echo
a bbox into the prompt (so the model can attend to the box span as one unit).

Coordinate format itself stays "[xmin, xmax, ymin, ymax, zmin, zmax, yaw]" —
that's what the dataset already uses; we don't reformat into center-form.
"""
import argparse
import json
import re
from pathlib import Path

# 7-tuple bracket. We do NOT match nested "[[...]]" as a single hit — we want
# to wrap each inner tuple so multi-box answers like
#   "at [[8.4,...],[-0.5,...]]"
# become
#   "at [<|box_start|>[8.4,...]<|box_end|>,<|box_start|>[-0.5,...]<|box_end|>]"
_BBOX_RE = re.compile(
    r"\[\s*"
    r"(?:-?\d+\.?\d*\s*,\s*){6}"
    r"-?\d+\.?\d*\s*"
    r"\]"
)

BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"


def wrap_boxes(text: str) -> str:
    """Wrap every 7-element bracket span with start/end markers."""
    if not text or "[" not in text:
        return text
    # Skip already-wrapped spans (idempotent).
    if BOX_START in text:
        return text
    return _BBOX_RE.sub(lambda m: f"{BOX_START}{m.group(0)}{BOX_END}", text)


def transform_sample(sample: dict) -> dict:
    """In-place wrap of every conversation value that contains a 7-tuple bbox."""
    for turn in sample.get("conversations", []):
        v = turn.get("value")
        if v:
            turn["value"] = wrap_boxes(v)
    return sample


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nucaption",
        default="/home/byounggun/B4DL/lidarllm_only_dataset/stage1_train_converted.json",
    )
    ap.add_argument(
        "--nugrounding",
        default="/home/byounggun/B4DL/3dtesting_dataset/train.json",
    )
    ap.add_argument(
        "--out",
        default="/data1/byounggun/3davs_b4dl/data/stage1_combined.json",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    cap = json.load(open(args.nucaption))
    grd = json.load(open(args.nugrounding))
    print(f"[info] nuCaption  = {len(cap):>7} samples")
    print(f"[info] nuGrounding= {len(grd):>7} samples")

    n_wrapped = 0
    for x in cap:
        x.setdefault("task", "nucaption")
        # nuCaption has no boxes; transform is a no-op but cheap.
        transform_sample(x)
    for x in grd:
        x.setdefault("task", "nugrounding")
        before = json.dumps(x["conversations"])
        transform_sample(x)
        if BOX_START in json.dumps(x["conversations"]):
            n_wrapped += 1
        del before

    print(f"[info] wrapped {n_wrapped} nuGrounding samples with box span tokens")
    merged = cap + grd
    print(f"[info] merged    = {len(merged):>7} samples")
    with open(args.out, "w") as f:
        json.dump(merged, f)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()

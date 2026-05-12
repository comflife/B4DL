"""Simple 0-999 linear quantizer for 3D bbox coordinates.

Replaces the complex Q3D codebook with uniform 1000-bin quantization per axis.
All coordinates are quantized to integers in [0, 999] and represented as
special tokens <0>, <1>, ..., <999>.

Box format:
    <|box_start|><cx><cy><cz><w><l><h><yaw><|box_end|>

where each <> is a coordinate token (0-999).
"""
from __future__ import annotations

import math
import re
from typing import List, Sequence, Tuple

# Axis ranges (metres / radians)
_RANGES = {
    "cx": (-85.0, 85.0),
    "cy": (-85.0, 85.0),
    "cz": (-5.0, 5.0),
    "w": (0.0, 5.0),
    "l": (0.0, 20.0),
    "h": (0.0, 5.0),
    "yaw": (-math.pi, math.pi),
}

_AXIS_NAMES = ["cx", "cy", "cz", "w", "l", "h", "yaw"]
N_BINS = 1000

BOX_START = "<|box_start|>"
BOX_END = "<|box_end|>"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _quantize(v: float, lo: float, hi: float) -> int:
    """Map continuous value to 0-999 integer."""
    v = _clamp(v, lo, hi)
    if hi == lo:
        return 0
    return int(round((v - lo) / (hi - lo) * (N_BINS - 1)))


def _dequantize(idx: int, lo: float, hi: float) -> float:
    """Map 0-999 integer back to continuous value."""
    idx = max(0, min(N_BINS - 1, idx))
    return lo + idx / (N_BINS - 1) * (hi - lo)


def encode_box(box7: Sequence[float]) -> List[int]:
    """Encode [xmin, xmax, ymin, ymax, zmin, zmax, yaw] -> 7 ints in [0, 999]."""
    xmin, xmax, ymin, ymax, zmin, zmax, yaw = box7
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5
    cz = (zmin + zmax) * 0.5
    w = xmax - xmin
    l = ymax - ymin
    h = zmax - zmin
    vals = [cx, cy, cz, w, l, h, yaw]
    idxs = []
    for name, v in zip(_AXIS_NAMES, vals):
        lo, hi = _RANGES[name]
        idxs.append(_quantize(v, lo, hi))
    return idxs


def decode_box(idxs: Sequence[int]) -> List[float]:
    """Decode 7 ints -> [xmin, xmax, ymin, ymax, zmin, zmax, yaw]."""
    if len(idxs) != 7:
        raise ValueError(f"Need 7 indices, got {len(idxs)}")
    vals = []
    for name, idx in zip(_AXIS_NAMES, idxs):
        lo, hi = _RANGES[name]
        vals.append(_dequantize(idx, lo, hi))
    cx, cy, cz, w, l, h, yaw = vals
    return [
        cx - 0.5 * w,
        cx + 0.5 * w,
        cy - 0.5 * l,
        cy + 0.5 * l,
        cz - 0.5 * h,
        cz + 0.5 * h,
        yaw,
    ]


def encode_box_to_text(box7: Sequence[float]) -> str:
    idxs = encode_box(box7)
    body = "".join(f"<{i}>" for i in idxs)
    return f"{BOX_START}{body}{BOX_END}"


# Regex to find <0> ... <999> tokens between box markers
_QBOX_RE = re.compile(
    re.escape(BOX_START)
    + r"" + r"(<(\d{1,3})>)" * 7
    + re.escape(BOX_END)
)


def parse_999_boxes(text: str) -> List[List[float]]:
    """Find all <|box_start|><i0>...<i6><|box_end|> and decode to boxes."""
    out = []
    for m in _QBOX_RE.finditer(text):
        try:
            idxs = [int(m.group(i + 1)) for i in range(7)]
            out.append(decode_box(idxs))
        except Exception:
            continue
    return out

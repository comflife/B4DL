"""Q3D — discrete coord codebook for nuGrounding boxes.

A 7-DOF AABB+yaw box `[xmin, xmax, ymin, ymax, zmin, zmax, yaw]` is rewritten
as eight tokens drawn from a shared 1024-entry coord codebook:

    <|box_start|><X><Y><Z><W><L><H><SIN><COS><|box_end|>

Per-axis affine schemes (1024 bins each):

  X, Y (centre, metres):  distance-aware piecewise linear
      800 bins on [-20, 20] m at 0.05 m   (near-field detail)
      224 bins on [-80, -20] ∪ [20, 80] m at ~0.536 m  (far-field coverage)

  Z (centre, metres):     linear on [-3, 3] m                (~5.9 mm bin)

  W, L, H (AABB extent):  log-space on [0.1, 20] m           (~0.5% rel. err)

  SIN, COS (yaw):         linear on [-1, 1]                  (~0.2% bin)

Why this layout
---------------
* Distance-aware xy gives 4 cm precision where most cars/peds live (<20 m)
  and still keeps coverage to 80 m. Uniform 1024-bin xy across [-80, 80]
  would give a 16 cm bin everywhere and waste resolution on far/empty
  cells.
* Log-space sizes treat a 0.5 m pedestrian and a 15 m trailer with the
  same *relative* precision instead of letting the small classes vanish
  into a 8 cm uniform bin.
* sin/cos yaw avoids the cyclic discontinuity that an angle-token
  scheme would have at ±π — the model never has to learn that `<-3.14>`
  and `<+3.14>` are the same direction.

Aux-loss support
----------------
For the per-axis L1 loss in `MMQwen.forward` we expose
`bin_value_table` — a (8, 1024) tensor giving the metre / unit value of
each codebook entry, in the order [X, Y, Z, W, L, H, SIN, COS]. The
expected coord is then `softmax(logits[..., coord_range]) @ bin_values[axis]`
which is fully differentiable.

Conventions
-----------
* AABB extents (W, L, H) are kept axis-aligned, matching the dataset
  format. We do *not* try to reconstruct oriented (length, width) from
  yaw — that's ambiguous and lossy.
* `encode_box` and `decode_tokens` round-trip with at most one bin of
  error per axis (which is the whole point — the quantiser is the
  bottleneck, the model just learns to pick bins).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Public constants.
# ---------------------------------------------------------------------------
N_BINS = 1024            # codebook size — number of <coord_*> tokens
N_COORD_PER_BOX = 8      # X, Y, Z, W, L, H, SIN, COS

AXIS_NAMES = ("X", "Y", "Z", "W", "L", "H", "SIN", "COS")
# Axis ids match position-within-box (0..7). Used by the aux-loss in
# MMQwen.forward to look up bin_value_table[axis_id].
AXIS_X, AXIS_Y, AXIS_Z, AXIS_W, AXIS_L, AXIS_H, AXIS_SIN, AXIS_COS = range(8)

COORD_TOKEN_PREFIX = "<coord_"
COORD_TOKEN_SUFFIX = ">"

# ---------------------------------------------------------------------------
# Distance-aware XY scheme (shared by X and Y).
# ---------------------------------------------------------------------------
_XY_NEAR_RANGE = 20.0    # |c| < 20 m  -> "near" zone
_XY_FAR_RANGE = 80.0     # |c| in [20, 80] m -> "far" zone (clamped beyond)
_XY_NEAR_BINS = 800      # 40 m / 0.05 m  -> 0.05 m resolution near the ego
_XY_FAR_HALF_BINS = (N_BINS - _XY_NEAR_BINS) // 2  # 112 each side
_XY_NEAR_BIN = (2.0 * _XY_NEAR_RANGE) / _XY_NEAR_BINS                 # 0.05 m
_XY_FAR_BIN = (_XY_FAR_RANGE - _XY_NEAR_RANGE) / _XY_FAR_HALF_BINS    # ~0.536 m

assert _XY_NEAR_BINS + 2 * _XY_FAR_HALF_BINS == N_BINS, (
    "XY bin allocation must sum to N_BINS"
)

# Index layout:
#   [0,                        _XY_NEAR_BINS)                near zone (-20..+20)
#   [_XY_NEAR_BINS,            _XY_NEAR_BINS + _XY_FAR_HALF_BINS)   far positive (+20..+80)
#   [_XY_NEAR_BINS+_XY_FAR_HALF_BINS, N_BINS)                far negative (-80..-20)


def _xy_encode(c: float) -> int:
    c = float(c)
    if c >= _XY_NEAR_RANGE:
        if c >= _XY_FAR_RANGE:
            return _XY_NEAR_BINS + _XY_FAR_HALF_BINS - 1
        idx = _XY_NEAR_BINS + int(round((c - _XY_NEAR_RANGE) / _XY_FAR_BIN))
        return min(idx, _XY_NEAR_BINS + _XY_FAR_HALF_BINS - 1)
    if c <= -_XY_NEAR_RANGE:
        if c <= -_XY_FAR_RANGE:
            return N_BINS - 1
        idx = (
            _XY_NEAR_BINS
            + _XY_FAR_HALF_BINS
            + int(round((-_XY_NEAR_RANGE - c) / _XY_FAR_BIN))
        )
        return min(idx, N_BINS - 1)
    # Near zone: map [-near, +near) to [0, near_bins)
    idx = int(round((c + _XY_NEAR_RANGE) / _XY_NEAR_BIN))
    return max(0, min(idx, _XY_NEAR_BINS - 1))


def _xy_decode(idx: int) -> float:
    if idx < 0 or idx >= N_BINS:
        raise ValueError(f"xy idx out of range: {idx}")
    if idx < _XY_NEAR_BINS:
        return -_XY_NEAR_RANGE + (idx + 0.5) * _XY_NEAR_BIN
    if idx < _XY_NEAR_BINS + _XY_FAR_HALF_BINS:
        local = idx - _XY_NEAR_BINS
        return _XY_NEAR_RANGE + (local + 0.5) * _XY_FAR_BIN
    local = idx - _XY_NEAR_BINS - _XY_FAR_HALF_BINS
    return -_XY_NEAR_RANGE - (local + 0.5) * _XY_FAR_BIN


# ---------------------------------------------------------------------------
# Z scheme — linear [-3, 3] m, 1024 bins.
# ---------------------------------------------------------------------------
_Z_MIN = -3.0
_Z_MAX = 3.0
_Z_BIN = (_Z_MAX - _Z_MIN) / N_BINS


def _z_encode(c: float) -> int:
    c = float(np.clip(c, _Z_MIN, _Z_MAX - 1e-6))
    return int((c - _Z_MIN) / _Z_BIN)


def _z_decode(idx: int) -> float:
    return _Z_MIN + (idx + 0.5) * _Z_BIN


# ---------------------------------------------------------------------------
# Size scheme — log-space [0.1, 20] m, 1024 bins.
# ---------------------------------------------------------------------------
_SIZE_MIN = 0.1
_SIZE_MAX = 20.0
_LOG_SIZE_MIN = math.log(_SIZE_MIN)
_LOG_SIZE_MAX = math.log(_SIZE_MAX)
_LOG_SIZE_BIN = (_LOG_SIZE_MAX - _LOG_SIZE_MIN) / N_BINS


def _size_encode(s: float) -> int:
    s = max(_SIZE_MIN, min(_SIZE_MAX, float(s)))
    return int((math.log(s) - _LOG_SIZE_MIN) / _LOG_SIZE_BIN)


def _size_decode(idx: int) -> float:
    return float(math.exp(_LOG_SIZE_MIN + (idx + 0.5) * _LOG_SIZE_BIN))


# ---------------------------------------------------------------------------
# Yaw via sin/cos — linear [-1, 1], 1024 bins each.
# ---------------------------------------------------------------------------
_TRIG_MIN = -1.0
_TRIG_MAX = 1.0
_TRIG_BIN = (_TRIG_MAX - _TRIG_MIN) / N_BINS


def _trig_encode(t: float) -> int:
    t = max(_TRIG_MIN, min(_TRIG_MAX - 1e-6, float(t)))
    return int((t - _TRIG_MIN) / _TRIG_BIN)


def _trig_decode(idx: int) -> float:
    return _TRIG_MIN + (idx + 0.5) * _TRIG_BIN


# ---------------------------------------------------------------------------
# Per-axis dispatch tables.
# ---------------------------------------------------------------------------
_ENCODERS = {
    AXIS_X: _xy_encode,
    AXIS_Y: _xy_encode,
    AXIS_Z: _z_encode,
    AXIS_W: _size_encode,
    AXIS_L: _size_encode,
    AXIS_H: _size_encode,
    AXIS_SIN: _trig_encode,
    AXIS_COS: _trig_encode,
}
_DECODERS = {
    AXIS_X: _xy_decode,
    AXIS_Y: _xy_decode,
    AXIS_Z: _z_decode,
    AXIS_W: _size_decode,
    AXIS_L: _size_decode,
    AXIS_H: _size_decode,
    AXIS_SIN: _trig_decode,
    AXIS_COS: _trig_decode,
}


def bin_value_table() -> np.ndarray:
    """Return float32 table of shape (N_COORD_PER_BOX, N_BINS) where
    `table[axis, idx]` is the metre / unit value of bin `idx` for that axis."""
    table = np.zeros((N_COORD_PER_BOX, N_BINS), dtype=np.float32)
    for axis in range(N_COORD_PER_BOX):
        dec = _DECODERS[axis]
        for i in range(N_BINS):
            table[axis, i] = dec(i)
    return table


# ---------------------------------------------------------------------------
# Public encode / decode for a 7-DOF AABB+yaw box.
# ---------------------------------------------------------------------------
def encode_box_indices(box7: Sequence[float]) -> List[int]:
    """`box7` = [xmin, xmax, ymin, ymax, zmin, zmax, yaw] -> 8 codebook indices."""
    xmin, xmax, ymin, ymax, zmin, zmax, yaw = (float(v) for v in box7)
    cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
    w, l, h = xmax - xmin, ymax - ymin, zmax - zmin
    s, c = math.sin(yaw), math.cos(yaw)
    return [
        _xy_encode(cx),
        _xy_encode(cy),
        _z_encode(cz),
        _size_encode(w),
        _size_encode(l),
        _size_encode(h),
        _trig_encode(s),
        _trig_encode(c),
    ]


def decode_box_indices(idxs: Sequence[int]) -> List[float]:
    """Inverse of `encode_box_indices` -> [xmin, xmax, ymin, ymax, zmin, zmax, yaw]."""
    if len(idxs) != N_COORD_PER_BOX:
        raise ValueError(f"need {N_COORD_PER_BOX} indices, got {len(idxs)}")
    cx = _xy_decode(int(idxs[0]))
    cy = _xy_decode(int(idxs[1]))
    cz = _z_decode(int(idxs[2]))
    w = _size_decode(int(idxs[3]))
    l = _size_decode(int(idxs[4]))
    h = _size_decode(int(idxs[5]))
    s = _trig_decode(int(idxs[6]))
    c = _trig_decode(int(idxs[7]))
    yaw = math.atan2(s, c)
    return [
        cx - 0.5 * w,
        cx + 0.5 * w,
        cy - 0.5 * l,
        cy + 0.5 * l,
        cz - 0.5 * h,
        cz + 0.5 * h,
        yaw,
    ]


# ---------------------------------------------------------------------------
# Token-name helpers.
# ---------------------------------------------------------------------------
def coord_token_names() -> List[str]:
    """All 1024 coord-token names in codebook order."""
    return [f"{COORD_TOKEN_PREFIX}{i}{COORD_TOKEN_SUFFIX}" for i in range(N_BINS)]


def encode_box_to_text(box7: Sequence[float], box_start: str, box_end: str) -> str:
    idxs = encode_box_indices(box7)
    body = "".join(f"{COORD_TOKEN_PREFIX}{i}{COORD_TOKEN_SUFFIX}" for i in idxs)
    return f"{box_start}{body}{box_end}"


# Single regex that, when run on text, finds quantised box spans and exposes
# the eight indices. Match groups: 1..8 -> X, Y, Z, W, L, H, SIN, COS.
import re as _re

_QBOX_BLOCK = (
    r"<\|box_start\|>"
    + r"".join([_re.escape(COORD_TOKEN_PREFIX) + r"(\d+)" + _re.escape(COORD_TOKEN_SUFFIX)] * N_COORD_PER_BOX)
    + r"<\|box_end\|>"
)
QBOX_RE = _re.compile(_QBOX_BLOCK)


def parse_quantized_boxes(text: str) -> List[List[float]]:
    """Find every `<|box_start|><coord_a>...<coord_h><|box_end|>` block in
    `text` and return the decoded 7-DOF AABB+yaw boxes."""
    out: List[List[float]] = []
    if not text or COORD_TOKEN_PREFIX not in text:
        return out
    for m in QBOX_RE.finditer(text):
        try:
            idxs = [int(m.group(i + 1)) for i in range(N_COORD_PER_BOX)]
            out.append(decode_box_indices(idxs))
        except Exception:
            continue
    return out


__all__ = [
    "N_BINS",
    "N_COORD_PER_BOX",
    "AXIS_NAMES",
    "AXIS_X",
    "AXIS_Y",
    "AXIS_Z",
    "AXIS_W",
    "AXIS_L",
    "AXIS_H",
    "AXIS_SIN",
    "AXIS_COS",
    "COORD_TOKEN_PREFIX",
    "COORD_TOKEN_SUFFIX",
    "QBOX_RE",
    "bin_value_table",
    "coord_token_names",
    "encode_box_indices",
    "decode_box_indices",
    "encode_box_to_text",
    "parse_quantized_boxes",
]

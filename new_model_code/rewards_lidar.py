"""LiDAR-aware GRPO rewards for nuGrounding outputs.

Output format the dataset uses (this is *not* center-form 7-DOF; it is the
axis-aligned-min-max plus yaw):
    [xmin, xmax, ymin, ymax, zmin, zmax, yaw]
either as a single bracketed list, wrapped in [[...]] / <|box_start|>...<|box_end|>
when there are multiple boxes, or — under the Q3D pipeline — as eight
`<coord_*>` tokens between `<|box_start|>` and `<|box_end|>`. We accept both
formats so this module can score legacy SFT outputs and quantised ones with
the same call signature.

We made two design choices that diverge from the user's brief because of this
data layout:

  * BEV IoU is computed on the axis-aligned XY rect, not on rotated corners.
    The data does not actually expose (length, width) of the oriented box
    independently of yaw — only the AABB extent — so reconstructing oriented
    corners would be ambiguous. Yaw is rewarded separately via cos(Δθ).
  * R_point_cov is *not* applied online. Loading raw nuScenes LiDAR per
    rollout would dominate GRPO step time (~100ms+ per sample). We instead
    reward what is cheap and high-signal.

Rewards we apply:

  R_format     small bonus for any parseable box
  R_miss       large negative if no parseable box but GT exists
  R_bev_iou    BEV IoU between matched pred / GT axis-aligned rects
  R_center     distance-adaptive Gaussian on (x, y, z) centers
  R_heading    max(0, cos(theta_pred - theta_gt))   (cyclic-aware via cos)
  R_safety     exp(-d_gt / d0); multiplies BEV-IoU + center
  R_sanity     ground-penetration penalty (zmin too far below 0)
  R_class      for det_object only: class-name match between pred and GT
  R_extra      small penalty per unmatched extra prediction

Distance-adaptive matching threshold: a (pred, gt) pair only counts as a match
if iou >= τ(d_gt), where τ relaxes for far objects.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- parsing
_BBOX_RE = re.compile(
    r"\[\s*"
    r"(-?\d+\.?\d*)\s*,\s*"  # xmin
    r"(-?\d+\.?\d*)\s*,\s*"  # xmax
    r"(-?\d+\.?\d*)\s*,\s*"  # ymin
    r"(-?\d+\.?\d*)\s*,\s*"  # ymax
    r"(-?\d+\.?\d*)\s*,\s*"  # zmin
    r"(-?\d+\.?\d*)\s*,\s*"  # zmax
    r"(-?\d+\.?\d*)\s*"  # yaw
    r"\]"
)

NUSC_CLASSES = (
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
)


def _strip_box_tokens(text: str) -> str:
    """Remove <|box_start|> / <|box_end|> markers but keep the bbox content
    so the regex below can still find the 7-tuple."""
    if not text:
        return text
    return text.replace("<|box_start|>", "").replace("<|box_end|>", "")


def parse_boxes(text: str) -> List[np.ndarray]:
    """Extract every box from `text`, decoding both legacy bracketed
    `[xmin,xmax,...,yaw]` form and Q3D `<|box_start|><coord_*>x8<|box_end|>`
    form. Result is always a list of np.float32 arrays in
    `[xmin, xmax, ymin, ymax, zmin, zmax, yaw]` layout, ready for the
    geometry helpers below."""
    if not text:
        return []
    boxes: List[np.ndarray] = []

    # Q3D form first (so we don't strip its <|box_start|> markers below and
    # then re-discover the inner coord tokens — _strip_box_tokens would
    # otherwise mangle multi-box runs).
    try:
        from qwen_mm.quantizer import parse_quantized_boxes  # local import: optional dep
        for box7 in parse_quantized_boxes(text):
            v = np.asarray(box7, dtype=np.float32)
            if v[0] <= v[1] and v[2] <= v[3] and v[4] <= v[5]:
                boxes.append(v)
    except Exception:
        # Quantizer unavailable or parse failed — silently fall back to text.
        pass

    # Legacy text form (still supported for old eval logs / mixed outputs).
    legacy_text = _strip_box_tokens(text)
    for m in _BBOX_RE.finditer(legacy_text):
        try:
            v = np.array([float(x) for x in m.groups()], dtype=np.float32)
            if v[0] <= v[1] and v[2] <= v[3] and v[4] <= v[5]:
                boxes.append(v)
        except Exception:
            continue
    return boxes


def parse_class(text: str) -> Optional[str]:
    """Return the first nuScenes class word found in `text`, else None."""
    if not text:
        return None
    low = text.lower()
    for c in NUSC_CLASSES:
        if c in low:
            return c
    return None


# --------------------------------------------------------------------------- geometry
def bev_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ax2, ay1, ay2 = a[0], a[1], a[2], a[3]
    bx1, bx2, by1, by2 = b[0], b[1], b[2], b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1) * (by2 - by1))
    union = a_area + b_area - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def box_center(b: np.ndarray) -> np.ndarray:
    return np.array(
        [(b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5, (b[4] + b[5]) * 0.5],
        dtype=np.float32,
    )


def gt_distance(b: np.ndarray) -> float:
    c = box_center(b)
    return float(np.linalg.norm(c[:2]))


def adaptive_iou_threshold(d: float) -> float:
    """User's spec: <10m -> 0.7, 10-30m -> 0.5, >=30m -> 0.3.
    Used as the *match* threshold. Below it the pair is treated as "no match"."""
    if d < 10.0:
        return 0.7
    if d < 30.0:
        return 0.5
    return 0.3


def adaptive_sigma(d_gt: float, k: float = 0.05, sigma_min: float = 0.3) -> float:
    return max(sigma_min, d_gt * k)


# --------------------------------------------------------------------------- per-pair scoring
def center_reward(pred: np.ndarray, gt: np.ndarray) -> float:
    sigma = adaptive_sigma(gt_distance(gt))
    err = float(np.linalg.norm(box_center(pred) - box_center(gt)))
    return float(math.exp(-(err ** 2) / (2.0 * sigma ** 2)))


def heading_reward(pred: np.ndarray, gt: np.ndarray) -> float:
    return max(0.0, float(math.cos(float(pred[6]) - float(gt[6]))))


def safety_weight(gt: np.ndarray, d0: float = 25.0) -> float:
    return float(math.exp(-gt_distance(gt) / d0))


def ground_penalty(b: np.ndarray, ground_z: float = -1.8, tol: float = 0.5) -> float:
    """Penalty for predicted box bottom that goes well below the ground plane.

    Default ground_z = -1.8 corresponds to the typical nuScenes LIDAR_TOP
    mounting height above the road. Boxes may legitimately reach down to
    ground_z; we only penalise zmin < ground_z - tol.
    """
    bottom = float(b[4])  # zmin
    if bottom < ground_z - tol:
        return min(2.0, abs(bottom - (ground_z - tol)))
    return 0.0


# --------------------------------------------------------------------------- matching
def adaptive_match(
    preds: Sequence[np.ndarray], gts: Sequence[np.ndarray]
) -> List[Tuple[int, int, float]]:
    """Greedy matcher with **soft** scoring.

    Earlier we required BEV IoU >= τ(d_gt) for a pair to count, but that left
    GRPO with reward variance ~0 in early training (every rollout's prediction
    sat below the threshold and got the same constant penalty).

    Now we accept any positive overlap, so the *value* (iou) carries the
    signal. Soft-match strategy:
      - Centre-distance fallback: if no overlap exists, we still match the
        single closest pred-gt pair so center/heading rewards can drive
        learning. We attribute iou=0 in that case.
      - Adaptive threshold becomes a **gain**: pairs above τ get a 1.5x
        weight on bev_iou so well-localised hits are rewarded extra.

    Returns list of (pred_idx, gt_idx, iou_score) where iou_score is the
    raw IoU (possibly multiplied by the adaptive bonus).
    """
    if not preds or not gts:
        return []
    P, G = len(preds), len(gts)
    iou = np.zeros((P, G), dtype=np.float32)
    for i in range(P):
        for j in range(G):
            iou[i, j] = bev_iou(preds[i], gts[j])
    work = iou.copy()
    matched: List[Tuple[int, int, float]] = []
    used_p, used_g = set(), set()
    while True:
        i, j = np.unravel_index(np.argmax(work), work.shape)
        if work[i, j] <= 0.0:
            break
        if i in used_p or j in used_g:
            work[i, j] = -1.0
            continue
        score = float(work[i, j])
        if score >= adaptive_iou_threshold(gt_distance(gts[j])):
            score *= 1.5  # confident match bonus
        matched.append((int(i), int(j), score))
        used_p.add(int(i))
        used_g.add(int(j))
        work[i, :] = -1.0
        work[:, j] = -1.0
        if len(matched) == min(P, G):
            break

    # Centre-distance fallback for unmatched preds — gives continuous reward
    # via center_reward + heading_reward even when IoU is 0.
    if not matched and preds and gts:
        # Match each pred to its nearest GT by centre distance.
        c_p = np.stack([box_center(p) for p in preds])
        c_g = np.stack([box_center(g) for g in gts])
        d = np.linalg.norm(c_p[:, None, :] - c_g[None, :, :], axis=-1)  # (P, G)
        used_p, used_g = set(), set()
        while True:
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] >= 1e9:
                break
            if i in used_p or j in used_g:
                d[i, j] = 1e9
                continue
            matched.append((int(i), int(j), 0.0))
            used_p.add(int(i))
            used_g.add(int(j))
            d[i, :] = 1e9
            d[:, j] = 1e9
            if len(matched) == min(P, G):
                break
    return matched


# --------------------------------------------------------------------------- top-level reward
@dataclass
class RewardWeights:
    bev_iou: float = 2.0  # boost real overlap signal
    center: float = 0.5
    heading: float = 0.3
    miss: float = 1.0
    format_bonus: float = 0.05
    extra_pred_penalty: float = 0.2
    sanity: float = 0.5
    class_match: float = 0.5
    # Linear "approach" reward used when IoU is 0 but a match exists via the
    # centre-distance fallback. Gives the policy a smooth gradient toward the
    # GT centre even when far away, instead of a flat floor it can exploit.
    approach_scale: float = 30.0  # metres at which approach reward hits 0
    approach_weight: float = 0.6


def compute_reward(
    pred_text: str,
    gt_text: str,
    template_type: Optional[str] = None,
    weights: RewardWeights = RewardWeights(),
) -> Tuple[float, dict]:
    """Compute scalar reward.

    template_type: "det_object" (class is the main signal, bbox is echoed) or
        "det_area" (bbox is the main signal). Anything else falls back to the
        bbox path with no class reward.
    """
    preds = parse_boxes(pred_text)
    gts = parse_boxes(gt_text)

    info = {
        "n_pred": len(preds),
        "n_gt": len(gts),
        "bev_iou": 0.0,
        "center": 0.0,
        "heading": 0.0,
        "missed": 0,
        "extra": 0,
        "ground_pen": 0.0,
        "class_ok": False,
        "template": template_type or "?",
    }

    # ---- det_object branch: the box is given in the prompt and just echoed
    # in the answer, so class accuracy is the only learning signal worth
    # rewarding. Wrong/missing class word -> negative.
    if template_type == "det_object":
        gt_class = parse_class(gt_text)
        pred_class = parse_class(pred_text)
        info["class_ok"] = bool(gt_class and pred_class and gt_class == pred_class)
        if info["class_ok"]:
            r = float(weights.class_match) * 2.0  # ~+1.0
        elif pred_class is None:
            r = -float(weights.class_match) * 2.0  # ~-1.0  (no class word)
        else:
            r = -float(weights.class_match)  # ~-0.5  (wrong class)
        return float(r), info

    # ---- det_area / generic bbox branch.
    if not gts:
        return 0.0, info
    if not preds:
        info["missed"] = len(gts)
        return -float(weights.miss), info

    matches = adaptive_match(preds, gts)
    n_match = len(matches)
    if n_match == 0:
        return -float(weights.miss) * 0.5 + float(weights.format_bonus), info

    bev_terms, center_terms, heading_terms, approach_terms = [], [], [], []
    for pi, gj, score in matches:
        p, g = preds[pi], gts[gj]
        sw = safety_weight(g)
        iou_raw = bev_iou(p, g)
        bev_terms.append(sw * iou_raw)
        center_terms.append(sw * center_reward(p, g))
        # Heading only meaningful when there's at least some overlap; for
        # fallback-only matches the model hasn't even found the object yet.
        if iou_raw > 0:
            heading_terms.append(heading_reward(p, g))
        else:
            heading_terms.append(0.0)
        # Linear approach reward in metres-space — always non-zero, smoothly
        # decreasing with centre distance. This is what gives the policy a
        # gradient when IoU=0 but ALSO discourages the "just emit any box"
        # exploit (the closer you are, the more reward you get).
        err = float(np.linalg.norm(box_center(p) - box_center(g)))
        approach_terms.append(max(0.0, 1.0 - err / weights.approach_scale))

    bev = float(np.mean(bev_terms))
    cen = float(np.mean(center_terms))
    hed = float(np.mean(heading_terms))
    app = float(np.mean(approach_terms))

    # Ground sanity over all predictions (penalise hallucinated underground boxes).
    ground = float(np.mean([ground_penalty(p) for p in preds]))

    info["bev_iou"] = bev
    info["center"] = cen
    info["heading"] = hed
    info["approach"] = app
    info["missed"] = max(0, len(gts) - n_match)
    info["extra"] = max(0, len(preds) - n_match)
    info["ground_pen"] = ground

    r = (
        weights.bev_iou * bev
        + weights.center * cen
        + weights.heading * hed
        + weights.approach_weight * app
        + weights.format_bonus
        - weights.miss * (info["missed"] / max(1, len(gts)))
        - weights.extra_pred_penalty * (info["extra"] / max(1, len(preds)))
        - weights.sanity * ground
    )
    return float(r), info


__all__ = [
    "parse_boxes",
    "parse_class",
    "bev_iou",
    "compute_reward",
    "RewardWeights",
    "NUSC_CLASSES",
]

"""AURORA VISION — verification metrics (ROC / FAR / FRR / EER / AUC).

Kept dependency-free (NumPy only) so that the Kotlin implementation in
``com.auroravision.core.calibration`` can reproduce the same numbers from the
same inputs, and so the evaluation is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class RocPoint:
    threshold: float
    far: float
    frr: float
    tar: float


def rates(same: Sequence[float], diff: Sequence[float], threshold: float) -> Tuple[float, float]:
    """(FAR, FRR) for a similarity threshold.  Accept when similarity >= t."""
    s = np.asarray(same, dtype=np.float64)
    d = np.asarray(diff, dtype=np.float64)
    if s.size == 0 or d.size == 0:
        raise ValueError("both same and diff score sets must be non-empty")
    frr = float(np.mean(s < threshold))
    far = float(np.mean(d >= threshold))
    return far, frr


def roc_curve(same: Sequence[float], diff: Sequence[float]) -> List[RocPoint]:
    """Sweep every distinct observed score as a threshold."""
    s = np.asarray(same, dtype=np.float64)
    d = np.asarray(diff, dtype=np.float64)
    thresholds = np.unique(np.concatenate([s, d]))
    points = []
    for t in thresholds:
        far, frr = rates(s, d, float(t))
        points.append(RocPoint(float(t), far, frr, 1.0 - frr))
    return points


def auc(same: Sequence[float], diff: Sequence[float]) -> float:
    """Area under the ROC curve (probability that a same pair outranks a diff pair).

    Computed exactly via the Mann-Whitney U statistic (ties count 1/2).
    """
    s = np.asarray(same, dtype=np.float64)
    d = np.asarray(diff, dtype=np.float64)
    allv = np.concatenate([s, d])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    rank_sum = ranks[: len(s)].sum()
    u = rank_sum - len(s) * (len(s) + 1) / 2.0
    return float(u / (len(s) * len(d)))


def eer(same: Sequence[float], diff: Sequence[float]) -> Tuple[float, float]:
    """Equal-error rate and the threshold that achieves it (linear interp)."""
    pts = roc_curve(same, diff)
    best = min(pts, key=lambda p: abs(p.far - p.frr))
    return float((best.far + best.frr) / 2.0), float(best.threshold)


def threshold_at_far(same: Sequence[float], diff: Sequence[float],
                     target_far: float) -> float:
    """Lowest threshold whose empirical FAR is <= ``target_far``.

    Uses the *different-pair* distribution only, i.e. a Neyman-Pearson style
    operating point; the same-pair distribution is then used to report TAR.
    """
    d = np.sort(np.asarray(diff, dtype=np.float64))
    if d.size == 0:
        raise ValueError("no impostor scores")
    # fraction of impostors >= t must be <= target_far
    k = int(np.floor(target_far * d.size))
    if k >= d.size:
        return float(d[0])
    # allow at most k impostor scores above the threshold
    idx = d.size - 1 - k
    return float(np.nextafter(d[idx], np.inf)) if idx + 1 < d.size else float(d[-1] + 1e-6)


def threshold_at_frr(same: Sequence[float], diff: Sequence[float],
                     target_frr: float) -> float:
    s = np.sort(np.asarray(same, dtype=np.float64))
    k = int(np.floor(target_frr * s.size))
    idx = min(k, s.size - 1)
    return float(np.nextafter(s[idx], -np.inf))


def summary(same: Sequence[float], diff: Sequence[float], threshold: float,
            target_far: float | None = None) -> Dict[str, object]:
    far, frr = rates(same, diff, threshold)
    a = auc(same, diff)
    e, e_thr = eer(same, diff)
    s = np.asarray(same, dtype=np.float64)
    d = np.asarray(diff, dtype=np.float64)
    return {
        "threshold": round(threshold, 6),
        "far": round(far, 6),
        "frr": round(frr, 6),
        "tar": round(1.0 - frr, 6),
        "accuracy": round(0.5 * ((1.0 - far) + (1.0 - frr)), 6),
        "auc": round(a, 6),
        "eer": round(e, 6),
        "eerThreshold": round(e_thr, 6),
        "samePairs": int(s.size),
        "diffPairs": int(d.size),
        "sameMean": round(float(s.mean()), 4) if s.size else None,
        "sameP05": round(float(np.percentile(s, 5)), 4) if s.size else None,
        "diffMean": round(float(d.mean()), 4) if d.size else None,
        "diffP95": round(float(np.percentile(d, 95)), 4) if d.size else None,
        "diffP99": round(float(np.percentile(d, 99)), 4) if d.size else None,
        "targetFar": target_far,
    }

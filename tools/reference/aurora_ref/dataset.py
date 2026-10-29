"""AURORA VISION — evaluation dataset handling (LFW + synthetic degradations).

This module deliberately contains **no fabricated data**: it either reads real
image files from disk or synthesises a *degraded* version of a real image with a
documented transform (so the effect on quality metrics is measurable).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .image import resize_bilinear


@dataclass(frozen=True)
class EvalImage:
    path: str
    identity: str
    name: str


def _identity_of(stem: str) -> str:
    parts = stem.split("_")
    if parts[-1].isdigit():
        return "_".join(parts[:-1])
    return stem


def load_lfw(root: str) -> List[EvalImage]:
    """Load an LFWCrop-style directory of ``Name_0001.ppm`` files."""
    out = []
    for fn in sorted(os.listdir(root)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in (".ppm", ".jpg", ".jpeg", ".png"):
            continue
        out.append(EvalImage(os.path.join(root, fn), _identity_of(stem), stem))
    return out


def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def build_pairs(images: List[EvalImage], max_same_per_identity: int = 3,
                seed: int = 20240517) -> Tuple[List[Tuple[EvalImage, EvalImage]],
                                               List[Tuple[EvalImage, EvalImage]]]:
    """Deterministically build same-identity and different-identity pairs."""
    rng = np.random.default_rng(seed)
    by_id: Dict[str, List[EvalImage]] = {}
    for im in images:
        by_id.setdefault(im.identity, []).append(im)

    same: List[Tuple[EvalImage, EvalImage]] = []
    for ident in sorted(by_id):
        group = sorted(by_id[ident], key=lambda e: e.name)
        if len(group) < 2:
            continue
        combos = [(group[i], group[j]) for i in range(len(group)) for j in range(i + 1, len(group))]
        if len(combos) > max_same_per_identity:
            picks = rng.choice(len(combos), size=max_same_per_identity, replace=False)
            combos = [combos[int(k)] for k in sorted(picks)]
        same.extend(combos)

    # impostor pairs: random distinct identities, same count as genuine pairs
    ids = sorted(by_id)
    diff: List[Tuple[EvalImage, EvalImage]] = []
    seen = set()
    guard = 0
    while len(diff) < len(same) and guard < len(same) * 200:
        guard += 1
        a, b = ids[int(rng.integers(0, len(ids)))], ids[int(rng.integers(0, len(ids)))]
        if a == b:
            continue
        ia = by_id[a][int(rng.integers(0, len(by_id[a])))]
        ib = by_id[b][int(rng.integers(0, len(by_id[b])))]
        key = (min(ia.name, ib.name), max(ia.name, ib.name))
        if key in seen:
            continue
        seen.add(key)
        diff.append((ia, ib))
    return same, diff


def split_dev_test(pairs, fraction_dev: float = 0.5):
    """Split deterministically by hashing the pair key (no identity leakage
    between the two halves is *not* enforced for impostor-free pairing; the
    split is reported honestly in the calibration report)."""
    dev, test = [], []
    for a, b in pairs:
        key = "|".join(sorted([a.name, b.name]))
        h = _stable_hash(key) / float(1 << 256)
        (dev if h < fraction_dev else test).append((a, b))
    return dev, test


# --------------------------------------------------------------------------- #
# Synthetic degradations (documented, deterministic)
# --------------------------------------------------------------------------- #
def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2 * sigma * sigma))
    k /= k.sum()
    out = img.astype(np.float64)
    # separable convolution with edge replication
    pad = np.pad(out, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    tmp = sum(k[i] * pad[i:i + out.shape[0]] for i in range(len(k)))
    pad = np.pad(tmp, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    out = sum(k[i] * pad[:, i:i + out.shape[1]] for i in range(len(k)))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _darken(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(np.rint(img.astype(np.float64) * factor), 0, 255).astype(np.uint8)


def _lowres(img: np.ndarray, px: int) -> np.ndarray:
    h, w = img.shape[:2]
    small = resize_bilinear(img, max(8, px), max(8, int(round(px * h / w))))
    return resize_bilinear(small, w, h)


def _occlude(img: np.ndarray, frac: float) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    out[int(h * (1 - frac)):, int(w * (1 - frac)):] = 0
    return out


def _rotate(img: np.ndarray, deg: float) -> np.ndarray:
    import math
    h, w = img.shape[:2]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    a = math.radians(deg)
    ca, sa = math.cos(a), math.sin(a)
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    sx = ca * (xs - cx) + sa * (ys - cy) + cx
    sy = -sa * (xs - cx) + ca * (ys - cy) + cy
    x0 = np.clip(np.floor(sx).astype(int), 0, w - 1)
    y0 = np.clip(np.floor(sy).astype(int), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = np.clip(sx - np.floor(sx), 0, 1)[..., None]
    fy = np.clip(sy - np.floor(sy), 0, 1)[..., None]
    f = img.astype(np.float64)
    top = f[y0, x0] * (1 - fx) + f[y0, x1] * fx
    bot = f[y1, x0] * (1 - fx) + f[y1, x1] * fx
    return np.clip(np.rint(top * (1 - fy) + bot * fy), 0, 255).astype(np.uint8)


DEGRADATIONS = {
    "blur_sigma1.5": lambda im: _blur(im, 1.5),
    "blur_sigma3": lambda im: _blur(im, 3.0),
    "blur_sigma5": lambda im: _blur(im, 5.0),
    "dark_35pct": lambda im: _darken(im, 0.35),
    "dark_15pct": lambda im: _darken(im, 0.15),
    "overexposed": lambda im: _darken(im, 1.9),
    "lowres_40px": lambda im: _lowres(im, 40),
    "lowres_24px": lambda im: _lowres(im, 24),
    "occluded_40pct": lambda im: _occlude(im, 0.4),
    "rotated_25deg": lambda im: _rotate(im, 25.0),
    "rotated_45deg": lambda im: _rotate(im, 45.0),
}

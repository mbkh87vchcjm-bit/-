"""AURORA VISION — reference imaging primitives.

The functions here are the *numeric contract* for the Kotlin implementation in
`:core` (com.auroravision.core.math).  Every operation is defined explicitly so
that the Python reference and the Android/JVM code can be compared bit-for-bit
(within float tolerance) by the golden-vector tests in `:core`.

Conventions
-----------
* Images are ``numpy.ndarray`` of shape ``(H, W, 3)``, dtype ``uint8``, RGB order.
* Geometry uses pixel centres at ``(x + 0.5, y + 0.5)`` (OpenCV / "align_corners=False").
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Resize
# --------------------------------------------------------------------------- #
def _axis_weights(src_size: int, dst_size: int):
    """Half-pixel bilinear weights for one axis (identical to OpenCV INTER_LINEAR)."""
    scale = src_size / dst_size
    dst = np.arange(dst_size, dtype=np.float32)
    src = (dst + 0.5) * scale - 0.5
    src = np.clip(src, 0.0, src_size - 1.0)
    i0 = np.floor(src).astype(np.int32)
    i1 = np.minimum(i0 + 1, src_size - 1)
    w = (src - i0).astype(np.float32)
    return i0, i1, w


def resize_bilinear(img: np.ndarray, dst_w: int, dst_h: int) -> np.ndarray:
    """Bilinear resize, uint8 in -> uint8 out (round-half-away-from-zero)."""
    src_h, src_w = img.shape[:2]
    if src_w == dst_w and src_h == dst_h:
        return img.copy()
    y0, y1, wy = _axis_weights(src_h, dst_h)
    x0, x1, wx = _axis_weights(src_w, dst_w)

    src = img.astype(np.float32)
    wy = wy[:, None, None]
    wx = wx[None, :, None]
    top = src[y0][:, x0] * (1.0 - wx) + src[y0][:, x1] * wx
    bot = src[y1][:, x0] * (1.0 - wx) + src[y1][:, x1] * wx
    out = top * (1.0 - wy) + bot * wy
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


def letterbox(img: np.ndarray, size: int, pad_value: int = 0, allow_upscale: bool = False):
    """Aspect-preserving resize into a square canvas.

    Returns ``(canvas, scale, pad_x, pad_y)`` such that a point ``p`` in the
    original image maps to ``p * scale + (pad_x, pad_y)`` in the canvas.

    By default the image is only ever *downscaled* (``allow_upscale=False``):
    upscaling a small photo costs time and blurs detail, and YuNet was trained
    on faces of 10-300 px.  The landmark-refinement pass is the one place that
    deliberately upscales (it works on a tight face crop).
    """
    src_h, src_w = img.shape[:2]
    scale = min(size / src_w, size / src_h)
    if not allow_upscale:
        scale = min(scale, 1.0)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = resize_bilinear(img, new_w, new_h)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


# --------------------------------------------------------------------------- #
# Affine warp (used by face alignment)
# --------------------------------------------------------------------------- #
def similarity_matrix(scale: float, rot: np.ndarray, tx: float, ty: float) -> np.ndarray:
    """Build the 2x3 matrix [scale*R | t] with R a 2x2 rotation."""
    m = np.zeros((2, 3), dtype=np.float64)
    m[:2, :2] = scale * rot
    m[0, 2] = tx
    m[1, 2] = ty
    return m


def invert_affine(m: np.ndarray) -> np.ndarray:
    """Invert a 2x3 affine matrix."""
    a, b, c = m[0]
    d, e, f = m[1]
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise ValueError("singular affine matrix")
    inv = np.array([
        [e / det, -b / det, (b * f - e * c) / det],
        [-d / det, a / det, (d * c - a * f) / det],
    ], dtype=np.float64)
    return inv


def warp_affine_bilinear(img: np.ndarray, m: np.ndarray, dst_w: int, dst_h: int,
                         border: int = 0) -> np.ndarray:
    """Warp ``img`` by the 2x3 matrix ``m`` (source -> destination), bilinear.

    Pixels outside the source are filled with ``border``.  Matches
    ``cv2.warpAffine(..., INTER_LINEAR, BORDER_CONSTANT)`` closely enough for
    parity testing (differences are limited to integer rounding).
    """
    inv = invert_affine(m)
    y, x = np.meshgrid(np.arange(dst_h, dtype=np.float32) + 0.5,
                       np.arange(dst_w, dtype=np.float32) + 0.5, indexing="ij")
    # destination -> source
    src_x = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2] - 0.5
    src_y = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2] - 0.5
    src_h, src_w = img.shape[:2]
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    fx = (src_x - x0).astype(np.float32)
    fy = (src_y - y0).astype(np.float32)
    x1, y1 = x0 + 1, y0 + 1

    def sample(xx, yy):
        valid = (xx >= 0) & (xx < src_w) & (yy >= 0) & (yy < src_h)
        cx = np.clip(xx, 0, src_w - 1)
        cy = np.clip(yy, 0, src_h - 1)
        vals = img[cy, cx].astype(np.float32)
        vals[~valid] = float(border)
        return vals

    p00 = sample(x0, y0)
    p10 = sample(x1, y0)
    p01 = sample(x0, y1)
    p11 = sample(x1, y1)
    top = p00 * (1.0 - fx)[..., None] + p10 * fx[..., None]
    bot = p01 * (1.0 - fx)[..., None] + p11 * fx[..., None]
    out = top * (1.0 - fy)[..., None] + bot * fy[..., None]
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Umeyama similarity transform (5-point face alignment, as used by SFace)
# --------------------------------------------------------------------------- #
def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    ``src``/``dst`` are ``(N, 2)`` float arrays.  Returns a 2x3 matrix.
    This mirrors OpenCV's ``Utils::getSimilarityTransform`` (Umeyama 1991),
    which is what ``FaceRecognizerSF::alignCrop`` uses.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_d = src - src_mean
    dst_d = dst - dst_mean

    cov = (dst_d.T @ src_d) / n          # 2x2
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        s[-1] *= -1
        r = u @ vt

    var_src = (src_d ** 2).sum() / n
    if var_src <= 0:
        raise ValueError("degenerate source points")
    scale = s.sum() / var_src
    t = dst_mean - scale * (r @ src_mean)
    return similarity_matrix(scale, r, t[0], t[1])


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def to_gray(img: np.ndarray) -> np.ndarray:
    """BT.601 luma, float32 in [0, 255]."""
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def crop(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    h_img, w_img = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(w_img, x + w), min(h_img, y + h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("empty crop")
    return img[y0:y1, x0:x1]

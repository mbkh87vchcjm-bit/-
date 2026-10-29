"""AURORA VISION — face quality analysis (deterministic, Kotlin-portable).

Every metric here is computed with plain arithmetic so that the Kotlin
implementation in ``com.auroravision.core.quality`` produces identical numbers.
Thresholds are calibrated by ``tools/reference/eval_quality.py`` on the LFW
crops plus synthetically degraded variants, and the chosen values are exported
to ``app/src/main/assets/calibration/quality_policy.json``.

The point of this module is to *refuse* to produce a confident comparison on an
unusable image instead of returning a meaningless similarity score.
"""
from __future__ import annotations

import numpy as np

# --- tunables (initial guesses; overwritten by the calibrated policy) ------- #
DEFAULTS = {
    "minFacePx": 48.0,          # smallest usable face (short side, pixels)
    "minFaceRatio": 0.03,       # face short side / image short side
    "minSharpness": 30.0,       # variance of Laplacian on 112x112 aligned crop
    "minBrightness": 45.0,      # mean luma
    "maxBrightness": 215.0,     # mean luma
    "minContrast": 18.0,        # std of luma
    "maxAbsYaw": 0.35,          # landmark asymmetry proxy, 0 = frontal
    "maxAbsRollDeg": 30.0,
}


def to_gray_float(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luma in [0, 255]."""
    return (0.299 * rgb[..., 0].astype(np.float64)
            + 0.587 * rgb[..., 1].astype(np.float64)
            + 0.114 * rgb[..., 2].astype(np.float64))


def laplacian_variance(gray: np.ndarray, inset_ratio: float = 0.15) -> float:
    """Variance of the 4-neighbour Laplacian (classic blur detector).

    A border of ``inset_ratio`` of the short side is excluded: the aligned crop
    is produced by a warp with zero padding, so when a face sits at the edge of
    the source photo the outer pixels contain an artificial black frame. That
    frame has a huge Laplacian response and would make a *blurrier* image look
    sharper (measured: blurred faces with clipped alignment scored 200-440
    instead of ~5). Border pixels inside the inset use replication.
    """
    m = max(1, int(round(min(gray.shape) * inset_ratio)))
    if gray.shape[0] <= 2 * m + 2 or gray.shape[1] <= 2 * m + 2:
        g = gray
    else:
        g = gray[m:gray.shape[0] - m, m:gray.shape[1] - m]
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    up = np.vstack([g[0:1, :], g[:-1, :]])
    down = np.vstack([g[1:, :], g[-1:, :]])
    left = np.hstack([g[:, 0:1], g[:, :-1]])
    right = np.hstack([g[:, 1:], g[:, -1:]])
    lap = up + down + left + right - 4.0 * g
    return float(np.var(lap))


def yaw_proxy(landmarks: np.ndarray) -> float:
    """Left/right asymmetry of the nose relative to the eyes.

    ``landmarks`` order: right eye, left eye, nose, right mouth, left mouth
    (image space, so "right eye" is the one on the left of the picture).
    Returns 0 for a frontal face, positive when the face turns to one side.
    """
    p = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    eye_r, eye_l, nose = p[0], p[1], p[2]
    d_r = float(np.linalg.norm(nose - eye_r))
    d_l = float(np.linalg.norm(nose - eye_l))
    if d_r + d_l <= 1e-6:
        return 0.0
    return (d_l - d_r) / (d_l + d_r)


def roll_degrees(landmarks: np.ndarray) -> float:
    p = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    dy = p[1][1] - p[0][1]
    dx = p[1][0] - p[0][0]
    return float(np.degrees(np.arctan2(dy, dx)))


def analyze_quality(aligned_rgb: np.ndarray, landmarks: np.ndarray,
                    face_w: float, face_h: float, image_w: int, image_h: int,
                    policy: dict | None = None) -> dict:
    """Compute all quality metrics and derive a status.

    status: ``ok`` | ``warning`` | ``rejected`` (rejected = do not compare).
    """
    pol = dict(DEFAULTS)
    if policy:
        pol.update(policy)
    gray = to_gray_float(aligned_rgb)
    sharpness = laplacian_variance(gray)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    yaw = abs(yaw_proxy(landmarks))
    roll = abs(roll_degrees(landmarks))
    short_side = min(face_w, face_h)
    ratio = short_side / float(max(1, min(image_w, image_h)))

    issues: list[dict] = []

    def add(code, severity, value, limit, ar_message, en_message):
        issues.append({"code": code, "severity": severity, "value": round(value, 3),
                       "limit": limit, "messageAr": ar_message, "messageEn": en_message})

    if short_side < pol["minFacePx"]:
        add("FACE_TOO_SMALL", "rejected", short_side, pol["minFacePx"],
            "الوجه صغير جدًا لمقارنة موثوقة", "Face is too small for a reliable comparison")
    elif short_side < pol["minFacePx"] * 1.6:
        add("FACE_SMALL", "warning", short_side, pol["minFacePx"] * 1.6,
            "حجم الوجه صغير نسبيًا", "Face is relatively small")
    if ratio < pol["minFaceRatio"]:
        add("FACE_RATIO_LOW", "warning", ratio, pol["minFaceRatio"],
            "الوجه يشغل مساحة صغيرة من الصورة", "Face occupies a small part of the image")
    if sharpness < pol["minSharpness"]:
        add("BLURRY", "rejected" if sharpness < pol["minSharpness"] * 0.5 else "warning",
            sharpness, pol["minSharpness"],
            "الصورة ضبابية أو غير واضحة", "Image is blurry or out of focus")
    if brightness < pol["minBrightness"]:
        add("TOO_DARK", "rejected" if brightness < pol["minBrightness"] * 0.6 else "warning",
            brightness, pol["minBrightness"],
            "الإضاءة ضعيفة جدًا", "Lighting is too dark")
    if brightness > pol["maxBrightness"]:
        add("OVEREXPOSED", "warning", brightness, pol["maxBrightness"],
            "الإضاءة ساطعة جدًا (تشبع)", "Image is overexposed")
    if contrast < pol["minContrast"]:
        add("LOW_CONTRAST", "warning", contrast, pol["minContrast"],
            "تباين الصورة منخفض", "Image contrast is low")
    if yaw > pol["maxAbsYaw"]:
        add("EXTREME_YAW", "warning" if yaw < pol["maxAbsYaw"] * 1.5 else "rejected",
            yaw, pol["maxAbsYaw"],
            "زاوية الوجه بعيدة عن الأمامية", "Face angle is far from frontal")
    if roll > pol["maxAbsRollDeg"]:
        add("EXTREME_ROLL", "warning", roll, pol["maxAbsRollDeg"],
            "ميل الوجه كبير", "Face is rotated significantly")

    severity = "ok"
    if any(i["severity"] == "rejected" for i in issues):
        severity = "rejected"
    elif issues:
        severity = "warning"

    # 0..100 heuristic score: 100 = every metric in its comfortable range.
    def norm(v, lo, hi):
        return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

    score = (0.35 * norm(sharpness, pol["minSharpness"] * 0.5, pol["minSharpness"] * 4)
             + 0.20 * (1.0 - abs(brightness - 130.0) / 130.0 if False else
                       float(np.clip(1.0 - abs(brightness - 128.0) / 90.0, 0.0, 1.0)))
             + 0.15 * norm(contrast, pol["minContrast"] * 0.5, pol["minContrast"] * 3)
             + 0.15 * float(np.clip(1.0 - yaw / max(1e-6, pol["maxAbsYaw"] * 1.5), 0.0, 1.0))
             + 0.15 * norm(short_side, pol["minFacePx"], pol["minFacePx"] * 4))
    score = float(np.clip(score, 0.0, 1.0))

    return {
        "status": severity,
        "score": round(score, 4),
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "yawProxy": round(yaw, 4),
        "rollDeg": round(roll, 2),
        "faceShortSidePx": round(float(short_side), 2),
        "faceRatio": round(float(ratio), 4),
        "issues": issues,
    }

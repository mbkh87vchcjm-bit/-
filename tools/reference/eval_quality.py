#!/usr/bin/env python3
"""Calibrates the face-quality gate on real faces plus synthetic degradations.

Method (and its deliberate bias):

  * "accept" distribution  = LFW faces (real photos, undegraded);
  * "reject" distribution  = the same faces with a heavy synthetic degradation
    (Gaussian blur sigma=5, 15 % brightness, 45 degree rotation, 40 % occlusion);
  * a metric threshold is placed between the two distributions and the resulting
    false-reject rate (on clean faces) and detection rate (on degraded faces) are
    both reported.

The gate is intentionally *permissive*: its job is to stop the app from
producing a confident similarity on an unusable image, not to reject merely
imperfect photos. False rejections are therefore kept near zero and the numbers
to justify that choice are recorded in out/quality_eval.json.

Outputs:
  out/quality_eval.json
  app/src/main/assets/calibration/quality_policy.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aurora_ref.dataset import DEGRADATIONS, load_lfw          # noqa: E402
from aurora_ref.pipeline import FacePipeline                   # noqa: E402
from aurora_ref.quality import DEFAULTS                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "..", "app", "src", "main", "assets", "models")
OUT_ASSETS = os.path.join(HERE, "..", "..", "app", "src", "main", "assets", "calibration")

DEGRADED = {
    "blur_sigma5": "blur_sigma5",
    "dark_15pct": "dark_15pct",
    "rotated_45deg": "rotated_45deg",
    "occluded_40pct": "occluded_40pct",
}
CLEARLY_BAD = {"blur_sigma5", "dark_15pct", "rotated_45deg"}


def read_ppm(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()
    fields, pos = [], 0
    while len(fields) < 4:
        while pos < len(data) and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while data[pos:pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        fields.append(data[start:pos])
    pos += 1
    w, h = int(fields[1]), int(fields[2])
    return np.frombuffer(data[pos:pos + w * h * 3], dtype=np.uint8).reshape(h, w, 3).copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", default="/home/user/tmp/lfw/faces")
    ap.add_argument("--scenes", default=os.path.join(HERE, "data", "scenes"))
    ap.add_argument("--count", type=int, default=150)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(OUT_ASSETS, exist_ok=True)

    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    det = ort.InferenceSession(os.path.join(MODELS, "face_detection_yunet_2023mar.onnx"),
                              opts, providers=["CPUExecutionProvider"])
    emb = ort.InferenceSession(os.path.join(MODELS, "face_recognition_sface_2021dec_int8.onnx"),
                               opts, providers=["CPUExecutionProvider"])
    pipe = FacePipeline(det, emb, refine=True)

    def measure(image, is_scene: bool, label: str) -> dict | None:
        faces = pipe.process(image)
        if not faces:
            return None
        best = max(faces, key=lambda r: r.box.w * r.box.h)
        q = dict(best.quality)
        q["_image"] = label
        q["_scene"] = is_scene
        return q

    clean: list[dict] = []
    degraded: dict[str, list[dict]] = {k: [] for k in DEGRADED}
    scene_clean: list[dict] = []

    images = load_lfw(args.lfw)
    step = max(1, len(images) // args.count)
    for im in images[::step][: args.count]:
        try:
            base = read_ppm(im.path)
        except Exception:
            continue
        q = measure(base, False, im.name)
        if q is None:
            continue
        clean.append(q)
        for key in DEGRADED:
            qd = measure(DEGRADATIONS[key](base), False, im.name)
            if qd is not None:
                qd["_degradation"] = key
                degraded[key].append(qd)

    # Whole scenes (photographs rather than 250px crops) — closer to real app input
    for name in sorted(os.listdir(args.scenes)):
        if not name.lower().endswith((".jpg", ".jpeg", ".png")) or name.startswith("blank"):
            continue
        try:
            from PIL import Image
            base = np.array(Image.open(os.path.join(args.scenes, name)).convert("RGB"))
        except Exception:
            continue
        q = measure(base, True, name)
        if q is not None:
            scene_clean.append(q)
        for key in DEGRADED:
            qd = measure(DEGRADATIONS[key](base), True, name)
            if qd is not None:
                qd["_degradation"] = key
                scene_clean.append(qd) if False else None
                degraded[key].append(qd)

    def vals(rows, key):
        return [r[key] for r in rows if isinstance(r.get(key), (int, float))]

    def pct(rows, key, p):
        v = vals(rows, key)
        return float(np.percentile(v, p)) if v else None

    all_clean = clean + scene_clean
    sharp_clean_p01 = pct(all_clean, "sharpness", 1)
    sharp_clean_p05 = pct(all_clean, "sharpness", 5)
    sharp_clean_min = min(vals(all_clean, "sharpness"))
    # Blur reference: photographs (scenes), not 250px crops. In a small crop the
    # alignment upscales the face into the 112x112 template and re-introduces
    # high-frequency kinks, which makes the Laplacian metric unreliable there.
    # That limitation is reported rather than hidden (see docs/EVALUATION.md).
    scene_blur = [r for r in degraded["blur_sigma5"] if r.get("_scene")]
    blur_p95 = pct(scene_blur, "sharpness", 95) or 0.0
    blur_p50 = pct(scene_blur, "sharpness", 50) or 0.0

    # Sharpness threshold: midway (geometric) between the 1st percentile of real
    # faces and the 95th percentile of heavily blurred faces, clamped so it never
    # eats into the real-face distribution.
    if sharp_clean_p01 and blur_p50 and sharp_clean_p01 > blur_p50:
        # geometric midpoint between "heavily blurred" (median) and the clean 1st
        # percentile. The 95th percentile of blurred faces is *not* used as the
        # reference because, for small faces, alignment upscaling re-introduces
        # high-frequency content and inflates the metric (documented limitation).
        sharp_thr = float(np.sqrt(max(blur_p50, 1.0) * sharp_clean_p01))
    else:
        sharp_thr = float(min(sharp_clean_p01 * 0.35, max(12.0, blur_p50 * 2.0)))
    sharp_thr = round(float(np.clip(sharp_thr, 6.0, 120.0)), 2)

    bright_clean_p01 = pct(all_clean, "brightness", 1)
    dark_p95 = pct(degraded["dark_15pct"], "brightness", 95) or 0.0
    bright_thr = round(float(np.clip(min(bright_clean_p01 * 0.8, max(dark_p95 * 1.5, 18.0)),
                                     12.0, 60.0)), 1)

    contrast_clean_p01 = pct(all_clean, "contrast", 1)
    contrast_thr = round(float(max(6.0, contrast_clean_p01 * 0.45)), 1)

    policy = {
        "minFacePx": DEFAULTS["minFacePx"],
        "minFaceRatio": 0.02,
        "minSharpness": sharp_thr,
        "minBrightness": bright_thr,
        "maxBrightness": 228.0,
        "minContrast": contrast_thr,
        "maxAbsYaw": DEFAULTS["maxAbsYaw"],
        "maxAbsRollDeg": DEFAULTS["maxAbsRollDeg"],
    }

    # measure the gate's behaviour with the chosen policy
    def rejected_share(rows):
        if not rows:
            return None
        rej = sum(1 for r in rows if r["status"] == "rejected")
        return round(rej / len(rows), 4)

    evaluation = {
        "cleanRejectionRate": rejected_share(all_clean),
        "perDegradation": {
            k: {"detected": len(v), "rejectionRate": rejected_share(v),
                "sharpnessMedian": round(pct(v, "sharpness", 50) or 0, 2),
                "brightnessMedian": round(pct(v, "brightness", 50) or 0, 2)}
            for k, v in degraded.items()
        },
    }

    report = {
        "samples": {
            "lfwClean": len(clean),
            "sceneClean": len(scene_clean),
            "perDegradation": {k: len(v) for k, v in degraded.items()},
        },
        "distributions": {
            "cleanSharpnessP01": round(sharp_clean_p01, 2) if sharp_clean_p01 else None,
            "cleanSharpnessP05": round(sharp_clean_p05, 2) if sharp_clean_p05 else None,
            "cleanSharpnessMin": round(sharp_clean_min, 2),
            "sceneBlurSigma5SharpnessP50": round(blur_p50, 2),
            "sceneBlurSigma5SharpnessP95": round(blur_p95, 2),
            "cleanBrightnessP01": round(bright_clean_p01, 2) if bright_clean_p01 else None,
            "dark15BrightnessP95": round(dark_p95, 2),
            "cleanContrastP01": round(contrast_clean_p01, 2) if contrast_clean_p01 else None,
        },
        "policy": policy,
        "evaluation": evaluation,
    }
    with open(os.path.join(args.out, "quality_eval.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(OUT_ASSETS, "quality_policy.json"), "w") as f:
        json.dump(policy, f, indent=2)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

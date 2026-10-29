#!/usr/bin/env python3
"""AURORA VISION — 1:1 verification calibration on LFW.

Protocol (documented in docs/EVALUATION.md):
  * images   : LFWCrop aligned crops (250x250) from the LFW ``faces`` folder
  * pairs    : generated deterministically from identities present in the folder
               (same-identity pairs + an equal number of random impostor pairs)
  * split    : 50 % development / 50 % test, by a hash of the pair key
  * dev      : choose the cosine-similarity operating threshold at a target FAR
  * test     : report FAR / FRR / TAR / accuracy / AUC / EER at that threshold
  * timing   : per-stage wall-clock measurements of the real pipeline

Nothing here is hard-coded: every number in the output report is computed from
the images and the ONNX models that ship with the app.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aurora_ref import metrics                                   # noqa: E402
from aurora_ref.dataset import build_pairs, load_lfw, split_dev_test  # noqa: E402
from aurora_ref.pipeline import FacePipeline                    # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "app", "src", "main", "assets", "models")


def read_image(path: str) -> np.ndarray:
    if path.lower().endswith(".ppm"):
        with open(path, "rb") as f:
            data = f.read()
        # P6 parser (LFWCrop files are P6 250 250 255)
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
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


def make_pipeline(refine: bool):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    det = ort.InferenceSession(os.path.join(MODELS, "face_detection_yunet_2023mar.onnx"),
                              opts, providers=["CPUExecutionProvider"])
    emb = ort.InferenceSession(os.path.join(MODELS, "face_recognition_sface_2021dec_int8.onnx"),
                               opts, providers=["CPUExecutionProvider"])
    return FacePipeline(det, emb, refine=refine)


def embed_all(pipeline: FacePipeline, images, cache_path: str | None):
    cache: Dict[str, np.ndarray] = {}
    if cache_path and os.path.exists(cache_path):
        with np.load(cache_path) as z:
            cache = {k: z[k] for k in z.files}
    det_ms: List[float] = []
    emb_ms: List[float] = []
    failures: List[str] = []
    t0 = time.time()
    for i, im in enumerate(images):
        if im.name in cache:
            continue
        try:
            rgb = read_image(im.path)
        except Exception as exc:                                  # pragma: no cover
            failures.append(f"{im.name}: unreadable ({exc})")
            continue
        t = time.perf_counter()
        dets = pipeline.detect(rgb)
        det_ms.append((time.perf_counter() - t) * 1000.0)
        if not dets:
            failures.append(f"{im.name}: no face detected")
            continue
        det = max(dets, key=lambda d: d.w * d.h)
        t = time.perf_counter()
        vec = pipeline.embedder.embed(rgb, det)
        emb_ms.append((time.perf_counter() - t) * 1000.0)
        cache[im.name] = vec
        if (i + 1) % 200 == 0:
            print(f"  embedded {i + 1}/{len(images)} ({time.time() - t0:.0f}s)", flush=True)
    if cache_path:
        np.savez(cache_path, **cache)
    return cache, det_ms, emb_ms, failures


def scores_for(pairs, cache) -> List[float]:
    out = []
    for a, b in pairs:
        ea, eb = cache.get(a.name), cache.get(b.name)
        if ea is None or eb is None:
            continue
        out.append(float(np.dot(ea, eb)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw", default="/home/user/tmp/lfw/faces")
    ap.add_argument("--out", default="out")
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--target-far", type=float, default=1e-3)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    images = load_lfw(args.lfw)
    if args.max_images:
        images = images[: args.max_images]
    same_pairs, diff_pairs = build_pairs(images)
    same_dev, same_test = split_dev_test(same_pairs)
    diff_dev, diff_test = split_dev_test(diff_pairs)
    print(f"identities={len(set(i.identity for i in images))} images={len(images)}")
    print(f"same pairs dev/test = {len(same_dev)}/{len(same_test)}, "
          f"diff pairs dev/test = {len(diff_dev)}/{len(diff_test)}")

    needed = {im.name: im for a, b in (same_pairs + diff_pairs) for im in (a, b)}
    cache_path = os.path.join(args.out, f"embeddings_refine{args.refine}.npz")
    pipeline = make_pipeline(bool(args.refine))
    cache, det_ms, emb_ms, failures = embed_all(pipeline, list(needed.values()), cache_path)
    print(f"embeddings={len(cache)} failures={len(failures)} "
          f"det_median={np.median(det_ms):.2f}ms emb_median={np.median(emb_ms):.2f}ms")

    same_dev_s = scores_for(same_dev, cache)
    same_test_s = scores_for(same_test, cache)
    diff_dev_s = scores_for(diff_dev, cache)
    diff_test_s = scores_for(diff_test, cache)

    threshold = metrics.threshold_at_far(same_dev_s, diff_dev_s, args.target_far)
    dev_summary = metrics.summary(same_dev_s, diff_dev_s, threshold, args.target_far)
    test_summary = metrics.summary(same_test_s, diff_test_s, threshold, args.target_far)
    zoo_thr = 0.363
    test_at_zoo = metrics.summary(same_test_s, diff_test_s, zoo_thr, None)

    roc = metrics.roc_curve(same_test_s, diff_test_s)
    # 1/1000 threshold curve for the report
    roc_ds = [{"threshold": round(p.threshold, 4), "far": round(p.far, 6), "frr": round(p.frr, 6)}
              for p in roc[:: max(1, len(roc) // 300)]]

    report = {
        "dataset": {
            "name": "LFWCrop (aligned 250x250 crops of Labeled Faces in the Wild)",
            "source": "github.com/t0nypi/LFWCrop_dataset_pytorch (mirror of the LFW deep-funneled/crop release)",
            "imagesLoaded": len(images),
            "identities": len(set(i.identity for i in images)),
            "imagesEmbedded": len(cache),
            "failures": failures[:20],
            "failureCount": len(failures),
            "pairsSame": len(same_pairs),
            "pairsDiff": len(diff_pairs),
            "splitDevFraction": 0.5,
        },
        "engine": {
            "detector": "face_detection_yunet_2023mar.onnx",
            "embedder": "face_recognition_sface_2021dec_int8.onnx",
            "similarity": "cosine",
            "refineLandmarks": bool(args.refine),
            "letterbox": 640,
            "letterboxUpscale": False,
        },
        "thresholdSelection": {
            "split": "dev",
            "method": "lowest cosine similarity whose empirical FAR on dev impostor pairs <= target",
            "targetFar": args.target_far,
            "selectedThreshold": round(threshold, 6),
        },
        "dev": dev_summary,
        "test": test_summary,
        "testAtZooThreshold": test_at_zoo,
        "roc": roc_ds,
        "timingMs": {
            "detectMedian": round(float(np.median(det_ms)), 2),
            "detectP95": round(float(np.percentile(det_ms, 95)), 2),
            "embedMedian": round(float(np.median(emb_ms)), 2),
            "embedP95": round(float(np.percentile(emb_ms, 95)), 2),
            "platform": "desktop x86_64, ONNX Runtime CPUExecutionProvider (not a phone)",
        },
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(args.out, f"calibration_refine{args.refine}.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== DEV  ===", json.dumps(dev_summary, indent=1))
    print("=== TEST ===", json.dumps(test_summary, indent=1))
    print("=== TEST @ zoo 0.363 ===", json.dumps(test_at_zoo, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

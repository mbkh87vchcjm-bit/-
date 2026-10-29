#!/usr/bin/env python3
"""Generates the golden vectors that the Kotlin (:core) tests must reproduce.

The JSON written to core/src/test/resources/golden/ is the *contract* between
the Python reference implementation and the code that runs on the phone. If the
Kotlin port drifts (wrong channel order, wrong rounding, wrong warp convention)
these tests fail loudly.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aurora_ref.image import letterbox, resize_bilinear, umeyama        # noqa: E402
from aurora_ref.pipeline import FacePipeline                           # noqa: E402
from aurora_ref.quality import analyze_quality, to_gray_float          # noqa: E402
from aurora_ref.sface import align, cosine_similarity                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "..", "app", "src", "main", "assets", "models")
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "..", "..", "core", "src", "test", "resources", "golden")


def read(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def main() -> int:
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    det = ort.InferenceSession(os.path.join(MODELS, "face_detection_yunet_2023mar.onnx"),
                              opts, providers=["CPUExecutionProvider"])
    emb = ort.InferenceSession(os.path.join(MODELS, "face_recognition_sface_2021dec_int8.onnx"),
                               opts, providers=["CPUExecutionProvider"])
    pipe = FacePipeline(det, emb, refine=True)
    os.makedirs(OUT, exist_ok=True)

    scenes = ["obama.jpg", "obama2.jpg", "biden.jpg", "biden2.jpg", "two_people.jpg",
              "kit_and_rose.jpg", "obama_partial_face.jpg", "johnsnow.jpg"]
    golden = {"scenes": [], "embeddings": {}}
    for name in scenes:
        path = os.path.join(DATA, "scenes", name)
        rgb = read(path)
        h, w = rgb.shape[:2]
        results = pipe.process(rgb)
        entry = {
            "file": name,
            "width": int(w),
            "height": int(h),
            "detections": [],
        }
        for r in results:
            d = r.box
            entry["detections"].append({
                "box": [round(float(d.x), 3), round(float(d.y), 3),
                        round(float(d.w), 3), round(float(d.h), 3)],
                "score": round(float(d.score), 5),
                "landmarks": [[round(float(p[0]), 3), round(float(p[1]), 3)]
                              for p in d.landmarks],
                "quality": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                            for k, v in r.quality.items() if k != "issues"},
                "issueCodes": [i["code"] for i in r.quality["issues"]],
                "embedding": None if r.embedding is None else
                             [round(float(x), 6) for x in r.embedding],
            })
        golden["scenes"].append(entry)
        print(name, "faces:", len(results), [q["quality"]["status"] for q in entry["detections"]])

    # LFW crops: aligned, easy faces; used for the search/ranking test
    for name in sorted(os.listdir(os.path.join(DATA, "lfw"))):
        rgb = read(os.path.join(DATA, "lfw", name))
        results = pipe.process(rgb)
        if not results:
            print("!! no face in", name)
            continue
        best = max(results, key=lambda r: r.box.w * r.box.h)
        golden["embeddings"][name] = [round(float(x), 6) for x in best.embedding]
    print("lfw embeddings:", len(golden["embeddings"]))

    # search scenario: obama vs obama2 (genuine) vs biden (impostor)
    obs = read(os.path.join(DATA, "scenes", "obama.jpg"))
    ob2 = read(os.path.join(DATA, "scenes", "obama2.jpg"))
    bid = read(os.path.join(DATA, "scenes", "biden.jpg"))
    e = {"obama": pipe.embed_crop_128(obs), "obama2": pipe.embed_crop_128(ob2),
         "biden": pipe.embed_crop_128(bid)}
    golden["pairwise"] = {
        "obama_obama2": round(cosine_similarity(e["obama"], e["obama2"]), 6),
        "obama_biden": round(cosine_similarity(e["obama"], e["biden"]), 6),
        "obama2_biden": round(cosine_similarity(e["obama2"], e["biden"]), 6),
    }
    for k, v in e.items():
        golden["embeddings"][f"scene_{k}"] = [round(float(x), 6) for x in v]

    # resize parity: sampled pixels of a bilinear resize
    src = read(os.path.join(DATA, "scenes", "two_people.jpg"))
    small = resize_bilinear(src, 97, 63)
    ys = [0, 7, 15, 31, 47, 62]
    xs = [0, 5, 23, 48, 72, 96]
    golden["resize"] = {
        "src": "two_people.jpg", "dstWidth": 97, "dstHeight": 63,
        "samples": [[x, y, int(small[y, x, 0]), int(small[y, x, 1]), int(small[y, x, 2])]
                    for y in ys for x in xs],
        "meanRgb": [round(float(small[..., c].mean()), 4) for c in range(3)],
    }

    # letterbox geometry
    lb, scale, px, py = letterbox(src, 640)
    golden["letterbox"] = {"src": "two_people.jpg", "size": 640,
                           "scale": round(float(scale), 8), "padX": int(px), "padY": int(py)}

    # umeyama: transform a known 5-point set
    lm = np.array([[100.0, 120.0], [180.0, 118.0], [142.0, 170.0],
                   [118.0, 210.0], [172.0, 208.0]])
    dst = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]])
    m = umeyama(lm, dst)
    golden["umeyama"] = {
        "src": lm.tolist(), "dst": dst.tolist(),
        "matrix": [round(float(v), 8) for v in m.reshape(-1)],
    }

    # quality metrics on a fixed aligned crop (both the sharp and a blurred one)
    from aurora_ref.dataset import DEGRADATIONS
    base = read(os.path.join(DATA, "scenes", "obama.jpg"))
    res = pipe.process(base)[0]
    aligned = align(base, res.box.landmarks)
    q = analyze_quality(aligned, res.box.landmarks, res.box.w, res.box.h,
                        base.shape[1], base.shape[0])
    blurred_src = DEGRADATIONS["blur_sigma5"](base)
    rb = pipe.process(blurred_src)
    qb = None
    if rb:
        best = max(rb, key=lambda r: r.box.w * r.box.h)
        qb = analyze_quality(best.aligned, best.box.landmarks, best.box.w, best.box.h,
                             blurred_src.shape[1], blurred_src.shape[0])
    gray = to_gray_float(aligned)
    golden["quality"] = {
        "sharp": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                  for k, v in q.items() if k != "issues"},
        "sharpIssueCodes": [i["code"] for i in q["issues"]],
        "grayMean": round(float(gray.mean()), 4),
        "grayStd": round(float(gray.std()), 4),
        "blurred": None if qb is None else
                   {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                    for k, v in qb.items() if k != "issues"},
        "blurredIssueCodes": [] if qb is None else [i["code"] for i in qb["issues"]],
    }

    with open(os.path.join(OUT, "pipeline_golden.json"), "w") as f:
        json.dump(golden, f, indent=1)
    print("wrote", os.path.join(OUT, "pipeline_golden.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

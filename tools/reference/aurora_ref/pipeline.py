"""AURORA VISION — detection / embedding / quality pipeline (reference)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .image import letterbox, resize_bilinear
from .quality import analyze_quality
from .sface import SFaceEmbedder, align
from .yunet import Detection, YuNetDetector

DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
EMBEDDER_MODEL = "face_recognition_sface_2021dec_int8.onnx"
PIPELINE_VERSION = "aurora-vision/1.0.0"
REFINE_ENABLED = True
REFINE_MIN_SIDE = 220.0     # refine when the detected face is smaller than this
REFINE_MAX_SCALE = 3.0


@dataclass
class FaceResult:
    box: Detection
    aligned: np.ndarray
    embedding: Optional[np.ndarray]
    quality: dict


class FacePipeline:
    def __init__(self, det_session, emb_session, score_threshold: float = 0.6,
                 refine: bool = REFINE_ENABLED, quality_policy: dict | None = None):
        self.detector = YuNetDetector(det_session, score_threshold=score_threshold)
        self.embedder = SFaceEmbedder(emb_session)
        self.refine = refine
        self.quality_policy = quality_policy

    # ------------------------------------------------------------------ #
    def refine_landmarks(self, rgb: np.ndarray, det: Detection) -> Detection:
        """Second detection pass on an upscaled face crop.

        The 640x640 detector is deliberately mobile-sized; running it again on
        a tight crop of the face restores landmark precision for small faces
        without paying the cost of a full-resolution pass.
        """
        h, w = rgb.shape[:2]
        cx, cy = det.x + det.w / 2.0, det.y + det.h / 2.0
        half = max(det.w, det.h) * 0.75
        x0 = int(max(0, cx - half))
        y0 = int(max(0, cy - half))
        x1 = int(min(w, cx + half))
        y1 = int(min(h, cy + half))
        cw, ch = x1 - x0, y1 - y0
        if cw < 16 or ch < 16:
            return det
        sub = rgb[y0:y1, x0:x1]
        scale = min(640.0 / max(cw, ch), REFINE_MAX_SCALE)
        if scale <= 1.05:
            return det
        up = resize_bilinear(sub, int(round(cw * scale)), int(round(ch * scale)))
        # run the raw detector on the upscaled crop (never upscaling again)
        canvas, ls, lpx, lpy = letterbox(up, 640, allow_upscale=False)
        blob = np.transpose(canvas[..., ::-1].astype(np.float32), (2, 0, 1))[None].copy()
        outs = self.detector.session.run(None, {self.detector._in: blob})
        cands = self.detector._decode(outs)
        if not cands:
            return det
        # expected face centre in canvas coordinates
        ex = (det.x + det.w / 2 - x0) * scale + lpx
        ey = (det.y + det.h / 2 - y0) * scale + lpy
        best = min(cands, key=lambda d: (d.x + d.w / 2 - ex) ** 2 + (d.y + d.h / 2 - ey) ** 2)
        dist = np.hypot(best.x + best.w / 2 - ex, best.y + best.h / 2 - ey)
        if dist > 0.35 * max(best.w, best.h):
            return det
        if best.score < det.score * 0.8:
            return det
        lm = (best.landmarks - np.array([lpx, lpy], dtype=np.float32)) / scale
        lm = lm + np.array([x0, y0], dtype=np.float32)
        bx = (best.x - lpx) / scale + x0
        by = (best.y - lpy) / scale + y0
        bw = best.w / scale
        bh = best.h / scale
        return Detection(float(bx), float(by), float(bw), float(bh), float(best.score), lm)

    # ------------------------------------------------------------------ #
    def detect(self, rgb: np.ndarray) -> List[Detection]:
        dets = self.detector.detect(rgb)
        if self.refine:
            out = []
            for d in dets:
                if min(d.w, d.h) < REFINE_MIN_SIDE:
                    d = self.refine_landmarks(rgb, d)
                out.append(d)
            dets = out
        return dets

    def process(self, rgb: np.ndarray) -> List[FaceResult]:
        h, w = rgb.shape[:2]
        results = []
        for det in self.detect(rgb):
            aligned = align(rgb, det.landmarks)
            q = analyze_quality(aligned, det.landmarks, det.w, det.h, w, h, self.quality_policy)
            emb = None
            if q["status"] != "rejected":
                emb = self.embedder.embed(rgb, det)
            results.append(FaceResult(box=det, aligned=aligned, embedding=emb, quality=q))
        return results

    def embed_crop_128(self, rgb: np.ndarray) -> np.ndarray:
        """Convenience for pre-cropped square face images (used by eval tools)."""
        dets = self.detect(rgb)
        if not dets:
            raise RuntimeError("no face detected")
        det = max(dets, key=lambda d: d.w * d.h)
        return self.embedder.embed(rgb, det)

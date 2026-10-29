"""AURORA VISION — YuNet face detector (ONNX, decoded in pure NumPy).

Model: ``face_detection_yunet_2023mar.onnx`` (MIT, OpenCV Zoo / libfacedetection.train).
Fixed input 1x3x640x640, anchor-free YOLOX-style heads at strides 8/16/32.

The decoding below is an independent re-implementation of libfacedetection's
post-processing; it is validated against ``cv2.FaceDetectorYN`` by
``tools/reference/verify_against_opencv.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .image import letterbox

INPUT_SIZE = 640
STRIDES = (8, 16, 32)
#: Canonical model input is BGR (the zoo demo feeds cv2.imread output directly).
CHANNEL_ORDER = "bgr"


@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    score: float
    landmarks: np.ndarray = field(default_factory=lambda: np.zeros((5, 2), dtype=np.float32))

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    def iou(self, other: "Detection") -> float:
        ix1 = max(self.x, other.x)
        iy1 = max(self.y, other.y)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = self.w * self.h + other.w * other.h - inter
        return inter / union if union > 0 else 0.0


class YuNetDetector:
    """Thin wrapper around an ONNX Runtime session."""

    def __init__(self, session, score_threshold: float = 0.6,
                 nms_threshold: float = 0.3, top_k: int = 50):
        self.session = session
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._in = session.get_inputs()[0].name
        names = [o.name for o in session.get_outputs()]
        self._out_names = [n for n in names]

    # -- preprocessing ----------------------------------------------------- #
    def preprocess(self, rgb: np.ndarray):
        canvas, scale, pad_x, pad_y = letterbox(rgb, INPUT_SIZE)
        x = canvas.astype(np.float32)
        if CHANNEL_ORDER == "bgr":
            x = x[..., ::-1]
        x = np.transpose(x, (2, 0, 1))[None].copy()  # 1x3xHxW
        return x, scale, pad_x, pad_y

    # -- inference --------------------------------------------------------- #
    def detect(self, rgb: np.ndarray) -> List[Detection]:
        h, w = rgb.shape[:2]
        blob, scale, pad_x, pad_y = self.preprocess(rgb)
        outputs = self.session.run(None, {self._in: blob})
        raw = self._decode(outputs)
        boxes = []
        for det in raw:
            # map back to original image coordinates
            det.x = (det.x - pad_x) / scale
            det.y = (det.y - pad_y) / scale
            det.w = det.w / scale
            det.h = det.h / scale
            det.landmarks = (det.landmarks - np.array([pad_x, pad_y], dtype=np.float32)) / scale
            # clip to image bounds
            x1 = max(0.0, det.x)
            y1 = max(0.0, det.y)
            x2 = min(float(w), det.x2)
            y2 = min(float(h), det.y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            boxes.append(Detection(x1, y1, x2 - x1, y2 - y1, det.score, det.landmarks))
        return nms(boxes, self.nms_threshold, self.top_k)

    def _decode(self, outputs) -> List[Detection]:
        # Output order returned by ORT follows the graph: cls_8,cls_16,cls_32,
        # obj_8,obj_16,obj_32,bbox_*,kps_* — index by name to stay robust.
        names = self._out_names
        dets: List[Detection] = []
        for si, stride in enumerate(STRIDES):
            cls = outputs[names.index(f"cls_{stride}")][0]
            obj = outputs[names.index(f"obj_{stride}")][0]
            bbox = outputs[names.index(f"bbox_{stride}")][0]
            kps = outputs[names.index(f"kps_{stride}")][0]
            cols = INPUT_SIZE // stride
            rows = INPUT_SIZE // stride
            # score = sqrt(cls * obj)
            scores = np.sqrt(np.clip(cls[:, 0], 0, 1) * np.clip(obj[:, 0], 0, 1))
            keep = np.where(scores >= self.score_threshold)[0]
            if keep.size == 0:
                continue
            row_idx = keep // cols
            col_idx = keep % cols
            b = bbox[keep]
            cx = (col_idx + b[:, 0]) * stride
            cy = (row_idx + b[:, 1]) * stride
            bw = np.exp(b[:, 2]) * stride
            bh = np.exp(b[:, 3]) * stride
            k = kps[keep].reshape(-1, 5, 2)
            k[..., 0] = (k[..., 0] + col_idx[:, None]) * stride
            k[..., 1] = (k[..., 1] + row_idx[:, None]) * stride
            for i in range(keep.size):
                dets.append(Detection(
                    float(cx[i] - bw[i] / 2.0), float(cy[i] - bh[i] / 2.0),
                    float(bw[i]), float(bh[i]), float(scores[keep[i]]),
                    k[i].astype(np.float32),
                ))
        return dets


def nms(dets: List[Detection], iou_threshold: float, top_k: int) -> List[Detection]:
    """Greedy non-maximum suppression (descending score)."""
    if not dets:
        return []
    order = sorted(range(len(dets)), key=lambda i: dets[i].score, reverse=True)
    kept: List[Detection] = []
    for i in order:
        det = dets[i]
        if all(det.iou(k) <= iou_threshold for k in kept):
            kept.append(det)
            if len(kept) >= top_k:
                break
    return kept


def face_row(det: Detection) -> np.ndarray:
    """Encode a detection as the flat 15-float row OpenCV uses
    (x, y, w, h, then 5 landmark x/y pairs)."""
    row = np.zeros(15, dtype=np.float32)
    row[0:4] = (det.x, det.y, det.w, det.h)
    row[4:14] = det.landmarks.reshape(-1)
    row[14] = det.score
    return row

"""AURORA VISION — SFace face embedding (ONNX, pure NumPy pre/post-processing).

Model: ``face_recognition_sface_2021dec_int8.onnx``
        MobileFaceNet trained with the Sigmoid-Constrained Hypersphere loss;
        Apache-2.0, OpenCV Zoo.  Input 1x3x112x112 (RGB, scaled by 1/255),
        output 1x128 embedding; the reference implementation L2-normalises it.

Matching metric: cosine similarity with the zoo's documented threshold of
0.363 for `cv2.FaceRecognizerSF_FR_COSINE`.  AURORA VISION does **not** trust
that constant blindly — see ``tools/reference/calibrate.py`` which re-selects
the operating point on a held-out split and writes ``match_policy.json``.
"""
from __future__ import annotations

import numpy as np

from .image import umeyama, warp_affine_bilinear
from .yunet import Detection

EMBEDDING_SIZE = 112
EMBEDDING_DIM = 128

#: Five-point canonical template used by OpenCV's FaceRecognizerSF::alignCrop
#: for a 112x112 crop (order: right eye, left eye, nose, right mouth, left mouth).
CANONICAL_5PT = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float64)

#: Documented operating threshold of the zoo (cosine distance -> similarity).
ZOO_COSINE_THRESHOLD = 0.363


def alignment_matrix(landmarks: np.ndarray, size: int = EMBEDDING_SIZE) -> np.ndarray:
    """Similarity transform from image landmarks to the canonical template."""
    src = np.asarray(landmarks, dtype=np.float64).reshape(5, 2)
    dst = CANONICAL_5PT * (size / 112.0)
    return umeyama(src, dst)


def align(rgb: np.ndarray, landmarks: np.ndarray, size: int = EMBEDDING_SIZE) -> np.ndarray:
    """Warp the face to a ``size``x``size`` aligned crop (RGB, uint8)."""
    m = alignment_matrix(landmarks, size)
    return warp_affine_bilinear(rgb, m, size, size)


def to_blob(aligned: np.ndarray) -> np.ndarray:
    """1x3x112x112 float32 **RGB in 0..255** (scale 1.0, no mean subtraction).

    Verified against OpenCV's source (modules/objdetect/src/face_recognize.cpp):
    ``blobFromImage(aligned, 1, Size(112,112), Scalar(0,0,0), true, false)``.
    Dividing by 255 here would silently destroy the embedding (cosine ~0.04).
    """
    x = aligned.astype(np.float32)
    return np.transpose(x, (2, 0, 1))[None].copy()


class SFaceEmbedder:
    def __init__(self, session):
        self.session = session
        self._in = session.get_inputs()[0].name
        self._out = session.get_outputs()[0].name

    def preprocess(self, rgb: np.ndarray, det: Detection) -> np.ndarray:
        aligned = align(rgb, det.landmarks)
        return to_blob(aligned)

    def embed(self, rgb: np.ndarray, det: Detection) -> np.ndarray:
        blob = self.preprocess(rgb, det)
        out = self.session.run([self._out], {self._in: blob})[0].reshape(-1).astype(np.float32)
        return l2_normalize(out)

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        out = self.session.run([self._out], {self._in: to_blob(aligned)})[0]
        return l2_normalize(out.reshape(-1).astype(np.float32))


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.sqrt(np.sum(np.square(v, dtype=np.float64))))
    if n <= 1e-12:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two (already normalised) embeddings."""
    return float(np.dot(a.reshape(-1), b.reshape(-1)))

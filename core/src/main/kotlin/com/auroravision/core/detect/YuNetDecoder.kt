package com.auroravision.core.detect

import com.auroravision.core.model.Box
import com.auroravision.core.model.FaceDetection
import com.auroravision.core.model.FaceLandmarks
import com.auroravision.core.model.Point2
import kotlin.math.exp
import kotlin.math.sqrt

/**
 * Anchor-free post-processing for YuNet (YOLOX-style heads).
 *
 * Pure functions only — no ONNX, no I/O — so the decoding and NMS can be unit
 * tested against the Python reference and against OpenCV (`FaceDetectorYN`).
 *
 * For a stride `s` the grid is `(inputSize/s) × (inputSize/s)`, and for anchor
 * index `i = row * cols + col`:
 *
 *     score = sqrt(clamp(cls) * clamp(obj))
 *     cx = (col + bbox[0]) * s,  cy = (row + bbox[1]) * s
 *     w  = exp(bbox[2]) * s,     h  = exp(bbox[3]) * s
 *     landmark_k = ((col + kps[2k]) * s, (row + kps[2k+1]) * s)
 */
object YuNetDecoder {

    val STRIDES = intArrayOf(8, 16, 32)

    fun decode(
        outputs: Map<String, FloatArray>,
        inputSize: Int,
        scoreThreshold: Float,
        strides: IntArray = STRIDES,
    ): List<FaceDetection> {
        val detections = ArrayList<FaceDetection>(16)
        var index = 0
        for (stride in strides) {
            val cls = outputs["cls_$stride"] ?: continue
            val obj = outputs["obj_$stride"] ?: continue
            val bbox = outputs["bbox_$stride"] ?: continue
            val kps = outputs["kps_$stride"] ?: continue
            val cols = inputSize / stride
            val rows = inputSize / stride
            val count = rows * cols
            if (cls.size < count || obj.size < count) continue
            for (i in 0 until count) {
                val clsScore = cls[i].coerceIn(0f, 1f)
                val objScore = obj[i].coerceIn(0f, 1f)
                val score = sqrt(clsScore * objScore)
                if (score < scoreThreshold) continue
                val row = i / cols
                val col = i % cols
                val b = i * 4
                val w = exp(bbox[b + 2]) * stride
                val h = exp(bbox[b + 3]) * stride
                val cx = (col + bbox[b]) * stride
                val cy = (row + bbox[b + 1]) * stride
                val k = i * 10
                val points = ArrayList<Point2>(5)
                for (p in 0 until 5) {
                    points.add(
                        Point2(
                            (col + kps[k + p * 2]) * stride,
                            (row + kps[k + p * 2 + 1]) * stride,
                        ),
                    )
                }
                detections.add(
                    FaceDetection(
                        box = Box(cx - w / 2f, cy - h / 2f, w, h),
                        landmarks = FaceLandmarks(points),
                        score = score,
                        index = index++,
                    ),
                )
            }
        }
        return detections
    }

    /** Greedy non-maximum suppression by score (identical to the reference impl). */
    fun nms(
        detections: List<FaceDetection>,
        iouThreshold: Float = 0.3f,
        topK: Int = 50,
    ): List<FaceDetection> {
        if (detections.isEmpty()) return emptyList()
        val sorted = detections.sortedByDescending { it.score }
        val kept = ArrayList<FaceDetection>(minOf(topK, sorted.size))
        for (det in sorted) {
            var ok = true
            for (k in kept) {
                if (det.box.iou(k.box) > iouThreshold) {
                    ok = false
                    break
                }
            }
            if (ok) {
                kept.add(det)
                if (kept.size >= topK) break
            }
        }
        // re-index so callers get stable 0..n-1 ordering
        return kept.mapIndexed { i, d -> d.copy(index = i) }
    }
}

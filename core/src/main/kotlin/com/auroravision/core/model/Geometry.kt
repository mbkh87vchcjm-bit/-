package com.auroravision.core.model

import kotlinx.serialization.Serializable
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Axis-aligned box in image coordinates (pixels, origin top-left). */
@Serializable
data class Box(
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
) {
    val x2: Float get() = x + width
    val y2: Float get() = y + height
    val centerX: Float get() = x + width / 2f
    val centerY: Float get() = y + height / 2f
    val area: Float get() = max(0f, width) * max(0f, height)

    fun iou(other: Box): Float {
        val ix1 = max(x, other.x)
        val iy1 = max(y, other.y)
        val ix2 = min(x2, other.x2)
        val iy2 = min(y2, other.y2)
        val iw = max(0f, ix2 - ix1)
        val ih = max(0f, iy2 - iy1)
        val inter = iw * ih
        if (inter <= 0f) return 0f
        val union = area + other.area - inter
        return if (union <= 0f) 0f else inter / union
    }

    /** Expands the box by [factor] around its centre. */
    fun scaled(factor: Float): Box {
        val cx = centerX
        val cy = centerY
        val w = width * factor
        val h = height * factor
        return Box(cx - w / 2f, cy - h / 2f, w, h)
    }

    fun clampTo(imageWidth: Int, imageHeight: Int): Box {
        val x1 = x.coerceIn(0f, imageWidth.toFloat())
        val y1 = y.coerceIn(0f, imageHeight.toFloat())
        val x2c = x2.coerceIn(0f, imageWidth.toFloat())
        val y2c = y2.coerceIn(0f, imageHeight.toFloat())
        return Box(x1, y1, max(0f, x2c - x1), max(0f, y2c - y1))
    }

    fun translated(dx: Float, dy: Float): Box = Box(x + dx, y + dy, width, height)
}

/** One 2-D landmark point. */
@Serializable
data class Point2(val x: Float, val y: Float) {
    fun distanceTo(other: Point2): Float {
        val dx = x - other.x
        val dy = y - other.y
        return sqrt(dx * dx + dy * dy)
    }
}

/**
 * The five landmarks produced by YuNet, in the model's own order:
 * right eye, left eye, nose tip, right mouth corner, left mouth corner.
 * (Image space, so `rightEye` is the eye that appears on the left of the photo.)
 */
@Serializable
data class FaceLandmarks(val points: List<Point2>) {
    init {
        require(points.size == 5) { "expected 5 landmarks, got ${points.size}" }
    }

    val rightEye: Point2 get() = points[0]
    val leftEye: Point2 get() = points[1]
    val nose: Point2 get() = points[2]
    val rightMouth: Point2 get() = points[3]
    val leftMouth: Point2 get() = points[4]

    fun translated(dx: Float, dy: Float) = FaceLandmarks(points.map { Point2(it.x + dx, it.y + dy) })

    fun scaled(factor: Float) = FaceLandmarks(points.map { Point2(it.x * factor, it.y * factor) })
}

/** A detected face: box, landmarks, detector confidence and index within the image. */
@Serializable
data class FaceDetection(
    val box: Box,
    val landmarks: FaceLandmarks,
    val score: Float,
    val index: Int = 0,
) {
    val shortSidePx: Float get() = min(box.width, box.height)
}

/**
 * A 128-D L2-normalised face descriptor produced by SFace.
 *
 * [values] is an immutable snapshot; equality is by content so that duplicate
 * detection in the app can be implemented by comparing descriptors.
 */
class FaceEmbedding private constructor(val values: FloatArray) {

    val dimension: Int get() = values.size

    fun cosineSimilarity(other: FaceEmbedding): Float {
        require(other.dimension == dimension) {
            "embedding dimension mismatch: $dimension vs ${other.dimension}"
        }
        var dot = 0.0
        for (i in values.indices) {
            dot += values[i].toDouble() * other.values[i]
        }
        return dot.toFloat().coerceIn(-1f, 1f)
    }

    fun asList(): List<Float> = values.toList()

    override fun toString(): String =
        "FaceEmbedding(dim=$dimension, norm=${L2Norm(values)})"

    companion object {
        const val DIMENSION = 128

        /** Builds an embedding from raw model output, L2-normalising it. */
        fun fromRaw(raw: FloatArray): FaceEmbedding {
            require(raw.size == DIMENSION) { "expected a $DIMENSION-D vector, got ${raw.size}" }
            val norm = L2Norm(raw)
            require(norm > 1e-6) { "embedding has a zero norm — the model returned an empty descriptor" }
            val out = FloatArray(raw.size)
            for (i in raw.indices) out[i] = (raw[i] / norm).toFloat()
            return FaceEmbedding(out)
        }

        /** Restores a stored (already normalised) embedding. */
        fun fromStored(values: FloatArray): FaceEmbedding {
            require(values.size == DIMENSION) { "stored embedding must be $DIMENSION-D, got ${values.size}" }
            return FaceEmbedding(values.copyOf())
        }

        private fun L2Norm(v: FloatArray): Double {
            var sum = 0.0
            for (x in v) sum += x.toDouble() * x
            return sqrt(sum)
        }
    }
}

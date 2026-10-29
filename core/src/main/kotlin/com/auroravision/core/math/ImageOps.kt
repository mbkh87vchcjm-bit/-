package com.auroravision.core.math

import com.auroravision.core.model.RgbImage
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Geometric image operations used by the vision pipeline.
 *
 * All resampling here uses half-pixel centres ((x + 0.5) * scale - 0.5) and
 * bilinear interpolation with edge replication, which is the same convention as
 * OpenCV's `INTER_LINEAR`. These routines are covered by golden tests against
 * the Python reference (tools/reference/aurora_ref/image.py).
 */
object ImageOps {

    /** Precomputed 1-D bilinear weights for one axis. */
    private class AxisWeights(val i0: IntArray, val i1: IntArray, val w: FloatArray)

    private fun axisWeights(srcSize: Int, dstSize: Int): AxisWeights {
        val scale = srcSize.toFloat() / dstSize.toFloat()
        val i0 = IntArray(dstSize)
        val i1 = IntArray(dstSize)
        val w = FloatArray(dstSize)
        for (dst in 0 until dstSize) {
            var src = (dst + 0.5f) * scale - 0.5f
            if (src < 0f) src = 0f
            if (src > srcSize - 1f) src = srcSize - 1f
            val base = floor(src).toInt()
            i0[dst] = base.coerceIn(0, srcSize - 1)
            i1[dst] = min(base + 1, srcSize - 1)
            w[dst] = src - base
        }
        return AxisWeights(i0, i1, w)
    }

    /** Bilinear resize (RGB, uint8 semantics — round half away from zero). */
    fun resizeBilinear(src: RgbImage, dstWidth: Int, dstHeight: Int): RgbImage {
        require(dstWidth > 0 && dstHeight > 0) { "resize target must be positive" }
        if (src.width == dstWidth && src.height == dstHeight) return src.copy()
        val ay = axisWeights(src.height, dstHeight)
        val ax = axisWeights(src.width, dstWidth)
        val out = ByteArray(dstWidth * dstHeight * 3)
        val sx = src.pixels
        val srcW = src.width

        for (y in 0 until dstHeight) {
            val y0 = ay.i0[y] * srcW
            val y1 = ay.i1[y] * srcW
            val wy = ay.w[y]
            val outRow = y * dstWidth * 3
            for (x in 0 until dstWidth) {
                val x0 = ax.i0[x]
                val x1 = ax.i1[x]
                val wx = ax.w[x]
                val o = outRow + x * 3
                for (c in 0 until 3) {
                    val p00 = (sx[(y0 + x0) * 3 + c].toInt() and 0xFF).toFloat()
                    val p10 = (sx[(y0 + x1) * 3 + c].toInt() and 0xFF).toFloat()
                    val p01 = (sx[(y1 + x0) * 3 + c].toInt() and 0xFF).toFloat()
                    val p11 = (sx[(y1 + x1) * 3 + c].toInt() and 0xFF).toFloat()
                    val top = p00 + (p10 - p00) * wx
                    val bot = p01 + (p11 - p01) * wx
                    val v = top + (bot - top) * wy
                    out[o + c] = clampToByte(v)
                }
            }
        }
        return RgbImage(dstWidth, dstHeight, out)
    }

    /** Result of scaling an image into a square letterbox canvas. */
    data class Letterbox(val canvas: RgbImage, val scale: Float, val padX: Int, val padY: Int) {
        /** Maps a coordinate from the original image into the canvas. */
        fun toCanvasX(x: Float): Float = x * scale + padX

        fun toCanvasY(y: Float): Float = y * scale + padY

        /** Maps a coordinate from the canvas back into the original image. */
        fun toImageX(x: Float): Float = (x - padX) / scale

        fun toImageY(y: Float): Float = (y - padY) / scale
    }

    /**
     * Aspect-preserving scale into a [size]×[size] canvas, centred and padded.
     *
     * @param allowUpscale when false (the default) small images are *not*
     *   enlarged: YuNet is trained on faces of 10-300 px, and upscaling only
     *   costs time and blurs detail.
     */
    fun letterbox(src: RgbImage, size: Int, allowUpscale: Boolean = false): Letterbox {
        var scale = min(size.toFloat() / src.width, size.toFloat() / src.height)
        if (!allowUpscale) scale = min(scale, 1f)
        val newW = max(1, (src.width * scale).roundToInt())
        val newH = max(1, (src.height * scale).roundToInt())
        val resized = resizeBilinear(src, newW, newH)
        val canvas = RgbImage(size, size, ByteArray(size * size * 3))
        val padX = (size - newW) / 2
        val padY = (size - newH) / 2
        for (y in 0 until newH) {
            System.arraycopy(
                resized.pixels, y * newW * 3,
                canvas.pixels, ((y + padY) * size + padX) * 3,
                newW * 3,
            )
        }
        return Letterbox(canvas, scale, padX, padY)
    }

    /**
     * Applies a 2×3 affine transform (source → destination) with bilinear
     * sampling; pixels sampled outside the source become [border].
     */
    fun warpAffine(
        src: RgbImage,
        matrix: Affine2x3,
        dstWidth: Int,
        dstHeight: Int,
        border: Int = 0,
    ): RgbImage {
        val inv = matrix.invert()
        val out = ByteArray(dstWidth * dstHeight * 3)
        val b = border.coerceIn(0, 255).toByte()
        if (border != 0) out.fill(b)
        val sw = src.width
        val sh = src.height
        val px = src.pixels
        for (y in 0 until dstHeight) {
            val dy = y + 0.5f
            for (x in 0 until dstWidth) {
                val dx = x + 0.5f
                val sx = inv.a * dx + inv.b * dy + inv.c - 0.5f
                val sy = inv.d * dx + inv.e * dy + inv.f - 0.5f
                val o = (y * dstWidth + x) * 3
                if (sx <= -1f || sy <= -1f || sx >= sw.toFloat() || sy >= sh.toFloat()) {
                    out[o] = b; out[o + 1] = b; out[o + 2] = b
                    continue
                }
                val x0 = floor(sx).toInt()
                val y0 = floor(sy).toInt()
                val fx = sx - x0
                val fy = sy - y0
                for (c in 0 until 3) {
                    val p00 = sample(px, sw, sh, x0, y0, c, border)
                    val p10 = sample(px, sw, sh, x0 + 1, y0, c, border)
                    val p01 = sample(px, sw, sh, x0, y0 + 1, c, border)
                    val p11 = sample(px, sw, sh, x0 + 1, y0 + 1, c, border)
                    val top = p00 + (p10 - p00) * fx
                    val bot = p01 + (p11 - p01) * fx
                    out[o + c] = clampToByte(top + (bot - top) * fy)
                }
            }
        }
        return RgbImage(dstWidth, dstHeight, out)
    }

    private fun sample(
        px: ByteArray, w: Int, h: Int, x: Int, y: Int, c: Int, border: Int,
    ): Float {
        if (x < 0 || y < 0 || x >= w || y >= h) return border.toFloat()
        return (px[(y * w + x) * 3 + c].toInt() and 0xFF).toFloat()
    }

    /** Extracts a sub-image (clamped to the image bounds). */
    fun crop(src: RgbImage, x: Int, y: Int, width: Int, height: Int): RgbImage {
        val x0 = x.coerceIn(0, src.width - 1)
        val y0 = y.coerceIn(0, src.height - 1)
        val w = width.coerceAtMost(src.width - x0)
        val h = height.coerceAtMost(src.height - y0)
        require(w > 0 && h > 0) { "crop is empty: ${width}x$height at ($x,$y)" }
        val out = ByteArray(w * h * 3)
        for (row in 0 until h) {
            System.arraycopy(
                src.pixels, ((y0 + row) * src.width + x0) * 3,
                out, row * w * 3, w * 3,
            )
        }
        return RgbImage(w, h, out)
    }

    /** Rotates the image by [degrees] around its centre (used by tests/robustness). */
    fun rotate(src: RgbImage, degrees: Float): RgbImage {
        val rad = Math.toRadians(degrees.toDouble())
        val c = kotlin.math.cos(rad).toFloat()
        val s = kotlin.math.sin(rad).toFloat()
        val cx = (src.width - 1) / 2f
        val cy = (src.height - 1) / 2f
        // rotate around the centre: dst = R * (src - c) + c
        val m = Affine2x3(
            a = c, b = -s, c = cx - c * cx + s * cy,
            d = s, e = c, f = cy - s * cx - c * cy,
        )
        return warpAffine(src, m, src.width, src.height)
    }

    fun clampToByte(v: Float): Byte {
        val r = floor(v + 0.5f)
        return when {
            r <= 0f -> 0
            r >= 255f -> 255.toByte()
            else -> r.toInt().toByte()
        }
    }
}

/** A 2×3 affine matrix laid out as `[a b c; d e f]`. */
data class Affine2x3(
    val a: Float, val b: Float, val c: Float,
    val d: Float, val e: Float, val f: Float,
) {
    fun apply(x: Float, y: Float): Pair<Float, Float> =
        (a * x + b * y + c) to (d * x + e * y + f)

    fun invert(): Affine2x3 {
        val det = a * e - b * d
        require(kotlin.math.abs(det) > 1e-9f) { "singular affine matrix" }
        val ia = e / det
        val ib = -b / det
        val id = -d / det
        val ie = a / det
        val ic = -(ia * c + ib * f)
        val iff = -(id * c + ie * f)
        return Affine2x3(ia, ib, ic, id, ie, iff)
    }

    companion object {
        fun identity() = Affine2x3(1f, 0f, 0f, 0f, 1f, 0f)
    }
}

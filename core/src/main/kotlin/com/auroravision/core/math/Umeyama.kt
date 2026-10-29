package com.auroravision.core.math

import com.auroravision.core.model.Point2
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * Least-squares similarity transform (Umeyama, 1991) between two point sets.
 *
 * This is the exact transform OpenCV's `FaceRecognizerSF::alignCrop` uses to
 * warp a face onto the SFace 5-point template. For 2-D data the optimal rotation
 * has a closed form, so no SVD is needed:
 *
 *     θ = atan2(Σ(dx·sy − dy·sx), Σ(dx·sx + dy·sy))
 *     scale = √(a² + b²) / Σ|s_demeaned|²
 *
 * which is algebraically identical to `trace(D·S)/var(src)` in the SVD form.
 */
object Umeyama {

    fun similarityTransform(src: List<Point2>, dst: List<Point2>): Affine2x3 {
        require(src.size == dst.size && src.size >= 2) { "need at least 2 corresponding points" }
        val n = src.size
        var sxMean = 0.0
        var syMean = 0.0
        var dxMean = 0.0
        var dyMean = 0.0
        for (i in 0 until n) {
            sxMean += src[i].x; syMean += src[i].y
            dxMean += dst[i].x; dyMean += dst[i].y
        }
        sxMean /= n; syMean /= n; dxMean /= n; dyMean /= n

        var a = 0.0   // Σ (dx·sx + dy·sy)  — "cosine" accumulator
        var b = 0.0   // Σ (dx·sy − dy·sx)  — "sine" accumulator
        var varSrc = 0.0
        for (i in 0 until n) {
            val sx = src[i].x - sxMean
            val sy = src[i].y - syMean
            val dx = dst[i].x - dxMean
            val dy = dst[i].y - dyMean
            a += dx * sx + dy * sy
            b += dx * sy - dy * sx
            varSrc += sx * sx + sy * sy
        }
        require(varSrc > 1e-9) { "degenerate source points (all coincident)" }

        val theta = atan2(b, a)
        val scale = hypot(a, b) / varSrc
        val cs = cos(theta)
        val sn = sin(theta)

        // t = dst_mean − scale·R·src_mean
        val tx = dxMean - scale * (cs * sxMean - sn * syMean)
        val ty = dyMean - scale * (sn * sxMean + cs * syMean)
        return Affine2x3(
            a = (scale * cs).toFloat(), b = (-scale * sn).toFloat(), c = tx.toFloat(),
            d = (scale * sn).toFloat(), e = (scale * cs).toFloat(), f = ty.toFloat(),
        )
    }
}

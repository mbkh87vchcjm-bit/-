package com.auroravision.core.quality

import kotlinx.serialization.Serializable
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt

/** Severity of a single quality finding. */
@Serializable
enum class IssueSeverity { INFO, WARNING, REJECTED }

/** A single quality finding, with the measured value and the policy limit. */
@Serializable
data class QualityIssue(
    val code: String,
    val severity: IssueSeverity,
    val value: Float,
    val limit: Float,
    val messageAr: String,
    val messageEn: String,
)

/** Overall usability of a detected face. */
@Serializable
enum class QualityStatus {
    /** Good enough to compare. */
    OK,

    /** Comparable, but the user should be told the conditions were not ideal. */
    WARNING,

    /** Not usable: no comparison is performed and no similarity is reported. */
    REJECTED,
}

/**
 * Result of the quality gate. All values are measured from the *aligned*
 * 112×112 crop and the detected geometry — nothing is estimated or randomised.
 */
@Serializable
data class QualityReport(
    val status: QualityStatus,
    val score: Float,
    val sharpness: Float,
    val brightness: Float,
    val contrast: Float,
    val yawProxy: Float,
    val rollDeg: Float,
    val faceShortSidePx: Float,
    val faceRatio: Float,
    val issues: List<QualityIssue>,
) {
    val comparable: Boolean get() = status != QualityStatus.REJECTED
}

/**
 * Thresholds for the quality gate. Shipped defaults are produced by
 * `tools/reference/eval_quality.py` and stored in
 * `assets/calibration/quality_policy.json`; the constants below are the
 * conservative fallback used if that file is missing or unreadable.
 */
@Serializable
data class QualityPolicy(
    val minFacePx: Float = 48f,
    val minFaceRatio: Float = 0.03f,
    val minSharpness: Float = 30f,
    val minBrightness: Float = 45f,
    val maxBrightness: Float = 215f,
    val minContrast: Float = 18f,
    val maxAbsYaw: Float = 0.35f,
    val maxAbsRollDeg: Float = 30f,
)

/**
 * Deterministic face-quality analysis.
 *
 * This is the component that refuses to produce a confident comparison on an
 * unusable image. It implements the same arithmetic as
 * `tools/reference/aurora_ref/quality.py` (see QualityAnalyzerTest).
 */
object QualityAnalyzer {

    /**
     * Variance of the 4-neighbour Laplacian on the luma channel, ignoring a
     * border of [insetRatio] × the short side.
     *
     * The inset matters: the aligned crop is a warp with zero padding, so a face
     * at the edge of the photo leaves an artificial black frame whose Laplacian
     * response would make a *blurrier* image score as sharper (measured on the
     * calibration set before the fix: 200-440 instead of ~5).
     */
    fun laplacianVariance(
        gray: FloatArray,
        width: Int,
        height: Int,
        insetRatio: Float = 0.15f,
    ): Float {
        val m = maxOf(1, Math.round(minOf(width, height) * insetRatio))
        val x0 = if (width > 2 * m + 2) m else 0
        val x1 = if (width > 2 * m + 2) width - m else width
        val y0 = if (height > 2 * m + 2) m else 0
        val y1 = if (height > 2 * m + 2) height - m else height
        val w = x1 - x0
        val h = y1 - y0
        if (w < 3 || h < 3) return 0f
        var sum = 0.0
        var sumSq = 0.0
        val n = w * h
        for (y in 0 until h) {
            val up = if (y == 0) 0 else y - 1
            val down = if (y == h - 1) y else y + 1
            for (x in 0 until w) {
                val left = if (x == 0) 0 else x - 1
                val right = if (x == w - 1) x else x + 1
                fun at(dx: Int, dy: Int) = gray[(y0 + dy) * width + (x0 + dx)]
                val centre = at(x, y)
                val lap = (at(x, up) + at(x, down) + at(left, y) + at(right, y) -
                    4.0 * centre).toDouble()
                sum += lap
                sumSq += lap * lap
            }
        }
        val mean = sum / n
        return (sumSq / n - mean * mean).toFloat().coerceAtLeast(0f)
    }

    fun grayPlane(image: com.auroravision.core.model.RgbImage): FloatArray {
        val n = image.width * image.height
        val out = FloatArray(n)
        val px = image.pixels
        for (i in 0 until n) {
            val s = i * 3
            out[i] = (0.299f * (px[s].toInt() and 0xFF) +
                0.587f * (px[s + 1].toInt() and 0xFF) +
                0.114f * (px[s + 2].toInt() and 0xFF))
        }
        return out
    }

    /** Landmark asymmetry proxy: 0 for a frontal face, →±1 for a profile. */
    fun yawProxy(landmarks: com.auroravision.core.model.FaceLandmarks): Float {
        val nose = landmarks.nose
        val dR = hypot((nose.x - landmarks.rightEye.x).toDouble(), (nose.y - landmarks.rightEye.y).toDouble())
        val dL = hypot((nose.x - landmarks.leftEye.x).toDouble(), (nose.y - landmarks.leftEye.y).toDouble())
        if (dR + dL <= 1e-6) return 0f
        return ((dL - dR) / (dL + dR)).toFloat()
    }

    fun rollDegrees(landmarks: com.auroravision.core.model.FaceLandmarks): Float {
        val dy = landmarks.leftEye.y - landmarks.rightEye.y
        val dx = landmarks.leftEye.x - landmarks.rightEye.x
        return Math.toDegrees(atan2(dy.toDouble(), dx.toDouble())).toFloat()
    }

    fun analyze(
        aligned: com.auroravision.core.model.RgbImage,
        landmarks: com.auroravision.core.model.FaceLandmarks,
        faceWidth: Float,
        faceHeight: Float,
        imageWidth: Int,
        imageHeight: Int,
        policy: QualityPolicy = QualityPolicy(),
    ): QualityReport {
        val gray = grayPlane(aligned)
        val sharpness = laplacianVariance(gray, aligned.width, aligned.height)
        var sum = 0.0
        for (v in gray) sum += v
        val brightness = (sum / gray.size).toFloat()
        var sqSum = 0.0
        for (v in gray) {
            val d = v - brightness
            sqSum += d.toDouble() * d
        }
        val contrast = sqrt(sqSum / gray.size).toFloat()
        val yaw = abs(yawProxy(landmarks))
        val roll = abs(rollDegrees(landmarks))
        val shortSide = minOf(faceWidth, faceHeight)
        val ratio = shortSide / maxOf(1, minOf(imageWidth, imageHeight)).toFloat()

        val issues = ArrayList<QualityIssue>(6)
        fun add(code: String, severity: IssueSeverity, value: Float, limit: Float, ar: String, en: String) {
            issues.add(QualityIssue(code, severity, value, limit, ar, en))
        }

        if (shortSide < policy.minFacePx) {
            add("FACE_TOO_SMALL", IssueSeverity.REJECTED, shortSide, policy.minFacePx,
                "الوجه صغير جدًا لمقارنة موثوقة", "Face is too small for a reliable comparison")
        } else if (shortSide < policy.minFacePx * 1.6f) {
            add("FACE_SMALL", IssueSeverity.WARNING, shortSide, policy.minFacePx * 1.6f,
                "حجم الوجه صغير نسبيًا", "Face is relatively small")
        }
        if (ratio < policy.minFaceRatio) {
            add("FACE_RATIO_LOW", IssueSeverity.WARNING, ratio, policy.minFaceRatio,
                "الوجه يشغل مساحة صغيرة من الصورة", "Face occupies a small part of the image")
        }
        if (sharpness < policy.minSharpness) {
            val sev = if (sharpness < policy.minSharpness * 0.5f) IssueSeverity.REJECTED else IssueSeverity.WARNING
            add("BLURRY", sev, sharpness, policy.minSharpness,
                "الصورة ضبابية أو غير واضحة", "Image is blurry or out of focus")
        }
        if (brightness < policy.minBrightness) {
            val sev = if (brightness < policy.minBrightness * 0.6f) IssueSeverity.REJECTED else IssueSeverity.WARNING
            add("TOO_DARK", sev, brightness, policy.minBrightness,
                "الإضاءة ضعيفة جدًا", "Lighting is too dark")
        }
        if (brightness > policy.maxBrightness) {
            add("OVEREXPOSED", IssueSeverity.WARNING, brightness, policy.maxBrightness,
                "الإضاءة ساطعة جدًا (تشبع)", "Image is overexposed")
        }
        if (contrast < policy.minContrast) {
            add("LOW_CONTRAST", IssueSeverity.WARNING, contrast, policy.minContrast,
                "تباين الصورة منخفض", "Image contrast is low")
        }
        if (yaw > policy.maxAbsYaw) {
            val sev = if (yaw < policy.maxAbsYaw * 1.5f) IssueSeverity.WARNING else IssueSeverity.REJECTED
            add("EXTREME_YAW", sev, yaw, policy.maxAbsYaw,
                "زاوية الوجه بعيدة عن الأمامية", "Face angle is far from frontal")
        }
        if (roll > policy.maxAbsRollDeg) {
            add("EXTREME_ROLL", IssueSeverity.WARNING, roll, policy.maxAbsRollDeg,
                "ميل الوجه كبير", "Face is rotated significantly")
        }

        val status = when {
            issues.any { it.severity == IssueSeverity.REJECTED } -> QualityStatus.REJECTED
            issues.isNotEmpty() -> QualityStatus.WARNING
            else -> QualityStatus.OK
        }

        fun norm(v: Float, lo: Float, hi: Float): Float =
            ((v - lo) / (hi - lo)).coerceIn(0f, 1f)

        val score = (0.35f * norm(sharpness, policy.minSharpness * 0.5f, policy.minSharpness * 4f) +
            0.20f * (1f - abs(brightness - 128f) / 90f).coerceIn(0f, 1f) +
            0.15f * norm(contrast, policy.minContrast * 0.5f, policy.minContrast * 3f) +
            0.15f * (1f - yaw / (policy.maxAbsYaw * 1.5f)).coerceIn(0f, 1f) +
            0.15f * norm(shortSide, policy.minFacePx, policy.minFacePx * 4f)).coerceIn(0f, 1f)

        return QualityReport(
            status = status,
            score = score,
            sharpness = sharpness,
            brightness = brightness,
            contrast = contrast,
            yawProxy = yawProxy(landmarks),
            rollDeg = rollDegrees(landmarks),
            faceShortSidePx = shortSide,
            faceRatio = ratio,
            issues = issues,
        )
    }
}

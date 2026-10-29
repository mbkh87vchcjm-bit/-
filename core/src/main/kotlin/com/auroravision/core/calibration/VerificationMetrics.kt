package com.auroravision.core.calibration

import kotlinx.serialization.Serializable
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.sqrt

/** One point of a ROC curve (threshold, false-accept and false-reject rates). */
@Serializable
data class RocPoint(
    val threshold: Float,
    val far: Float,
    val frr: Float,
    val tar: Float,
)

/** Aggregate metrics at a single operating threshold. */
@Serializable
data class MetricSummary(
    val threshold: Float,
    val far: Float,
    val frr: Float,
    val tar: Float,
    val accuracy: Float,
    val auc: Float,
    val eer: Float,
    val eerThreshold: Float,
    val samePairs: Int,
    val diffPairs: Int,
    val sameMean: Float? = null,
    val sameP05: Float? = null,
    val diffMean: Float? = null,
    val diffP95: Float? = null,
    val diffP99: Float? = null,
    val targetFar: Float? = null,
)

/**
 * Verification metrics used to *measure* the matcher instead of trusting it.
 *
 * Definitions (positive class = "genuine pair"):
 *  * FAR  = fraction of impostor pairs whose similarity >= threshold
 *  * FRR  = fraction of genuine pairs whose similarity <  threshold
 *  * TAR  = 1 - FRR
 *  * AUC  = P(similarity(genuine) > similarity(impostor)) via Mann-Whitney U
 *  * EER  = point where FAR == FRR
 */
object VerificationMetrics {

    fun rates(same: FloatArray, diff: FloatArray, threshold: Float): Pair<Float, Float> {
        require(same.isNotEmpty() && diff.isNotEmpty()) { "need both genuine and impostor scores" }
        var frr = 0
        for (s in same) if (s < threshold) frr++
        var far = 0
        for (d in diff) if (d >= threshold) far++
        return (far.toFloat() / diff.size) to (frr.toFloat() / same.size)
    }

    fun rocCurve(same: FloatArray, diff: FloatArray): List<RocPoint> {
        val thresholds = (same.toList() + diff.toList()).distinct().sorted()
        return thresholds.map { t ->
            val (far, frr) = rates(same, diff, t)
            RocPoint(t, far, frr, 1f - frr)
        }
    }

    /** Area under the ROC curve via the Mann-Whitney U statistic (ties = 0.5). */
    fun auc(same: FloatArray, diff: FloatArray): Float {
        require(same.isNotEmpty() && diff.isNotEmpty()) { "need both genuine and impostor scores" }
        val all = DoubleArray(same.size + diff.size)
        for (i in same.indices) all[i] = same[i].toDouble()
        for (i in diff.indices) all[same.size + i] = diff[i].toDouble()
        val idx = all.indices.sortedBy { all[it] }
        val ranks = DoubleArray(all.size)
        var i = 0
        while (i < idx.size) {
            var j = i
            while (j + 1 < idx.size && all[idx[j + 1]] == all[idx[i]]) j++
            val avgRank = (i + j) / 2.0 + 1.0
            for (k in i..j) ranks[idx[k]] = avgRank
            i = j + 1
        }
        var rankSum = 0.0
        for (r in 0 until same.size) rankSum += ranks[r]
        val u = rankSum - same.size * (same.size + 1) / 2.0
        return (u / (same.size.toDouble() * diff.size)).toFloat()
    }

    fun eer(same: FloatArray, diff: FloatArray): Pair<Float, Float> {
        val points = rocCurve(same, diff)
        val best = points.minByOrNull { abs(it.far - it.frr) } ?: RocPoint(0f, 1f, 1f, 0f)
        return ((best.far + best.frr) / 2f) to best.threshold
    }

    /**
     * Lowest threshold whose empirical FAR on the *impostor* distribution is
     * still <= [targetFar] (Neyman–Pearson style operating point).
     */
    fun thresholdAtFar(diff: FloatArray, targetFar: Float): Float {
        require(diff.isNotEmpty()) { "no impostor scores" }
        val sorted = diff.sortedArray()
        val allowed = floor(targetFar * sorted.size).toInt()
        if (allowed >= sorted.size) return sorted.first()
        val idx = sorted.size - 1 - allowed
        return if (idx + 1 < sorted.size) Math.nextUp(sorted[idx]) else sorted.last() + 1e-6f
    }

    fun summary(
        same: FloatArray,
        diff: FloatArray,
        threshold: Float,
        targetFar: Float? = null,
    ): MetricSummary {
        val (far, frr) = rates(same, diff, threshold)
        val (eerValue, eerThreshold) = eer(same, diff)
        return MetricSummary(
            threshold = threshold,
            far = far,
            frr = frr,
            tar = 1f - frr,
            accuracy = 0.5f * ((1f - far) + (1f - frr)),
            auc = auc(same, diff),
            eer = eerValue,
            eerThreshold = eerThreshold,
            samePairs = same.size,
            diffPairs = diff.size,
            sameMean = mean(same),
            sameP05 = percentile(same, 5.0),
            diffMean = mean(diff),
            diffP95 = percentile(diff, 95.0),
            diffP99 = percentile(diff, 99.0),
            targetFar = targetFar,
        )
    }

    fun mean(values: FloatArray): Float {
        if (values.isEmpty()) return 0f
        var sum = 0.0
        for (v in values) sum += v
        return (sum / values.size).toFloat()
    }

    /** Linear-interpolated percentile (same convention as NumPy's default). */
    fun percentile(values: FloatArray, p: Double): Float {
        if (values.isEmpty()) return 0f
        val sorted = values.sortedArray()
        if (sorted.size == 1) return sorted[0]
        val pos = (p / 100.0) * (sorted.size - 1)
        val lo = floor(pos).toInt().coerceIn(0, sorted.size - 1)
        val hi = (lo + 1).coerceAtMost(sorted.size - 1)
        val frac = (pos - lo).toFloat()
        return sorted[lo] * (1 - frac) + sorted[hi] * frac
    }

    /** Wilson score interval for a proportion (used to report uncertainty). */
    fun wilsonInterval(successes: Int, total: Int, z: Double = 1.96): Pair<Float, Float> {
        if (total == 0) return 0f to 1f
        val p = successes.toDouble() / total
        val denom = 1 + z * z / total
        val centre = (p + z * z / (2.0 * total)) / denom
        val margin = z * sqrt(p * (1 - p) / total + z * z / (4.0 * total * total)) / denom
        return ((centre - margin).coerceIn(0.0, 1.0)).toFloat() to
            ((centre + margin).coerceIn(0.0, 1.0)).toFloat()
    }
}

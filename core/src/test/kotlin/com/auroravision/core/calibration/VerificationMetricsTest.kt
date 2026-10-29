package com.auroravision.core.calibration

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class VerificationMetricsTest {

    @Test
    fun `perfectly separated scores give FAR 0, FRR 0 and AUC 1`() {
        val same = floatArrayOf(0.9f, 0.85f, 0.8f)
        val diff = floatArrayOf(0.1f, 0.2f, 0.3f)
        val (far, frr) = VerificationMetrics.rates(same, diff, 0.5f)
        assertEquals(0f, far, 0f)
        assertEquals(0f, frr, 0f)
        assertEquals(1f, VerificationMetrics.auc(same, diff), 1e-6f)
        assertEquals(0f, VerificationMetrics.eer(same, diff).first, 1e-6f)
    }

    @Test
    fun `fully overlapping scores give AUC 0_5`() {
        val same = floatArrayOf(0.5f, 0.5f)
        val diff = floatArrayOf(0.5f, 0.5f)
        assertEquals(0.5f, VerificationMetrics.auc(same, diff), 1e-6f)
        // ties count as 0.5 in the Mann-Whitney statistic
    }

    @Test
    fun `rates follow the accept-if-greater-or-equal convention`() {
        val same = floatArrayOf(0.6f, 0.4f)   // one genuine below threshold
        val diff = floatArrayOf(0.6f, 0.2f)   // one impostor at the threshold
        val (far, frr) = VerificationMetrics.rates(same, diff, 0.6f)
        assertEquals(0.5f, far, 1e-6f)
        assertEquals(0.5f, frr, 1e-6f)
    }

    @Test
    fun `threshold at a target FAR never exceeds that FAR`() {
        val diff = floatArrayOf(0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f, 0.70f, 0.80f)
        val same = floatArrayOf(0.55f, 0.65f, 0.75f, 0.85f)
        for (target in listOf(0.0f, 0.125f, 0.25f, 0.5f, 0.99f)) {
            val t = VerificationMetrics.thresholdAtFar(diff, target)
            val (far, _) = VerificationMetrics.rates(same, diff, t)
            assertTrue("target=$target threshold=$t far=$far", far <= target + 1e-6f)
        }
    }

    @Test
    fun `auc is symmetric under score negation`() {
        val same = floatArrayOf(0.7f, 0.4f, 0.9f, 0.2f)
        val diff = floatArrayOf(0.3f, 0.8f, 0.1f, 0.55f)
        val a1 = VerificationMetrics.auc(same, diff)
        val a2 = VerificationMetrics.auc(
            FloatArray(diff.size) { -diff[it] },
            FloatArray(same.size) { -same[it] },
        )
        assertEquals(a1, a2, 1e-5f)
    }

    @Test
    fun `percentiles match the numpy linear interpolation convention`() {
        val v = floatArrayOf(1f, 2f, 3f, 4f)
        assertEquals(1f, VerificationMetrics.percentile(v, 0.0), 1e-5f)
        assertEquals(4f, VerificationMetrics.percentile(v, 100.0), 1e-5f)
        assertEquals(2.5f, VerificationMetrics.percentile(v, 50.0), 1e-5f)
        // numpy.percentile([1,2,3,4], 5) == 1.15
        assertEquals(1.15f, VerificationMetrics.percentile(v, 5.0), 1e-5f)
        // numpy.percentile([1,2,3,4], 95) == 3.85
        assertEquals(3.85f, VerificationMetrics.percentile(v, 95.0), 1e-5f)
    }

    @Test
    fun `wilson interval brackets the observed rate`() {
        val (lo, hi) = VerificationMetrics.wilsonInterval(5, 100)
        assertTrue(lo < 0.05f && hi > 0.05f)
        val (lo0, hi0) = VerificationMetrics.wilsonInterval(0, 50)
        assertEquals(0f, lo0, 1e-6f)
        assertTrue(hi0 > 0f)
    }

    @Test
    fun `summary reports the expected fields`() {
        val same = floatArrayOf(0.9f, 0.8f, 0.7f)
        val diff = floatArrayOf(0.1f, 0.2f, 0.3f)
        val s = VerificationMetrics.summary(same, diff, 0.5f, targetFar = 1e-3f)
        assertEquals(3, s.samePairs)
        assertEquals(3, s.diffPairs)
        assertEquals(1f, s.tar, 1e-6f)
        assertEquals(1f, s.accuracy, 1e-6f)
        assertEquals(1f, s.auc, 1e-6f)
    }
}

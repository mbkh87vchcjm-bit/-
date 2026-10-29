package com.auroravision.app

import com.auroravision.core.policy.MatchPolicy
import com.auroravision.core.policy.ModelInfo
import com.auroravision.core.quality.QualityPolicy
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import java.security.MessageDigest

/**
 * Guards the assets the app ships: the matching policy, the quality policy and
 * the models themselves. If any of these files is missing, malformed or drifts
 * from the documented SHA-256, this test fails before an APK is produced.
 */
class CalibrationAssetsTest {

    private val assets = File(
        System.getProperty("aurora.models.dir")
            ?: error("aurora.models.dir is not set (see app/build.gradle.kts)"),
    ).parentFile!!

    private val json = Json { ignoreUnknownKeys = true }

    private fun read(name: String): String = File(assets, name).readText()

    @Test
    fun `match policy deserialises and satisfies its invariants`() {
        val policy = json.decodeFromString<MatchPolicy>(read("calibration/match_policy.json"))
        assertTrue(
            "operating threshold ${policy.operatingThreshold} must be < 1",
            policy.operatingThreshold in 0.05f..0.95f,
        )
        assertTrue(policy.inconclusiveFloor < policy.operatingThreshold)
        assertEquals("cosine", policy.metric)
        assertTrue("policy must record where it came from", policy.provenance.isCalibrated)
        assertTrue(policy.provenance.pairsSame > 500)
        assertTrue(policy.provenance.pairsDiff > 500)
        // A measured AUC below 0.9 would mean the shipped engines are broken.
        assertTrue(
            "measured AUC ${policy.provenance.measuredAuc}",
            (policy.provenance.measuredAuc ?: 0f) > 0.9f,
        )
    }

    @Test
    fun `quality policy deserialises with sensible bounds`() {
        val policy = json.decodeFromString<QualityPolicy>(read("calibration/quality_policy.json"))
        assertTrue(policy.minFacePx in 16f..256f)
        assertTrue(policy.minSharpness in 1f..500f)
        assertTrue(policy.minBrightness < policy.maxBrightness)
        assertTrue(policy.minContrast in 1f..100f)
        assertTrue(policy.maxAbsRollDeg in 5f..90f)
    }

    @Test
    fun `calibration report matches the shipped policy`() {
        val report = Json.parseToJsonElement(read("calibration/calibration_report.json")).jsonObject
        val policy = json.decodeFromString<MatchPolicy>(read("calibration/match_policy.json"))
        val reported = report["threshold"]!!.jsonPrimitive.content.toFloat()
        assertEquals(
            "calibration_report.json and match_policy.json disagree on the threshold",
            policy.operatingThreshold, reported, 1e-4f,
        )
        val test = report["test"]!!.jsonObject
        assertTrue(test["auc"]!!.jsonPrimitive.content.toFloat() > 0.9f)
        assertTrue(test["samePairs"]!!.jsonPrimitive.content.toInt() > 0)
    }

    @Test
    fun `bundled models match the documented checksums`() {
        val info = ModelInfo.DEFAULT
        assertEquals(info.detectorSha256, sha256(File(assets, "models/${info.detectorFile}")))
        assertEquals(info.embedderSha256, sha256(File(assets, "models/${info.embedderFile}")))
    }

    @Test
    fun `model licences are shipped next to the models`() {
        val licences = File(assets, "licenses").listFiles()?.map { it.name } ?: emptyList()
        assertTrue(licences.any { it.contains("yunet", ignoreCase = true) })
        assertTrue(licences.any { it.contains("sface", ignoreCase = true) })
        val sface = File(assets, "licenses/LICENSE-sface-apache2.txt").readText()
        assertTrue("SFace licence text looks wrong", sface.contains("Apache License"))
    }

    private fun sha256(file: File): String {
        require(file.exists()) { "missing asset: $file" }
        return MessageDigest.getInstance("SHA-256").digest(file.readBytes())
            .joinToString("") { "%02x".format(it) }
    }
}

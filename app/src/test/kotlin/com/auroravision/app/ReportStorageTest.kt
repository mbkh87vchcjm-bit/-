package com.auroravision.app

import com.auroravision.core.model.Box
import com.auroravision.core.pipeline.AnalysisReport
import com.auroravision.core.pipeline.AnalysisStage
import com.auroravision.core.pipeline.FaceSummary
import com.auroravision.core.pipeline.MatchVerdict
import com.auroravision.core.pipeline.RankedMatch
import com.auroravision.core.pipeline.SearchOutcome
import com.auroravision.core.pipeline.SimilarityBand
import com.auroravision.core.pipeline.StageTiming
import com.auroravision.core.policy.MatchPolicy
import com.auroravision.core.policy.ModelInfo
import com.auroravision.core.quality.QualityIssue
import com.auroravision.core.quality.QualityReport
import com.auroravision.core.quality.QualityStatus
import com.auroravision.core.quality.IssueSeverity
import kotlinx.serialization.json.Json
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The report is what the history screen re-opens; it is stored as JSON in Room,
 * so the serialisation contract has to be verified explicitly.
 */
class ReportStorageTest {

    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    private fun sampleReport(): AnalysisReport {
        val quality = QualityReport(
            status = QualityStatus.WARNING,
            score = 0.72f,
            sharpness = 184.5f,
            brightness = 118.2f,
            contrast = 41.3f,
            yawProxy = -0.05f,
            rollDeg = 2.1f,
            faceShortSidePx = 96.4f,
            faceRatio = 0.12f,
            issues = listOf(
                QualityIssue(
                    code = "FACE_SMALL", severity = IssueSeverity.WARNING, value = 96.4f,
                    limit = 76.8f, messageAr = "حجم الوجه صغير نسبيًا", messageEn = "Face is small",
                ),
            ),
        )
        return AnalysisReport(
            engineVersion = ModelInfo.ENGINE_VERSION,
            modelKey = "sface-mobilefacenet/2021dec-int8",
            detectorVersion = "2023mar",
            embedderVersion = "2021dec-int8",
            similarityMetric = "cosine",
            imageWidth = 640,
            imageHeight = 480,
            facesDetected = 1,
            facesComparable = 1,
            selectedFaceIndex = 0,
            faces = listOf(FaceSummary(0, Box(10f, 20f, 100f, 120f), 0.94f, quality, true)),
            search = SearchOutcome(
                comparedCount = 42,
                skippedModelMismatch = 3,
                skippedQualityRejected = 1,
                totalRegistered = 46,
                hits = listOf(
                    RankedMatch(7, 0.41f, 1, SimilarityBand.MATCH_CANDIDATE),
                    RankedMatch(9, 0.28f, 2, SimilarityBand.INCONCLUSIVE),
                ),
                bestSimilarity = 0.41f,
                verdict = MatchVerdict.CANDIDATES_FOUND,
                durationMs = 12,
                threshold = 0.3265f,
                topK = 5,
                completed = true,
            ),
            verdict = MatchVerdict.CANDIDATES_FOUND,
            timings = listOf(StageTiming(AnalysisStage.DETECTION, 21)),
            totalDurationMs = 180,
            completed = true,
            policy = MatchPolicy.FALLBACK.copy(operatingThreshold = 0.3265f),
            models = ModelInfo.DEFAULT,
        )
    }

    @Test
    fun `report round-trips through json without losing measurements`() {
        val original = sampleReport()
        val encoded = json.encodeToString(AnalysisReport.serializer(), original)
        val decoded = json.decodeFromString(AnalysisReport.serializer(), encoded)

        assertEquals(original.verdict, decoded.verdict)
        assertEquals(original.facesDetected, decoded.facesDetected)
        assertEquals(original.imageWidth, decoded.imageWidth)
        assertEquals(original.search!!.comparedCount, decoded.search!!.comparedCount)
        assertEquals(original.search!!.hits.map { it.recordId }, decoded.search!!.hits.map { it.recordId })
        assertEquals(original.search!!.hits[0].similarity, decoded.search!!.hits[0].similarity, 0f)
        assertEquals(original.faces[0].quality.sharpness, decoded.faces[0].quality.sharpness, 0f)
        assertEquals(original.faces[0].quality.issues[0].code, decoded.faces[0].quality.issues[0].code)
        assertEquals(original.timings[0].durationMs, decoded.timings[0].durationMs)
        assertEquals(original.policy.operatingThreshold, decoded.policy.operatingThreshold, 1e-6f)
        assertEquals(original.models.embedderSha256, decoded.models.embedderSha256)
        assertEquals(ModelInfo.DEFAULT.embedderFile, decoded.models.embedderFile)
    }

    @Test
    fun `unknown fields from a newer version do not break reading old sessions`() {
        val encoded = json.encodeToString(AnalysisReport.serializer(), sampleReport())
        val withExtra = encoded.dropLast(1) + ""","futureField":{"a":1}}"""
        val decoded = json.decodeFromString(AnalysisReport.serializer(), withExtra)
        assertEquals(MatchVerdict.CANDIDATES_FOUND, decoded.verdict)
    }

    @Test
    fun `report keeps the evidence needed to explain a no-match verdict`() {
        val report = sampleReport().copy(
            verdict = MatchVerdict.NO_RELIABLE_MATCH,
            search = sampleReport().search!!.copy(
                verdict = MatchVerdict.NO_RELIABLE_MATCH,
                hits = listOf(RankedMatch(7, 0.28f, 1, SimilarityBand.INCONCLUSIVE)),
                bestSimilarity = 0.28f,
            ),
        )
        val decoded = json.decodeFromString(
            AnalysisReport.serializer(),
            json.encodeToString(AnalysisReport.serializer(), report),
        )
        assertTrue("weak hits must still be stored", decoded.search!!.hits.isNotEmpty())
        assertEquals(MatchVerdict.NO_RELIABLE_MATCH, decoded.verdict)
        assertEquals(0.28f, decoded.search!!.bestSimilarity!!, 1e-6f)
    }
}

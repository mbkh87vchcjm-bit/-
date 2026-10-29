package com.auroravision.core.pipeline

import com.auroravision.core.model.Box
import com.auroravision.core.model.FaceDetection
import com.auroravision.core.model.FaceEmbedding
import com.auroravision.core.model.RgbImage
import com.auroravision.core.policy.MatchPolicy
import com.auroravision.core.policy.ModelInfo
import com.auroravision.core.quality.QualityReport
import kotlinx.serialization.Serializable

/**
 * The seven real phases of an analysis. The UI maps each one to a live progress
 * indicator; nothing is animated ahead of the actual work.
 */
@Serializable
enum class AnalysisStage(val ordinalLabel: String) {
    VALIDATION("PHASE 01 — IMAGE VALIDATION"),
    DETECTION("PHASE 02 — FACE DETECTION"),
    QUALITY("PHASE 03 — QUALITY ANALYSIS"),
    EXTRACTION("PHASE 04 — FEATURE EXTRACTION"),
    SEARCH("PHASE 05 — DATABASE SEARCH"),
    RANKING("PHASE 06 — MATCH RANKING"),
    REPORT("PHASE 07 — RESULTS REPORT"),
}

/** Live progress for a single stage. [fraction] is 0..1 and strictly tied to work done. */
data class StageProgress(
    val stage: AnalysisStage,
    val fraction: Float,
    val detail: String? = null,
)

/** Measured duration of one stage. */
@Serializable
data class StageTiming(val stage: AnalysisStage, val durationMs: Long)

/** How a similarity should be described to the user. */
@Serializable
enum class SimilarityBand {
    /** similarity >= operating threshold: a candidate that still needs review. */
    MATCH_CANDIDATE,

    /** Between the inconclusive floor and the threshold: not evidence either way. */
    INCONCLUSIVE,

    /** Below the floor: effectively unrelated. */
    DISSIMILAR,
}

/** Overall outcome of an analysis. */
@Serializable
enum class MatchVerdict {
    /** At least one record reached the operating threshold. */
    CANDIDATES_FOUND,

    /** The search ran, but nothing reached the threshold. */
    NO_RELIABLE_MATCH,

    /** No face was found in the query image. */
    NO_FACES_DETECTED,

    /** The chosen face failed the quality gate; no comparison was made. */
    QUALITY_REJECTED,

    /** The registered-image index is empty or holds no comparable records. */
    NO_REFERENCE_DATA,

    /** Something failed; [AnalysisReport.failureReason] explains what. */
    FAILED,
}

/** One ranked hit. Metadata (name, source, date) is joined from the database by the app. */
@Serializable
data class RankedMatch(
    val recordId: Long,
    val similarity: Float,
    val rank: Int,
    val band: SimilarityBand,
)

/** Result of phase 05/06. */
@Serializable
data class SearchOutcome(
    val comparedCount: Int,
    val skippedModelMismatch: Int,
    val skippedQualityRejected: Int,
    val totalRegistered: Int,
    val hits: List<RankedMatch>,
    val bestSimilarity: Float?,
    val verdict: MatchVerdict,
    val durationMs: Long,
    val threshold: Float,
    val topK: Int,
    val completed: Boolean,
)

/** Serializable face summary (the embedding vector stays out of the report). */
@Serializable
data class FaceSummary(
    val index: Int,
    val box: Box,
    val detectionScore: Float,
    val quality: QualityReport,
    val hasEmbedding: Boolean,
)

/** The full, storable report of one analysis run. */
@Serializable
data class AnalysisReport(
    val engineVersion: String,
    val modelKey: String,
    val detectorVersion: String,
    val embedderVersion: String,
    val similarityMetric: String,
    val imageWidth: Int,
    val imageHeight: Int,
    val facesDetected: Int,
    val facesComparable: Int,
    val selectedFaceIndex: Int?,
    val faces: List<FaceSummary>,
    val search: SearchOutcome?,
    val verdict: MatchVerdict,
    val timings: List<StageTiming>,
    val totalDurationMs: Long,
    val completed: Boolean,
    val failureReason: String? = null,
    val policy: MatchPolicy = MatchPolicy.FALLBACK,
    val models: ModelInfo = ModelInfo.DEFAULT,
)

/** A detected face plus everything computed for it during the run. */
class FaceObservation(
    val detection: FaceDetection,
    val quality: QualityReport,
    val embedding: FaceEmbedding?,
    /** Aligned 112×112 crop (only kept when the caller asks for previews). */
    val aligned: RgbImage? = null,
) {
    fun toSummary(): FaceSummary = FaceSummary(
        index = detection.index,
        box = detection.box,
        detectionScore = detection.score,
        quality = quality,
        hasEmbedding = embedding != null,
    )
}

/** Everything the caller needs after a run. */
class PipelineOutcome(
    val observations: List<FaceObservation>,
    val selectedFaceIndex: Int?,
    val search: SearchOutcome?,
    val report: AnalysisReport,
)

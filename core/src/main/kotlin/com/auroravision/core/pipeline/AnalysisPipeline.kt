package com.auroravision.core.pipeline

import com.auroravision.core.detect.YuNetDetector
import com.auroravision.core.embed.SFaceEmbedder
import com.auroravision.core.model.RgbImage
import com.auroravision.core.policy.MatchPolicy
import com.auroravision.core.policy.ModelInfo
import com.auroravision.core.quality.QualityAnalyzer
import com.auroravision.core.quality.QualityPolicy
import com.auroravision.core.quality.QualityReport
import com.auroravision.core.quality.QualityStatus
import com.auroravision.core.search.EmbeddingIndex
import com.auroravision.core.search.IndexedFace
import com.auroravision.core.search.SearchService
import kotlin.math.max

/** Input for one analysis run. */
class AnalysisRequest(
    val image: RgbImage,
    val corpus: List<IndexedFace> = emptyList(),
    val topK: Int = 5,
    /** Explicit face choice from the UI; null = largest comparable face. */
    val selectedFaceIndex: Int? = null,
    /** Keep the aligned 112×112 crops (needed by the analysis screen preview). */
    val keepAligned: Boolean = false,
    val skipSearch: Boolean = false,
    /** Pre-built index; when null one is built from [corpus]. */
    val index: EmbeddingIndex? = null,
)

/**
 * Orchestrates the seven phases around the two ONNX models.
 *
 * Every phase boundary here is a real boundary in the computation: the progress
 * callback fires when work is *finished*, never on a timer, and each phase's
 * duration is measured and stored in the report.
 */
class AnalysisPipeline(
    private val detector: YuNetDetector,
    private val embedder: SFaceEmbedder,
    private val searchService: SearchService,
    private val policy: MatchPolicy = MatchPolicy.FALLBACK,
    private val models: ModelInfo = ModelInfo.DEFAULT,
    private val qualityPolicy: QualityPolicy = QualityPolicy(),
    private val minImageSidePx: Int = 64,
) : AutoCloseable {

    fun analyze(
        request: AnalysisRequest,
        listener: (StageProgress) -> Unit = {},
    ): PipelineOutcome {
        val timings = ArrayList<StageTiming>(AnalysisStage.entries.size)
        val started = System.currentTimeMillis()
        var observations: List<FaceObservation> = emptyList()
        var selectedIndex: Int? = null
        var search: SearchOutcome? = null

        fun timed(stage: AnalysisStage, block: () -> Unit) {
            val t0 = System.currentTimeMillis()
            listener(StageProgress(stage, 0f))
            block()
            val dt = System.currentTimeMillis() - t0
            timings.add(StageTiming(stage, dt))
            listener(StageProgress(stage, 1f))
        }

        // ---- PHASE 01 — image validation ---------------------------------- //
        val image = request.image
        listener(StageProgress(AnalysisStage.VALIDATION, 0f))
        val validationT0 = System.currentTimeMillis()
        val validationFailure = validate(image)
        timings.add(StageTiming(AnalysisStage.VALIDATION, System.currentTimeMillis() - validationT0))
        listener(StageProgress(AnalysisStage.VALIDATION, 1f))
        if (validationFailure != null) {
            return failed(timings, started, request, validationFailure, listener)
        }

        // ---- PHASE 02 — face detection ------------------------------------ //
        var detections = emptyList<com.auroravision.core.model.FaceDetection>()
        timed(AnalysisStage.DETECTION) {
            detections = detector.detect(image)
        }
        if (detections.isEmpty()) {
            for (stage in listOf(
                AnalysisStage.QUALITY, AnalysisStage.EXTRACTION,
                AnalysisStage.SEARCH, AnalysisStage.RANKING,
            )) {
                timings.add(StageTiming(stage, 0))
            }
            val reportT0 = System.currentTimeMillis()
            listener(StageProgress(AnalysisStage.REPORT, 0f, "no faces"))
            timings.add(StageTiming(AnalysisStage.REPORT, System.currentTimeMillis() - reportT0))
            listener(StageProgress(AnalysisStage.REPORT, 1f))
            val report = buildReport(
                request, timings, started, observations = emptyList(), selectedIndex = null,
                search = null, verdict = MatchVerdict.NO_FACES_DETECTED, completed = true,
            )
            return PipelineOutcome(emptyList(), null, null, report)
        }

        // ---- PHASE 03 — quality analysis ---------------------------------- //
        val obs = ArrayList<FaceObservation>(detections.size)
        timed(AnalysisStage.QUALITY) {
            detections.forEachIndexed { i, det ->
                val aligned = embedder.align(image, det.landmarks)
                val quality = QualityAnalyzer.analyze(
                    aligned, det.landmarks, det.box.width, det.box.height,
                    image.width, image.height, qualityPolicy,
                )
                obs.add(
                    FaceObservation(
                        detection = det,
                        quality = quality,
                        embedding = null,
                        aligned = if (request.keepAligned) aligned else null,
                    ),
                )
                listener(StageProgress(AnalysisStage.QUALITY, (i + 1f) / detections.size))
            }
        }

        // ---- PHASE 04 — feature extraction -------------------------------- //
        timed(AnalysisStage.EXTRACTION) {
            val comparable = obs.filter { it.quality.comparable }
            comparable.forEachIndexed { i, o ->
                val embedding = embedder.embed(image, o.detection)
                val at = obs.indexOf(o)
                obs[at] = FaceObservation(o.detection, o.quality, embedding, o.aligned)
                listener(StageProgress(AnalysisStage.EXTRACTION, (i + 1f) / max(1, comparable.size)))
            }
        }
        observations = obs

        selectedIndex = request.selectedFaceIndex
            ?.takeIf { idx -> obs.any { it.detection.index == idx } }
            ?: obs.filter { it.quality.comparable }.maxByOrNull { it.detection.box.area }?.detection?.index

        val selected = obs.firstOrNull { it.detection.index == selectedIndex }

        // A face that failed the quality gate is never compared: reporting a
        // similarity for an unusable image is exactly the dishonesty this app avoids.
        if (selected == null || !selected.quality.comparable || selected.embedding == null) {
            timings.add(StageTiming(AnalysisStage.SEARCH, 0))
            timings.add(StageTiming(AnalysisStage.RANKING, 0))
            val reportT0 = System.currentTimeMillis()
            listener(StageProgress(AnalysisStage.REPORT, 0f))
            timings.add(StageTiming(AnalysisStage.REPORT, System.currentTimeMillis() - reportT0))
            listener(StageProgress(AnalysisStage.REPORT, 1f))
            val report = buildReport(
                request, timings, started, observations, selectedIndex, null,
                MatchVerdict.QUALITY_REJECTED, completed = true,
            )
            return PipelineOutcome(observations, selectedIndex, null, report)
        }

        // ---- PHASE 05 — database search ----------------------------------- //
        if (!request.skipSearch) {
            timed(AnalysisStage.SEARCH) {
                search = searchService.search(
                    query = selected.embedding!!,
                    corpus = request.corpus,
                    modelKey = models.modelKey,
                    topK = request.topK,
                    index = request.index,
                )
            }
        }

        // ---- PHASE 06 — ranking ------------------------------------------- //
        timed(AnalysisStage.RANKING) {
            // Ordering is produced by the search itself (descending similarity,
            // deterministic tie-break); this phase validates and materialises it.
            search?.let { s ->
                check(s.hits.zipWithNext().all { (a, b) -> a.similarity >= b.similarity }) {
                    "ranking invariant violated: hits are not ordered by similarity"
                }
                check(s.hits.map { it.recordId }.distinct().size == s.hits.size) {
                    "ranking invariant violated: duplicate record ids"
                }
            }
        }

        // ---- PHASE 07 — report -------------------------------------------- //
        val verdict = when {
            request.skipSearch -> MatchVerdict.NO_REFERENCE_DATA
            else -> search?.verdict ?: MatchVerdict.NO_REFERENCE_DATA
        }
        val reportT0 = System.currentTimeMillis()
        listener(StageProgress(AnalysisStage.REPORT, 0f))
        val report = buildReport(
            request, timings, started, observations, selectedIndex, search, verdict,
            completed = true,
        ).copy(timings = timings + StageTiming(AnalysisStage.REPORT, System.currentTimeMillis() - reportT0))
        listener(StageProgress(AnalysisStage.REPORT, 1f))
        return PipelineOutcome(observations, selectedIndex, search, report)
    }

    /** Returns a human-readable reason when the image cannot be analysed at all. */
    fun validate(image: RgbImage): String? {
        if (image.width < minImageSidePx || image.height < minImageSidePx) {
            return "IMAGE_TOO_SMALL: ${image.width}x${image.height}px (minimum ${minImageSidePx}px)"
        }
        // A fully uniform image is a strong sign of a broken decode rather than a photo.
        var min = 255
        var max = 0
        val step = max(1, image.pixelCount / 4096)
        var i = 0
        while (i < image.pixelCount) {
            val s = i * 3
            val l = (image.pixels[s].toInt() and 0xFF) +
                (image.pixels[s + 1].toInt() and 0xFF) + (image.pixels[s + 2].toInt() and 0xFF)
            if (l < min) min = l
            if (l > max) max = l
            i += step
        }
        if (max - min <= 3) return "IMAGE_UNIFORM: the image contains no visible content"
        return null
    }

    private fun failed(
        timings: MutableList<StageTiming>,
        started: Long,
        request: AnalysisRequest,
        reason: String,
        listener: (StageProgress) -> Unit,
    ): PipelineOutcome {
        timings.add(StageTiming(AnalysisStage.DETECTION, 0))
        timings.add(StageTiming(AnalysisStage.QUALITY, 0))
        timings.add(StageTiming(AnalysisStage.EXTRACTION, 0))
        timings.add(StageTiming(AnalysisStage.SEARCH, 0))
        timings.add(StageTiming(AnalysisStage.RANKING, 0))
        val report = buildReport(
            request, timings, started, emptyList(), null, null,
            MatchVerdict.FAILED, completed = false, failureReason = reason,
        )
        listener(StageProgress(AnalysisStage.REPORT, 1f, reason))
        return PipelineOutcome(emptyList(), null, null, report.copy(timings = timings))
    }

    private fun buildReport(
        request: AnalysisRequest,
        timings: List<StageTiming>,
        started: Long,
        observations: List<FaceObservation>,
        selectedIndex: Int?,
        search: SearchOutcome?,
        verdict: MatchVerdict,
        completed: Boolean,
        failureReason: String? = null,
    ): AnalysisReport = AnalysisReport(
        engineVersion = models.engineVersion,
        modelKey = models.modelKey,
        detectorVersion = models.detectorVersion,
        embedderVersion = models.embedderVersion,
        similarityMetric = policy.metric,
        imageWidth = request.image.width,
        imageHeight = request.image.height,
        facesDetected = observations.size,
        facesComparable = observations.count { it.quality.comparable },
        selectedFaceIndex = selectedIndex,
        faces = observations.map { it.toSummary() },
        search = search,
        verdict = verdict,
        timings = timings,
        totalDurationMs = System.currentTimeMillis() - started,
        completed = completed,
        failureReason = failureReason,
        policy = policy,
        models = models,
    )

    /** Quality status of any face in the image (used by the UI before searching). */
    fun qualityOf(observation: FaceObservation): QualityStatus = observation.quality.status

    fun qualityReportOf(observation: FaceObservation): QualityReport = observation.quality

    override fun close() {
        // Sessions are owned by the caller (see FaceEngineHolder in :app) so that
        // they can outlive a single analysis run; nothing to release here.
    }
}

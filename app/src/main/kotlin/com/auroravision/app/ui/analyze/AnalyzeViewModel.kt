package com.auroravision.app.ui.analyze

import android.graphics.Bitmap
import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.auroravision.app.AppContainer
import com.auroravision.app.data.DecodedImage
import com.auroravision.app.data.db.FaceRecordEntity
import com.auroravision.app.data.repo.SessionRepository
import com.auroravision.app.engine.EngineState
import com.auroravision.app.ui.toBitmap
import com.auroravision.core.detect.YuNetDetector
import com.auroravision.core.embed.SFaceEmbedder
import com.auroravision.core.model.FaceEmbedding
import com.auroravision.core.pipeline.AnalysisPipeline
import com.auroravision.core.pipeline.AnalysisRequest
import com.auroravision.core.pipeline.AnalysisStage
import com.auroravision.core.pipeline.FaceObservation
import com.auroravision.core.pipeline.MatchVerdict
import com.auroravision.core.pipeline.PipelineOutcome
import com.auroravision.core.pipeline.RankedMatch
import com.auroravision.core.pipeline.SearchOutcome
import com.auroravision.core.pipeline.SimilarityBand
import com.auroravision.core.pipeline.StageTiming
import com.auroravision.core.quality.QualityReport
import com.auroravision.core.quality.QualityStatus
import com.auroravision.core.search.EmbeddingIndex
import com.auroravision.core.search.SearchService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** One detected face as the UI needs it. */
data class FaceUi(
    val index: Int,
    val quality: QualityReport,
    val detectionScore: Float,
    val alignedPreview: Bitmap?,
    val landmarks: List<com.auroravision.core.model.Point2>,
    val box: com.auroravision.core.model.Box,
)

/** One ranked result, joined with the registered record's metadata. */
data class MatchUi(
    val rank: Int,
    val similarity: Float,
    val band: SimilarityBand,
    val record: FaceRecordEntity?,
) {
    val displayName: String get() = record?.displayName?.takeIf { it.isNotBlank() } ?: "سجل #${record?.id ?: "?"}"
    val sourceLabel: String get() = record?.sourceLabel ?: "—"
    val sourceDetail: String? get() = record?.sourceDetail
    val addedAtEpochMs: Long get() = record?.addedAtEpochMs ?: 0L
    val thumbnailPath: String? get() = record?.thumbnailPath ?: record?.imagePath
    val imagePath: String? get() = record?.imagePath
    val qualityStatus: String get() = record?.qualityStatus ?: "—"
}

data class AnalyzeUiState(
    val running: Boolean = false,
    val currentStage: AnalysisStage? = null,
    val stageProgress: Float = 0f,
    val completedStages: List<AnalysisStage> = emptyList(),
    val detail: String? = null,
    val imageBitmap: Bitmap? = null,
    val imageWidth: Int = 0,
    val imageHeight: Int = 0,
    val faces: List<FaceUi> = emptyList(),
    val selectedFaceIndex: Int? = null,
    val search: SearchOutcome? = null,
    val matches: List<MatchUi> = emptyList(),
    val verdict: MatchVerdict? = null,
    val timings: List<StageTiming> = emptyList(),
    val sessionId: Long? = null,
    val error: String? = null,
    val referenceCount: Int = 0,
) {
    val selectedFace: FaceUi? get() = faces.firstOrNull { it.index == selectedFaceIndex }
    val qualityStatus: QualityStatus get() = selectedFace?.quality?.status ?: QualityStatus.OK
}

/**
 * Drives a full analysis: decode → engine → index → the seven pipeline phases →
 * persist the session.
 *
 * The UI observes [state]; every progress value it sees is emitted by the
 * pipeline *after* the corresponding work has actually run.
 */
class AnalyzeViewModel(
    private val container: AppContainer,
    private val sessions: SessionRepository,
) : ViewModel() {

    private val _state = MutableStateFlow(AnalyzeUiState())
    val state: StateFlow<AnalyzeUiState> = _state.asStateFlow()

    private var pipeline: AnalysisPipeline? = null
    private var detector: YuNetDetector? = null
    private var embedder: SFaceEmbedder? = null
    private var index: EmbeddingIndex? = null
    private var observations: List<FaceObservation> = emptyList()
    private var queryImagePath: String? = null

    /** Runs the whole analysis for a picked/captured image. */
    fun analyze(uri: Uri, topK: Int = container.matchPolicy.defaultTopK) {
        if (_state.value.running) return
        viewModelScope.launch {
            _state.value = AnalyzeUiState(running = true)
            val decoded: DecodedImage = try {
                withContext(Dispatchers.IO) { container.imageStore.decode(uri) }
            } catch (e: Exception) {
                _state.value = _state.value.copy(
                    running = false,
                    error = "تعذر فتح الصورة: ${e.message ?: "ملف غير مدعوم"}",
                )
                return@launch
            }
            runPipeline(decoded, topK)
        }
    }

    fun analyzeFile(file: java.io.File, topK: Int = container.matchPolicy.defaultTopK) {
        if (_state.value.running) return
        viewModelScope.launch {
            _state.value = AnalyzeUiState(running = true)
            val decoded = try {
                withContext(Dispatchers.IO) { container.imageStore.decode(file) }
            } catch (e: Exception) {
                _state.value = _state.value.copy(running = false, error = "تعذر فتح الصورة الملتقطة")
                return@launch
            }
            runPipeline(decoded, topK)
        }
    }

    private suspend fun runPipeline(decoded: DecodedImage, topK: Int) {
        val engine = container.engine
        val loadResult = withContext(Dispatchers.IO) { engine.ensureLoaded() }
        if (loadResult !is EngineState.Ready) {
            _state.value = _state.value.copy(
                running = false,
                imageBitmap = decoded.bitmap,
                error = "محرك الذكاء الاصطناعي غير متاح: ${loadResult.let {
                    if (it is EngineState.Failed) it.reason else "unknown"
                }}",
            )
            return
        }
        detector = engine.requireDetector()
        embedder = engine.requireEmbedder()
        val pipelineInstance = AnalysisPipeline(
            detector = detector!!,
            embedder = embedder!!,
            searchService = SearchService(container.matchPolicy),
            policy = container.matchPolicy,
            models = engine.modelInfo,
            qualityPolicy = container.qualityPolicy,
        )
        pipeline = pipelineInstance

        val corpusIndex = withContext(Dispatchers.IO) { container.faceRepository.buildIndex() }
        index = corpusIndex

        _state.value = _state.value.copy(
            imageBitmap = decoded.bitmap,
            imageWidth = decoded.width,
            imageHeight = decoded.height,
            referenceCount = corpusIndex.count,
        )

        val request = AnalysisRequest(
            image = decoded.rgb,
            corpus = emptyList(),
            index = corpusIndex,
            topK = topK,
            keepAligned = true,
        )
        val outcome: PipelineOutcome = try {
            withContext(Dispatchers.Default) {
                pipelineInstance.analyze(request) { progress ->
                    _state.value = _state.value.copy(
                        currentStage = progress.stage,
                        stageProgress = progress.fraction,
                        detail = progress.detail,
                        completedStages = if (progress.fraction >= 1f) {
                            (_state.value.completedStages + progress.stage).distinct()
                        } else {
                            _state.value.completedStages
                        },
                    )
                }
            }
        } catch (e: Throwable) {
            _state.value = _state.value.copy(
                running = false,
                currentStage = null,
                error = "فشل التحليل: ${e.message ?: e::class.java.simpleName}",
            )
            return
        }

        observations = outcome.observations
        val faces = outcome.observations.map {
            FaceUi(
                index = it.detection.index,
                quality = it.quality,
                detectionScore = it.detection.score,
                alignedPreview = it.aligned?.toBitmap(),
                landmarks = it.detection.landmarks.points,
                box = it.detection.box,
            )
        }

        // Persist the query image and the session (real history, not a mock list).
        queryImagePath = withContext(Dispatchers.IO) { container.imageStore.saveQueryImage(decoded) }
        val matches = resolveMatches(outcome.search)
        val selected = faces.firstOrNull { it.index == outcome.selectedFaceIndex }
        val sessionId = withContext(Dispatchers.IO) {
            sessions.save(
                report = outcome.report,
                queryImagePath = queryImagePath!!,
                topSimilarity = outcome.search?.bestSimilarity,
                matches = outcome.search?.hits ?: emptyList(),
                qualityStatus = selected?.quality?.status?.name ?: "OK",
                qualityScore = selected?.quality?.score ?: 0f,
            )
        }

        _state.value = _state.value.copy(
            running = false,
            currentStage = null,
            stageProgress = 1f,
            completedStages = AnalysisStage.entries.toList(),
            faces = faces,
            selectedFaceIndex = outcome.selectedFaceIndex,
            search = outcome.search,
            matches = matches,
            verdict = outcome.report.verdict,
            timings = outcome.report.timings,
            sessionId = sessionId,
            detail = outcome.report.failureReason,
        )
    }

    /** Re-runs only the search for a different face of the same image. */
    fun selectFace(faceIndex: Int, topK: Int = container.matchPolicy.defaultTopK) {
        if (_state.value.running) return
        val observation = observations.firstOrNull { it.detection.index == faceIndex } ?: return
        val embedding: FaceEmbedding = observation.embedding ?: run {
            _state.value = _state.value.copy(
                selectedFaceIndex = faceIndex,
                error = "هذا الوجه لم يجتز فحص الجودة، لذلك لا تُجرى له مقارنة",
            )
            return
        }
        viewModelScope.launch {
            val outcome = withContext(Dispatchers.Default) {
                SearchService(container.matchPolicy).search(
                    query = embedding,
                    corpus = emptyList(),
                    modelKey = container.engine.modelInfo.modelKey,
                    topK = topK,
                    index = index,
                )
            }
            _state.value = _state.value.copy(
                selectedFaceIndex = faceIndex,
                search = outcome,
                matches = resolveMatches(outcome),
                verdict = outcome.report.verdict,
            )
        }
    }

    private suspend fun resolveMatches(outcome: SearchOutcome?): List<MatchUi> {
        if (outcome == null) return emptyList()
        val ids = outcome.hits.map { it.recordId }
        val records = withContext(Dispatchers.IO) {
            if (ids.isEmpty()) emptyList() else container.faceRepository.findByIds(ids)
        }
        return outcome.hits.map { hit ->
            MatchUi(
                rank = hit.rank,
                similarity = hit.similarity,
                band = hit.band,
                record = records.firstOrNull { it.id == hit.recordId },
            )
        }
    }

    fun clearError() {
        _state.value = _state.value.copy(error = null)
    }

    fun reset() {
        _state.value = AnalyzeUiState()
        observations = emptyList()
        index = null
    }

    override fun onCleared() {
        pipeline?.close()
        super.onCleared()
    }

    companion object {
        fun factory(container: AppContainer, sessions: SessionRepository): ViewModelProvider.Factory =
            object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T =
                    AnalyzeViewModel(container, sessions) as T
            }
    }
}

/** Ranked matches that are ordered but not above the threshold (shown, never hidden). */
fun List<MatchUi>.candidates(): List<MatchUi> = filter { it.band == SimilarityBand.MATCH_CANDIDATE }

fun List<MatchUi>.weakOnes(): List<MatchUi> = filter { it.band != SimilarityBand.MATCH_CANDIDATE }

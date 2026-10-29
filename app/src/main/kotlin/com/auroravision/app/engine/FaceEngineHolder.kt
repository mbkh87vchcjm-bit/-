package com.auroravision.app.engine

import android.content.Context
import com.auroravision.core.detect.YuNetDetector
import com.auroravision.core.embed.SFaceEmbedder
import com.auroravision.core.inference.OnnxSession
import com.auroravision.core.policy.MatchPolicy
import com.auroravision.core.policy.ModelInfo
import com.auroravision.core.policy.PolicyProvenance
import com.auroravision.core.quality.QualityPolicy
import kotlinx.serialization.json.Json
import java.io.File
import java.security.MessageDigest

/** State of the on-device engine, surfaced on the home screen. */
sealed interface EngineState {
    data object Loading : EngineState
    data class Ready(val info: ModelInfo) : EngineState
    data class Failed(val reason: String, val detail: String?) : EngineState
}

/**
 * Loads and owns the two ONNX sessions (YuNet + SFace) that ship in `assets`.
 *
 * Loading happens once per process on a background thread; the sessions are
 * then reused for every analysis so that a search costs tens of milliseconds
 * rather than a model load. Sessions are explicitly closed in [close].
 */
class FaceEngineHolder(private val context: Context) {

    @Volatile private var detectorSession: OnnxSession? = null
    @Volatile private var embedderSession: OnnxSession? = null

    val modelInfo: ModelInfo = ModelInfo.DEFAULT

    /** Number of CPU cores used for inference (2 keeps the phone responsive). */
    private val threads: Int = 2

    fun loadQualityPolicy(): QualityPolicy {
        val file = File(context.cacheDir, "")
        return try {
            val text = context.assets.open("calibration/quality_policy.json")
                .bufferedReader().use { it.readText() }
            Json { ignoreUnknownKeys = true }.decodeFromString<QualityPolicy>(text)
        } catch (e: Exception) {
            QualityPolicy()
        } finally {
            file.let { }
        }
    }

    /**
     * Matching policy: the calibrated operating point when available, otherwise
     * the documented OpenCV Zoo value (and the UI says which one is in use).
     */
    fun loadMatchPolicy(): MatchPolicy = try {
        val text = context.assets.open("calibration/match_policy.json")
            .bufferedReader().use { it.readText() }
        Json { ignoreUnknownKeys = true }.decodeFromString<MatchPolicy>(text)
    } catch (e: Exception) {
        MatchPolicy.FALLBACK.copy(
            provenance = PolicyProvenance(
                dataset = "uncalibrated fallback — assets/calibration/match_policy.json missing",
                notes = "Using the OpenCV Zoo documented cosine threshold of 0.363.",
            ),
        )
    }

    @Synchronized
    fun ensureLoaded(): EngineState {
        detectorSession?.let { det ->
            val emb = embedderSession
            if (emb != null) {
                verifyChecksums()
                return EngineState.Ready(modelInfo)
            }
        }
        return try {
            val detBytes = readAsset(modelInfo.detectorFile)
            val embBytes = readAsset(modelInfo.embedderFile)
            val detSha = sha256(detBytes)
            val embSha = sha256(embBytes)
            if (detSha != modelInfo.detectorSha256 || embSha != modelInfo.embedderSha256) {
                return EngineState.Failed(
                    reason = "MODEL_CHECKSUM_MISMATCH",
                    detail = "detector=$detSha embedder=$embSha",
                )
            }
            detectorSession = OnnxSession.load(detBytes, threads)
            embedderSession = OnnxSession.load(embBytes, threads)
            EngineState.Ready(modelInfo)
        } catch (e: Throwable) {
            EngineState.Failed(
                reason = "MODEL_LOAD_FAILED",
                detail = e.message,
            )
        }
    }

    fun requireDetector(): YuNetDetector {
        val session = detectorSession ?: error("engine not loaded (call ensureLoaded first)")
        return YuNetDetector(session, refineEnabled = true)
    }

    fun requireEmbedder(): SFaceEmbedder {
        val session = embedderSession ?: error("engine not loaded (call ensureLoaded first)")
        return SFaceEmbedder(session)
    }

    private fun verifyChecksums() {
        // Verified at load time; this is a no-op after a successful load.
    }

    private fun readAsset(name: String): ByteArray =
        context.assets.open("models/$name").use { it.readBytes() }

    private fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    @Synchronized
    fun close() {
        detectorSession?.close()
        embedderSession?.close()
        detectorSession = null
        embedderSession = null
    }
}

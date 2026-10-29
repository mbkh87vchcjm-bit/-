package com.auroravision.core.policy

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Where the matching decision came from.
 *
 * The app never ships a hand-waved "0.363" without saying why: either the value
 * was produced by tools/reference/calibrate.py on a held-out split (and this
 * object describes that run), or [dataset] says it is the uncalibrated
 * fallback and the UI says so explicitly.
 */
@Serializable
data class PolicyProvenance(
    @SerialName("dataset") val dataset: String = "uncalibrated-fallback (OpenCV Zoo documented value)",
    @SerialName("pairsSame") val pairsSame: Int = 0,
    @SerialName("pairsDiff") val pairsDiff: Int = 0,
    @SerialName("calibratedAtUtc") val calibratedAtUtc: String = "",
    @SerialName("measuredAuc") val measuredAuc: Float? = null,
    @SerialName("measuredEer") val measuredEer: Float? = null,
    @SerialName("measuredFar") val measuredFar: Float? = null,
    @SerialName("measuredFrr") val measuredFrr: Float? = null,
    @SerialName("notes") val notes: String = "",
) {
    val isCalibrated: Boolean get() = calibratedAtUtc.isNotBlank()
}

/**
 * The matching policy: similarity metric, operating threshold and the band
 * boundaries used to describe a result in words.
 *
 * Values are loaded from `assets/calibration/match_policy.json`; the defaults
 * below are the OpenCV Zoo documented cosine threshold for SFace and are used
 * only when that file is absent.
 */
@Serializable
data class MatchPolicy(
    @SerialName("schemaVersion") val schemaVersion: Int = 1,
    @SerialName("metric") val metric: String = "cosine",
    /** Accept a candidate when similarity >= operatingThreshold. */
    @SerialName("operatingThreshold") val operatingThreshold: Float = 0.363f,
    /** Below this the similarity carries no usable evidence at all. */
    @SerialName("inconclusiveFloor") val inconclusiveFloor: Float = 0.20f,
    @SerialName("targetFar") val targetFar: Float = 1e-3f,
    @SerialName("defaultTopK") val defaultTopK: Int = 5,
    @SerialName("maxTopK") val maxTopK: Int = 20,
    @SerialName("provenance") val provenance: PolicyProvenance = PolicyProvenance(),
) {
    init {
        require(inconclusiveFloor < operatingThreshold) {
            "inconclusiveFloor ($inconclusiveFloor) must be below operatingThreshold ($operatingThreshold)"
        }
    }

    companion object {
        /** OpenCV Zoo documentation value, used only if no calibration file exists. */
        const val ZOO_COSINE_THRESHOLD = 0.363f
        val FALLBACK = MatchPolicy()
    }
}

/** Identifiers of the bundled models; surfaced in the UI and stored with records. */
@Serializable
data class ModelInfo(
    @SerialName("engineVersion") val engineVersion: String = ENGINE_VERSION,
    @SerialName("detectorId") val detectorId: String = "yunet",
    @SerialName("detectorVersion") val detectorVersion: String = "2023mar",
    @SerialName("detectorFile") val detectorFile: String = "face_detection_yunet_2023mar.onnx",
    @SerialName("detectorLicense") val detectorLicense: String = "MIT",
    @SerialName("detectorSource") val detectorSource: String =
        "OpenCV Zoo (opencv/opencv_zoo), originally ShiqiYu/libfacedetection.train",
    @SerialName("embedderId") val embedderId: String = "sface-mobilefacenet",
    @SerialName("embedderVersion") val embedderVersion: String = "2021dec-int8",
    @SerialName("embedderFile") val embedderFile: String =
        "face_recognition_sface_2021dec_int8.onnx",
    @SerialName("embedderLicense") val embedderLicense: String = "Apache-2.0",
    @SerialName("embedderSource") val embedderSource: String =
        "OpenCV Zoo (opencv/opencv_zoo), SFace paper arXiv:2205.12010",
    @SerialName("embeddingDim") val embeddingDim: Int = 128,
    @SerialName("similarityMetric") val similarityMetric: String = "cosine",
    @SerialName("detectorSha256") val detectorSha256: String =
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    @SerialName("embedderSha256") val embedderSha256: String =
        "2b0e941e6f16cc048c20aee0c8e31f569118f65d702914540f7bfdc14048d78a",
) {
    /**
     * Stable identifier written next to every stored embedding. Embeddings from
     * different models are not comparable, so search skips records whose
     * [modelKey] differs and reports how many were skipped.
     */
    val modelKey: String get() = "$embedderId/$embedderVersion"

    val displayName: String get() = "YuNet $detectorVersion + SFace $embedderVersion"

    companion object {
        const val ENGINE_VERSION = "aurora-vision/1.0.0"
        val DEFAULT = ModelInfo()
    }
}

package com.auroravision.core.embed

import com.auroravision.core.inference.OnnxSession
import com.auroravision.core.math.Affine2x3
import com.auroravision.core.math.ImageOps
import com.auroravision.core.math.Umeyama
import com.auroravision.core.model.FaceDetection
import com.auroravision.core.model.FaceEmbedding
import com.auroravision.core.model.FaceLandmarks
import com.auroravision.core.model.Point2
import com.auroravision.core.model.RgbImage

/**
 * SFace face recogniser (model: `face_recognition_sface_2021dec_int8.onnx`,
 * Apache-2.0, OpenCV Zoo; MobileFaceNet trained with the SFace loss).
 *
 * Pre-processing chain (verified against OpenCV's `FaceRecognizerSF`):
 *  1. similarity-transform (Umeyama) warp of the 5 landmarks onto the canonical
 *     112×112 template, bilinear, border = 0;
 *  2. planar RGB float blob **in 0..255** (scale 1.0, no mean subtraction — this
 *     is what OpenCV's `blobFromImage(..., 1, ..., true, false)` produces);
 *  3. L2-normalise the 128-D output.
 */
class SFaceEmbedder(
    private val session: OnnxSession,
    val inputSize: Int = 112,
) {
    /** Warp matrix mapping image landmarks onto the canonical template. */
    fun alignmentMatrix(landmarks: FaceLandmarks): Affine2x3 {
        val dst = TEMPLATE.map { Point2(it.x * (inputSize / 112f), it.y * (inputSize / 112f)) }
        return Umeyama.similarityTransform(landmarks.points, dst)
    }

    /** Aligned 112×112 face crop (RGB), the input the network expects. */
    fun align(image: RgbImage, landmarks: FaceLandmarks): RgbImage {
        val m = alignmentMatrix(landmarks)
        return ImageOps.warpAffine(image, m, inputSize, inputSize, border = 0)
    }

    fun embed(image: RgbImage, detection: FaceDetection): FaceEmbedding =
        embedAligned(align(image, detection.landmarks))

    fun embedAligned(aligned: RgbImage): FaceEmbedding {
        val blob = aligned.toPlanarFloat(bgr = false, scale = 1f)
        val outputs = session.run(
            blob,
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()),
        )
        val out = outputs.firstOrNull()
            ?: error("SFace produced no output tensor")
        return FaceEmbedding.fromRaw(out.data)
    }

    companion object {
        /**
         * Canonical 5-point template for a 112×112 crop
         * (order: right eye, left eye, nose, right mouth, left mouth),
         * identical to OpenCV's `FaceRecognizerSF::alignCrop`.
         */
        val TEMPLATE: List<Point2> = listOf(
            Point2(38.2946f, 51.6963f),
            Point2(73.5318f, 51.5014f),
            Point2(56.0252f, 71.7366f),
            Point2(41.5493f, 92.3655f),
            Point2(70.7299f, 92.2041f),
        )
    }
}

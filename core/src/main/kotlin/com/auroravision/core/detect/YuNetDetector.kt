package com.auroravision.core.detect

import com.auroravision.core.inference.OnnxSession
import com.auroravision.core.math.ImageOps
import com.auroravision.core.model.Box
import com.auroravision.core.model.FaceDetection
import com.auroravision.core.model.FaceLandmarks
import com.auroravision.core.model.Point2
import com.auroravision.core.model.RgbImage
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * YuNet face detector (model: `face_detection_yunet_2023mar.onnx`, MIT).
 *
 * The model has a fixed 1×3×640×640 input. Images are letterboxed to 640×640
 * (never upscaled), which keeps the mobile cost roughly constant regardless of
 * camera resolution.
 *
 * For faces smaller than [refineBelowPx] a *second* pass runs on an upscaled
 * crop of the face: the same network then sees far more face pixels, which
 * measurably improves landmark precision (and therefore alignment quality)
 * at a fraction of the cost of a full-resolution pass.
 */
class YuNetDetector(
    private val session: OnnxSession,
    val inputSize: Int = 640,
    val scoreThreshold: Float = 0.6f,
    val nmsThreshold: Float = 0.3f,
    val maxFaces: Int = 20,
    val refineBelowPx: Float = 220f,
    val refineMaxUpscale: Float = 3f,
    val refineEnabled: Boolean = true,
) {

    fun detect(image: RgbImage, refine: Boolean = refineEnabled): List<FaceDetection> {
        val lb = ImageOps.letterbox(image, inputSize, allowUpscale = false)
        val raw = forward(lb)

        val mapped = ArrayList<FaceDetection>(raw.size)
        for (det in raw) {
            val box = Box(
                x = lb.toImageX(det.box.x),
                y = lb.toImageY(det.box.y),
                width = det.box.width / lb.scale,
                height = det.box.height / lb.scale,
            ).clampTo(image.width, image.height)
            if (box.width < 4f || box.height < 4f) continue
            val lm = FaceLandmarks(
                det.landmarks.points.map { Point2(lb.toImageX(it.x), lb.toImageY(it.y)) },
            )
            mapped.add(det.copy(box = box, landmarks = lm))
        }
        val suppressed = YuNetDecoder.nms(mapped, nmsThreshold, maxFaces)
            .sortedByDescending { it.box.area }
            .mapIndexed { i, d -> d.copy(index = i) }
        if (!refine) return suppressed
        return suppressed.map { det ->
            if (det.shortSidePx < refineBelowPx) refineOnCrop(image, det) else det
        }
    }

    /** Runs the network on a letterboxed canvas and decodes detections (canvas coords). */
    private fun forward(canvas: ImageOps.Letterbox): List<FaceDetection> {
        val blob = canvas.canvas.toPlanarFloat(bgr = true, scale = 1f)
        val outputs = session.run(blob, longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong()))
        val byName = HashMap<String, FloatArray>(outputs.size)
        for (o in outputs) byName[o.name] = o.data
        return YuNetDecoder.decode(byName, inputSize, scoreThreshold)
    }

    private fun refineOnCrop(image: RgbImage, det: FaceDetection): FaceDetection {
        val half = max(det.box.width, det.box.height) * 0.75f
        val x0 = max(0, (det.box.centerX - half).roundToInt())
        val y0 = max(0, (det.box.centerY - half).roundToInt())
        val x1 = min(image.width, (det.box.centerX + half).roundToInt())
        val y1 = min(image.height, (det.box.centerY + half).roundToInt())
        val cw = x1 - x0
        val ch = y1 - y0
        if (cw < 16 || ch < 16) return det

        val upscale = min(inputSize.toFloat() / max(cw, ch), refineMaxUpscale)
        if (upscale <= 1.05f) return det

        val up = ImageOps.resizeBilinear(
            ImageOps.crop(image, x0, y0, cw, ch),
            max(1, (cw * upscale).roundToInt()),
            max(1, (ch * upscale).roundToInt()),
        )
        val lb = ImageOps.letterbox(up, inputSize, allowUpscale = false)
        val cands = forward(lb)
        if (cands.isEmpty()) return det

        val ex = (det.box.centerX - x0) * upscale * lb.scale + lb.padX
        val ey = (det.box.centerY - y0) * upscale * lb.scale + lb.padY
        val best = cands.minByOrNull { c ->
            hypot(c.box.centerX - ex, c.box.centerY - ey)
        } ?: return det
        val dist = hypot(best.box.centerX - ex, best.box.centerY - ey)
        if (dist > 0.35f * max(best.box.width, best.box.height)) return det
        if (best.score < det.score * 0.8f) return det

        // canvas -> image: divide by (upscale * letterbox scale), then add the crop origin
        val total = upscale * lb.scale
        fun mapX(v: Float) = (v - lb.padX) / total + x0
        fun mapY(v: Float) = (v - lb.padY) / total + y0
        return det.copy(
            box = Box(
                x = mapX(best.box.x),
                y = mapY(best.box.y),
                width = best.box.width / total,
                height = best.box.height / total,
            ).clampTo(image.width, image.height),
            landmarks = FaceLandmarks(
                best.landmarks.points.map { Point2(mapX(it.x), mapY(it.y)) },
            ),
            score = best.score,
        )
    }
}

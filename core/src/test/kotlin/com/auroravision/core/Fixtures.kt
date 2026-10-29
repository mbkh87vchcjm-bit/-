package com.auroravision.core

import com.auroravision.core.detect.YuNetDetector
import com.auroravision.core.embed.SFaceEmbedder
import com.auroravision.core.inference.OnnxSession
import com.auroravision.core.model.RgbImage
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import java.io.File
import javax.imageio.ImageIO

/**
 * Test fixtures: real images and the real ONNX models.
 *
 * The tests run the *actual* models through the *actual* Kotlin code — the same
 * classes the Android app uses — because `:core` is a plain JVM module and the
 * ONNX Runtime API is identical on both platforms.
 */
object Fixtures {

    private val repoRoot: File by lazy {
        val prop = System.getProperty("aurora.repo.root")
            ?: error("aurora.repo.root system property is missing (see core/build.gradle.kts)")
        File(prop)
    }

    val dataDir: File by lazy { File(repoRoot, "tools/reference/data") }

    val modelsDir: File by lazy {
        File(
            System.getProperty("aurora.models.dir")
                ?: File(repoRoot, "app/src/main/assets/models").absolutePath,
        )
    }

    val calibrationDir: File by lazy {
        File(
            System.getProperty("aurora.calibration.dir")
                ?: File(repoRoot, "app/src/main/assets/calibration").absolutePath,
        )
    }

    fun scene(name: String): RgbImage = load(File(dataDir, "scenes/$name"))

    fun lfw(name: String): RgbImage = load(File(dataDir, "lfw/$name"))

    fun load(file: File): RgbImage {
        check(file.exists()) { "missing test fixture: $file" }
        val img = ImageIO.read(file)
        val w = img.width
        val h = img.height
        val bytes = ByteArray(w * h * 3)
        val row = IntArray(w)
        for (y in 0 until h) {
            img.getRGB(0, y, w, 1, row, 0, w)
            for (x in 0 until w) {
                val p = row[x]
                val i = (y * w + x) * 3
                bytes[i] = (p shr 16 and 0xFF).toByte()
                bytes[i + 1] = (p shr 8 and 0xFF).toByte()
                bytes[i + 2] = (p and 0xFF).toByte()
            }
        }
        return RgbImage(w, h, bytes)
    }

    fun golden(): JsonObject {
        val text = Fixtures::class.java.getResourceAsStream("/golden/pipeline_golden.json")
            ?.readBytes()?.decodeToString()
            ?: error("golden/pipeline_golden.json is missing from the test resources")
        return Json.parseToJsonElement(text) as JsonObject
    }

    fun detectorSession(): OnnxSession =
        OnnxSession.load(File(modelsDir, "face_detection_yunet_2023mar.onnx").readBytes(), threads = 2)

    fun embedderSession(): OnnxSession =
        OnnxSession.load(
            File(modelsDir, "face_recognition_sface_2021dec_int8.onnx").readBytes(),
            threads = 2,
        )

    fun detector(session: OnnxSession = detectorSession(), refine: Boolean = true) =
        YuNetDetector(session, refineEnabled = refine)

    fun embedder(session: OnnxSession = embedderSession()) = SFaceEmbedder(session)
}

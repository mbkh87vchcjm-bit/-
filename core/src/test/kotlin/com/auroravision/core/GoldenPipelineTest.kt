package com.auroravision.core

import com.auroravision.core.embed.SFaceEmbedder
import com.auroravision.core.math.ImageOps
import com.auroravision.core.model.FaceEmbedding
import com.auroravision.core.model.RgbImage
import com.auroravision.core.quality.QualityAnalyzer
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.JsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Runs the real ONNX models through the real Kotlin pipeline and compares the
 * output with the Python reference implementation (tools/reference).
 *
 * This is what makes "the AI engine works" a verifiable claim rather than a
 * screenshot: detection geometry, embeddings and quality metrics all have to
 * agree numerically with an independent implementation.
 */
class GoldenPipelineTest {

    private val golden: JsonObject by lazy { Fixtures.golden() }

    @Test
    fun `detections and embeddings match the python reference`() {
        val detectorSession = Fixtures.detectorSession()
        val embedderSession = Fixtures.embedderSession()
        try {
            val detector = Fixtures.detector(detectorSession, refine = true)
            val embedder = SFaceEmbedder(embedderSession)
            val scenes = golden["scenes"]!!.jsonArray
            var checkedFaces = 0

            for (scene in scenes) {
                val obj = scene.jsonObject
                val file = obj["file"]!!.jsonPrimitive.content
                val image = Fixtures.scene(file)
                assertEquals(
                    "width of $file", obj["width"]!!.jsonPrimitive.int, image.width,
                )
                assertEquals(
                    "height of $file", obj["height"]!!.jsonPrimitive.int, image.height,
                )

                val expected = obj["detections"]!!.jsonArray.map { it.jsonObject }
                val actual = detector.detect(image)
                assertEquals(
                    "face count for $file (reference=${expected.size}, actual=${actual.size})",
                    expected.size, actual.size,
                )

                expected.forEachIndexed { i, exp ->
                    val det = actual[i]
                    val box = exp["box"]!!.jsonArray.map { it.jsonPrimitive.float }
                    val expBox = com.auroravision.core.model.Box(box[0], box[1], box[2], box[3])
                    assertTrue(
                        "IoU too low for $file face $i: ${det.box.iou(expBox)}",
                        det.box.iou(expBox) > 0.97f,
                    )
                    val expScore = exp["score"]!!.jsonPrimitive.float
                    assertEquals("score for $file face $i", expScore, det.score, 5e-3f)

                    val expLandmarks = exp["landmarks"]!!.jsonArray.map { it.jsonArray }
                    expLandmarks.forEachIndexed { k, lm ->
                        val dx = abs(det.landmarks.points[k].x - lm[0].jsonPrimitive.float)
                        val dy = abs(det.landmarks.points[k].y - lm[1].jsonPrimitive.float)
                        assertTrue(
                            "landmark $k drift for $file face $i: ($dx, $dy)px",
                            dx < 4f && dy < 4f,
                        )
                    }

                    val expEmb = exp["embedding"]
                    if (expEmb == null || expEmb is kotlinx.serialization.json.JsonNull) {
                        // reference found a face but produced no embedding: acceptable only
                        // when the quality gate rejected it
                        assertTrue(
                            "no golden embedding for $file face $i, but the quality gate accepted it",
                            (exp["quality"]!!.jsonObject["status"]!!.jsonPrimitive.content) == "rejected",
                        )
                        return@forEachIndexed
                    }
                    val embedding = embedder.embed(image, det)
                    checkedFaces++
                    val goldenEmb = FaceEmbedding.fromStored(
                        expEmb.jsonArray.map { it.jsonPrimitive.float }.toFloatArray(),
                    )
                    val cos = embedding.cosineSimilarity(goldenEmb)
                    assertTrue(
                        "embedding drift for $file face $i: cosine=$cos",
                        cos > 0.999f,
                    )

                    // quality metrics must agree with the reference as well
                    val q = QualityAnalyzer.analyze(
                        embedder.align(image, det.landmarks), det.landmarks,
                        det.box.width, det.box.height, image.width, image.height,
                    )
                    val expQ = exp["quality"]!!.jsonObject
                    assertClose("sharpness $file/$i", expQ, "sharpness", q.sharpness, 0.02)
                    assertClose("brightness $file/$i", expQ, "brightness", q.brightness, 0.01)
                    assertClose("contrast $file/$i", expQ, "contrast", q.contrast, 0.02)
                    assertEquals(
                        "quality status for $file/$i",
                        expQ["status"]!!.jsonPrimitive.content,
                        q.status.name.lowercase(),
                    )
                    val codes = exp["issueCodes"]!!.jsonArray.map { it.jsonPrimitive.content }
                    assertEquals("issue codes for $file/$i", codes, q.issues.map { it.code })
                }
                println("$file → ${actual.size} face(s), ${actual.map { it.box }}")
            }
            assertTrue("the golden file should contain several faces", checkedFaces >= 5)
        } finally {
            detectorSession.close()
            embedderSession.close()
        }
    }

    @Test
    fun `bilinear resize matches the reference pixel samples`() {
        val resize = golden["resize"]!!.jsonObject
        val src = Fixtures.scene(resize["src"]!!.jsonPrimitive.content)
        val w = resize["dstWidth"]!!.jsonPrimitive.int
        val h = resize["dstHeight"]!!.jsonPrimitive.int
        val out = ImageOps.resizeBilinear(src, w, h)
        for (s in resize["samples"]!!.jsonArray) {
            val a = s.jsonArray.map { it.jsonPrimitive.int }
            val idx = (a[1] * w + a[0]) * 3
            assertEquals("R at (${a[0]},${a[1]})", a[2], out.pixels[idx].toInt() and 0xFF)
            assertEquals("G at (${a[0]},${a[1]})", a[3], out.pixels[idx + 1].toInt() and 0xFF)
            assertEquals("B at (${a[0]},${a[1]})", a[4], out.pixels[idx + 2].toInt() and 0xFF)
        }
    }

    @Test
    fun `letterbox geometry matches the reference`() {
        val lb = golden["letterbox"]!!.jsonObject
        val src = Fixtures.scene(lb["src"]!!.jsonPrimitive.content)
        val size = lb["size"]!!.jsonPrimitive.int
        val actual = ImageOps.letterbox(src, size)
        assertEquals(lb["scale"]!!.jsonPrimitive.float, actual.scale, 1e-6f)
        assertEquals(lb["padX"]!!.jsonPrimitive.int, actual.padX)
        assertEquals(lb["padY"]!!.jsonPrimitive.int, actual.padY)
    }

    @Test
    fun `pairwise similarities match the reference`() {
        val pairwise = golden["pairwise"]!!.jsonObject
        val embeddings = golden["embeddings"]!!.jsonObject
        fun emb(name: String): FaceEmbedding = FaceEmbedding.fromStored(
            embeddings["scene_$name"]!!.jsonArray.map { it.jsonPrimitive.float }.toFloatArray(),
        )
        val expected = mapOf(
            "obama_obama2" to pairwise["obama_obama2"]!!.jsonPrimitive.float,
            "obama_biden" to pairwise["obama_biden"]!!.jsonPrimitive.float,
            "obama2_biden" to pairwise["obama2_biden"]!!.jsonPrimitive.float,
        )
        val actual = mapOf(
            "obama_obama2" to emb("obama").cosineSimilarity(emb("obama2")),
            "obama_biden" to emb("obama").cosineSimilarity(emb("biden")),
            "obama2_biden" to emb("obama2").cosineSimilarity(emb("biden")),
        )
        for ((k, v) in expected) {
            assertEquals(k, v, actual[k]!!, 1e-3f)
        }
        // and the discrimination must be real, not incidental
        assertTrue(
            "genuine pair must score above impostor pair (${actual["obama_obama2"]} vs ${actual["obama_biden"]})",
            actual["obama_obama2"]!! > actual["obama_biden"]!! + 0.1f,
        )
    }

    private fun assertClose(label: String, obj: JsonObject, key: String, actual: Float, rel: Double) {
        val exp = obj[key]!!.jsonPrimitive.float
        val tol = (abs(exp) * rel).toFloat().coerceAtLeast(1e-3f)
        assertEquals(label, exp, actual, tol)
    }

    @Test
    fun `engine is deterministic across repeated runs`() {
        val session = Fixtures.detectorSession()
        val embSession = Fixtures.embedderSession()
        try {
            val detector = Fixtures.detector(session)
            val embedder = SFaceEmbedder(embSession)
            val image: RgbImage = Fixtures.scene("obama.jpg")
            val a = detector.detect(image).first()
            val b = detector.detect(image).first()
            assertEquals(a.box.x, b.box.x, 0f)
            assertEquals(a.score, b.score, 0f)
            val ea = embedder.embed(image, a)
            val eb = embedder.embed(image, b)
            assertEquals(1f, ea.cosineSimilarity(eb), 1e-6f)
        } finally {
            session.close()
            embSession.close()
        }
    }
}

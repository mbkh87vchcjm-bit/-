package com.auroravision.core.quality

import com.auroravision.core.Fixtures
import com.auroravision.core.model.FaceLandmarks
import com.auroravision.core.model.Point2
import com.auroravision.core.model.RgbImage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class QualityAnalyzerTest {

    private fun solid(v: Int, w: Int = 112, h: Int = 112) =
        RgbImage(w, h, ByteArray(w * h * 3) { v.toByte() })

    /** 112x112 image with a black/white stripe pattern of a given period. */
    private fun stripes(period: Int, w: Int = 112, h: Int = 112): RgbImage {
        val px = ByteArray(w * h * 3)
        for (y in 0 until h) for (x in 0 until w) {
            val v = if ((x / period + y / period) % 2 == 0) 240 else 15
            val i = (y * w + x) * 3
            px[i] = v.toByte(); px[i + 1] = v.toByte(); px[i + 2] = v.toByte()
        }
        return RgbImage(w, h, px)
    }

    private val frontal = FaceLandmarks(
        listOf(
            Point2(40f, 50f), Point2(72f, 50f), Point2(56f, 70f),
            Point2(44f, 90f), Point2(68f, 90f),
        ),
    )

    @Test
    fun `laplacian variance is zero for a flat image and high for stripes`() {
        val flat = QualityAnalyzer.laplacianVariance(QualityAnalyzer.grayPlane(solid(128)), 112, 112)
        assertEquals(0f, flat, 1e-6f)
        val striped = QualityAnalyzer.laplacianVariance(QualityAnalyzer.grayPlane(stripes(4)), 112, 112)
        assertTrue("stripes should produce a large Laplacian variance, got $striped", striped > 1000f)
    }

    @Test
    fun `yaw proxy is zero for a symmetric face and signed for a turned one`() {
        assertEquals(0f, QualityAnalyzer.yawProxy(frontal), 1e-6f)
        // nose pushed toward the right eye -> face turned
        val turned = FaceLandmarks(
            listOf(
                Point2(40f, 50f), Point2(72f, 50f), Point2(64f, 70f),
                Point2(44f, 90f), Point2(68f, 90f),
            ),
        )
        assertTrue(QualityAnalyzer.yawProxy(turned) < -0.1f)
    }

    @Test
    fun `roll is measured from the eye line`() {
        val rolled = FaceLandmarks(
            listOf(
                Point2(40f, 40f), Point2(72f, 60f), Point2(56f, 60f),
                Point2(44f, 80f), Point2(68f, 95f),
            ),
        )
        val roll = QualityAnalyzer.rollDegrees(rolled)
        assertTrue("expected a positive roll, got $roll", roll > 20f)
        assertTrue(QualityAnalyzer.analyze(
            solid(140), rolled, 80f, 80f, 400, 400,
        ).issues.any { it.code == "EXTREME_ROLL" })
    }

    @Test
    fun `rejects a face that is too small for a reliable comparison`() {
        val report = QualityAnalyzer.analyze(
            stripes(3), frontal, faceWidth = 30f, faceHeight = 30f,
            imageWidth = 1000, imageHeight = 1000,
        )
        assertEquals(QualityStatus.REJECTED, report.status)
        assertTrue(report.issues.any { it.code == "FACE_TOO_SMALL" })
        assertTrue(!report.comparable)
    }

    @Test
    fun `rejects a too dark image and warns about a bright one`() {
        val dark = QualityAnalyzer.analyze(solid(10), frontal, 200f, 200f, 400, 400)
        assertEquals(QualityStatus.REJECTED, dark.status)
        assertTrue(dark.issues.any { it.code == "TOO_DARK" })

        // A saturated *flat* image is also blurry, so only the exposure issue is
        // asserted here; the bright stripes image below isolates the exposure gate.
        val bright = QualityAnalyzer.analyze(solid(250), frontal, 200f, 200f, 400, 400)
        assertTrue(bright.issues.any { it.code == "OVEREXPOSED" })
        val brightDetail = QualityAnalyzer.analyze(stripes(6), frontal, 200f, 200f, 400, 400)
        assertEquals(QualityStatus.OK, brightDetail.status)
    }

    @Test
    fun `a good face passes the gate with a high score and no rejecting issue`() {
        val report = QualityAnalyzer.analyze(stripes(6), frontal, 160f, 160f, 400, 400)
        assertTrue(report.issues.none { it.severity == IssueSeverity.REJECTED })
        assertTrue("score should be decent, was ${report.score}", report.score > 0.5f)
    }

    @Test
    fun `metrics agree with the python reference on a real face`() {
        val golden = Fixtures.golden()["quality"]?.let { it as kotlinx.serialization.json.JsonObject }
            ?: return
        val scene = Fixtures.golden()["scenes"]!!.let { it as kotlinx.serialization.json.JsonArray }
            .map { it as kotlinx.serialization.json.JsonObject }
            .first { (it["file"]!! as kotlinx.serialization.json.JsonPrimitive).content == "obama.jpg" }
        val det = (scene["detections"]!! as kotlinx.serialization.json.JsonArray)[0]
            as kotlinx.serialization.json.JsonObject
        val q = golden["sharp"]!! as kotlinx.serialization.json.JsonObject
        val expSharpness = (q["sharpness"]!! as kotlinx.serialization.json.JsonPrimitive).content.toFloat()
        val expBrightness = (q["brightness"]!! as kotlinx.serialization.json.JsonPrimitive).content.toFloat()
        val expContrast = (q["contrast"]!! as kotlinx.serialization.json.JsonPrimitive).content.toFloat()

        val session = Fixtures.detectorSession()
        val embSession = Fixtures.embedderSession()
        try {
            val embedder = com.auroravision.core.embed.SFaceEmbedder(embSession)
            val image = Fixtures.scene("obama.jpg")
            val detection = Fixtures.detector(session).detect(image).first()
            val aligned = embedder.align(image, detection.landmarks)
            val report = QualityAnalyzer.analyze(
                aligned, detection.landmarks, detection.box.width, detection.box.height,
                image.width, image.height,
            )
            // Laplacian variance is highly sensitive to the exact resampled pixels:
            // the aligned crop comes from an affine warp whose coefficients are
            // computed in float32 here and float64 in the reference.
            assertEquals(expSharpness, report.sharpness, expSharpness * 0.10f + 1.0f)
            assertEquals(expBrightness, report.brightness, 1.5f)
            assertEquals(expContrast, report.contrast, 1.5f)
            // sanity: the detection we used must be the one the golden was built from
            val expScore = (det["score"]!! as kotlinx.serialization.json.JsonPrimitive).content.toFloat()
            assertEquals(expScore, detection.score, 0.01f)
        } finally {
            session.close()
            embSession.close()
        }
    }
}

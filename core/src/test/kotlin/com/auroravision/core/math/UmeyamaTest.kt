package com.auroravision.core.math

import com.auroravision.core.Fixtures
import com.auroravision.core.model.Point2
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.float
import org.junit.Assert.assertEquals
import org.junit.Test

class UmeyamaTest {

    private fun p(x: Float, y: Float) = Point2(x, y)

    @Test
    fun `recovers a known similarity transform exactly`() {
        // dst = 1.7 * R(23°) * src + (40, -12)
        val angle = Math.toRadians(23.0)
        val s = 1.7
        val c = kotlin.math.cos(angle)
        val sn = kotlin.math.sin(angle)
        val src = listOf(p(10f, 10f), p(50f, 12f), p(30f, 40f), p(14f, 70f), p(48f, 68f))
        val dst = src.map {
            p(
                (s * (c * it.x - sn * it.y) + 40).toFloat(),
                (s * (sn * it.x + c * it.y) - 12).toFloat(),
            )
        }
        val m = Umeyama.similarityTransform(src, dst)
        src.forEachIndexed { i, sp ->
            val (x, y) = m.apply(sp.x, sp.y)
            assertEquals(dst[i].x, x, 1e-3f)
            assertEquals(dst[i].y, y, 1e-3f)
        }
    }

    @Test
    fun `matches the python reference for the SFace template`() {
        val golden = Fixtures.golden()["umeyama"]!!.jsonObject
        val src = golden["src"]!!.jsonArray.map {
            val a = it.jsonArray
            p(a[0].jsonPrimitive.float, a[1].jsonPrimitive.float)
        }
        val expected = golden["matrix"]!!.jsonArray.map { it.jsonPrimitive.float }
        val m = Umeyama.similarityTransform(
            src,
            listOf(
                p(38.2946f, 51.6963f), p(73.5318f, 51.5014f), p(56.0252f, 71.7366f),
                p(41.5493f, 92.3655f), p(70.7299f, 92.2041f),
            ),
        )
        val actual = floatArrayOf(m.a, m.b, m.c, m.d, m.e, m.f)
        // the reference computes in float64; allow a tight tolerance in float32
        for (i in 0 until 6) {
            assertEquals("matrix[$i]", expected[i], actual[i], 2e-3f * maxOf(1f, kotlin.math.abs(expected[i])))
        }
    }

    @Test
    fun `maps the template onto itself as the identity`() {
        val tmpl = listOf(
            p(38.2946f, 51.6963f), p(73.5318f, 51.5014f), p(56.0252f, 71.7366f),
            p(41.5493f, 92.3655f), p(70.7299f, 92.2041f),
        )
        val m = Umeyama.similarityTransform(tmpl, tmpl)
        assertEquals(1f, m.a, 1e-4f)
        assertEquals(0f, m.b, 1e-4f)
        assertEquals(1f, m.e, 1e-4f)
        assertEquals(0f, m.c, 1e-3f)
        assertEquals(0f, m.f, 1e-3f)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `rejects degenerate point sets`() {
        Umeyama.similarityTransform(
            listOf(p(5f, 5f), p(5f, 5f), p(5f, 5f), p(5f, 5f), p(5f, 5f)),
            listOf(p(1f, 1f), p(2f, 2f), p(3f, 3f), p(4f, 4f), p(5f, 5f)),
        )
    }
}

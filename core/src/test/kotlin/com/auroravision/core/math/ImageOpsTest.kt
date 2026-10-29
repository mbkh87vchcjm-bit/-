package com.auroravision.core.math

import com.auroravision.core.model.RgbImage
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

class ImageOpsTest {

    private fun image(width: Int, height: Int, seed: Int = 7): RgbImage {
        val bytes = ByteArray(width * height * 3)
        var v = seed
        for (i in bytes.indices) {
            v = (v * 1103515245 + 12345) and 0x7FFFFFFF
            bytes[i] = (v shr 16).toByte()
        }
        return RgbImage(width, height, bytes)
    }

    @Test
    fun `identity resize returns an identical copy`() {
        val src = image(9, 5)
        val out = ImageOps.resizeBilinear(src, 9, 5)
        assertTrue(src.pixels.contentEquals(out.pixels))
    }

    @Test
    fun `2x2 average of a 4x4 checkerboard is exact`() {
        // 4x4 image, 2x2 blocks of 0 / 255 -> 2x2 result must be exactly 128
        val w = 4
        val h = 4
        val px = ByteArray(w * h * 3)
        for (y in 0 until h) {
            for (x in 0 until w) {
                val quad = ((x / 2) + (y / 2)) % 2
                val v = if (quad == 0) 255 else 0
                val i = (y * w + x) * 3
                px[i] = v.toByte(); px[i + 1] = v.toByte(); px[i + 2] = v.toByte()
            }
        }
        val out = ImageOps.resizeBilinear(RgbImage(w, h, px), 2, 2)
        for (b in out.pixels) {
            assertEquals(128, b.toInt() and 0xFF)
        }
    }

    @Test
    fun `letterbox never upscales by default and centres the image`() {
        val src = image(1000, 500)
        val lb = ImageOps.letterbox(src, 640)
        assertEquals(0.64f, lb.scale, 1e-6f)
        assertEquals(640, lb.canvas.width)
        assertEquals(320, (500 * lb.scale).toInt())
        assertEquals(0, lb.padX)
        assertEquals((640 - 320) / 2, lb.padY)
        // padded rows are black
        assertEquals(0, lb.canvas.pixels[0].toInt())
        // coordinate round trip
        val x = 123.4f
        assertEquals(x, lb.toImageX(lb.toCanvasX(x)), 1e-3f)
    }

    @Test
    fun `letterbox of a tiny image keeps its native scale`() {
        val src = image(80, 60)
        val lb = ImageOps.letterbox(src, 640)
        assertEquals(1f, lb.scale, 1e-6f)
        assertEquals((640 - 80) / 2, lb.padX)
    }

    @Test
    fun `warpAffine with the identity matrix is lossless`() {
        val src = image(13, 7)
        val out = ImageOps.warpAffine(src, Affine2x3.identity(), 13, 7)
        assertTrue(src.pixels.contentEquals(out.pixels))
    }

    @Test
    fun `warpAffine translation moves content and fills the border`() {
        val px = ByteArray(4 * 4 * 3) { 200.toByte() }
        val src = RgbImage(4, 4, px)
        val out = ImageOps.warpAffine(src, Affine2x3(1f, 0f, 2f, 0f, 1f, 0f), 4, 4, border = 0)
        // first two columns are outside the source
        assertEquals(0, out.pixels[0].toInt())
        assertEquals(0, out.pixels[(0 * 4 + 1) * 3].toInt())
        assertEquals(200, out.pixels[(0 * 4 + 2) * 3].toInt() and 0xFF)
    }

    @Test
    fun `rotate by 90 degrees is invertible within rounding`() {
        val src = image(21, 21)
        val rotated = ImageOps.rotate(src, 90f)
        val back = ImageOps.rotate(rotated, -90f)
        var maxDiff = 0
        for (i in src.pixels.indices) {
            val d = abs((src.pixels[i].toInt() and 0xFF) - (back.pixels[i].toInt() and 0xFF))
            if (d > maxDiff) maxDiff = d
        }
        // two bilinear resamplings: allow small interpolation error on the edges
        assertTrue("rotate round-trip drift was $maxDiff", maxDiff <= 8)
    }

    @Test
    fun `inverse of an affine matrix round-trips`() {
        val m = Affine2x3(1.3f, -0.2f, 12f, 0.15f, 0.9f, -7f)
        val inv = m.invert()
        val (x, y) = m.apply(37f, 91f)
        val (x2, y2) = inv.apply(x, y)
        assertEquals(37f, x2, 1e-3f)
        assertEquals(91f, y2, 1e-3f)
    }
}

package com.auroravision.core.model

/**
 * An 8-bit RGB image, tightly packed (3 bytes per pixel, row-major, RGB order).
 *
 * The whole vision core works on this type so that it stays free of Android
 * `Bitmap` and of any file-system assumptions. The app converts a decoded
 * `Bitmap` into an [RgbImage] once, and the core never touches Android again.
 */
class RgbImage(
    val width: Int,
    val height: Int,
    val pixels: ByteArray,
) {
    init {
        require(width > 0 && height > 0) { "image must have a positive size, got ${width}x$height" }
        require(pixels.size == width * height * 3) {
            "expected ${width * height * 3} bytes for ${width}x$height RGB, got ${pixels.size}"
        }
    }

    val pixelCount: Int get() = width * height

    fun red(index: Int): Int = pixels[index * 3].toInt() and 0xFF

    fun green(index: Int): Int = pixels[index * 3 + 1].toInt() and 0xFF

    fun blue(index: Int): Int = pixels[index * 3 + 2].toInt() and 0xFF

    /** BT.601 luma of pixel (x, y) in [0, 255]. */
    fun luma(x: Int, y: Int): Double {
        val i = (y * width + x) * 3
        val r = pixels[i].toInt() and 0xFF
        val g = pixels[i + 1].toInt() and 0xFF
        val b = pixels[i + 2].toInt() and 0xFF
        return 0.299 * r + 0.587 * g + 0.114 * b
    }

    fun copy(): RgbImage = RgbImage(width, height, pixels.copyOf())

    /**
     * Packs the image into a planar (CHW) float buffer, optionally reversing the
     * channel order to BGR — the YuNet detector expects BGR, SFace expects RGB.
     *
     * @param scale multiplier applied to every value (1.0 keeps 0..255).
     */
    fun toPlanarFloat(bgr: Boolean, scale: Float = 1f): FloatArray {
        val n = pixelCount
        val out = FloatArray(n * 3)
        for (i in 0 until n) {
            val s = i * 3
            val r = (pixels[s].toInt() and 0xFF) * scale
            val g = (pixels[s + 1].toInt() and 0xFF) * scale
            val b = (pixels[s + 2].toInt() and 0xFF) * scale
            if (bgr) {
                out[i] = b
                out[n + i] = g
                out[n * 2 + i] = r
            } else {
                out[i] = r
                out[n + i] = g
                out[n * 2 + i] = b
            }
        }
        return out
    }

    override fun toString(): String = "RgbImage(${width}x$height)"
}

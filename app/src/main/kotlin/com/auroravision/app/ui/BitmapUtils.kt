package com.auroravision.app.ui

import android.graphics.Bitmap
import com.auroravision.core.model.RgbImage

/** Converts a core [RgbImage] into an ARGB bitmap for display. */
fun RgbImage.toBitmap(): Bitmap {
    val pixels = IntArray(width * height)
    for (i in pixels.indices) {
        val s = i * 3
        val r = pixels.let { _ -> this.pixels[s].toInt() and 0xFF }
        val g = this.pixels[s + 1].toInt() and 0xFF
        val b = this.pixels[s + 2].toInt() and 0xFF
        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
    }
    return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
}

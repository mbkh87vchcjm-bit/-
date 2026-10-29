package com.auroravision.app.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.auroravision.core.model.RgbImage
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest
import java.util.UUID
import kotlin.math.max

/** A decoded image plus the metadata the pipeline and the UI need. */
class DecodedImage(
    val bitmap: Bitmap,
    val rgb: RgbImage,
    val bytes: ByteArray,
    val sha256: String,
    val width: Int,
    val height: Int,
)

/**
 * Owns the image files the app keeps.
 *
 * Policy: the *only* images stored are (a) images the user explicitly registers
 * in the reference database and (b) the query image of a search the user ran,
 * which is deleted when the history entry is deleted. Nothing is copied to
 * external storage, nothing is uploaded.
 */
class ImageStore(private val context: Context) {

    private val imagesDir = File(context.filesDir, "face_images").apply { mkdirs() }
    private val thumbsDir = File(context.filesDir, "face_thumbs").apply { mkdirs() }
    private val queriesDir = File(context.filesDir, "query_images").apply { mkdirs() }
    val capturesDir: File get() = File(context.cacheDir, "captures").apply { mkdirs() }

    /** Maximum stored edge; larger photos are downscaled once, at import time. */
    val maxStoredEdge: Int = 1600

    fun decode(uri: Uri): DecodedImage {
        val cr = context.contentResolver
        val bytes = cr.openInputStream(uri)?.use { it.readBytes() }
            ?: error("cannot open the selected image")
        return decodeBytes(bytes)
    }

    fun decode(file: File): DecodedImage = decodeBytes(file.readBytes())

    fun decodeBytes(bytes: ByteArray): DecodedImage {
        require(bytes.isNotEmpty()) { "the selected file is empty" }
        val sha = sha256(bytes)
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        require(bounds.outWidth > 0 && bounds.outHeight > 0) {
            "the selected file is not a decodable image"
        }
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxStoredEdge) sample *= 2
        val opts = BitmapFactory.Options().apply { inSampleSize = sample }
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
            ?: error("the selected image could not be decoded")
        val orientation = runCatching {
            ExifInterface(bytes.inputStream()).getAttributeInt(
                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL,
            )
        }.getOrDefault(ExifInterface.ORIENTATION_NORMAL)
        val rotated = applyOrientation(decoded, orientation)
        val scaled = downscale(rotated, maxStoredEdge)
        return DecodedImage(
            bitmap = scaled,
            rgb = toRgbImage(scaled),
            bytes = bytes,
            sha256 = sha,
            width = scaled.width,
            height = scaled.height,
        )
    }

    /** Persists a registered image; returns its path. */
    fun saveRegisteredImage(decoded: DecodedImage): String {
        val file = File(imagesDir, "${UUID.randomUUID()}.jpg")
        FileOutputStream(file).use { out ->
            decoded.bitmap.compress(Bitmap.CompressFormat.JPEG, 92, out)
        }
        return file.absolutePath
    }

    fun saveThumbnail(decoded: DecodedImage): String? {
        val edge = 320
        val scale = edge.toFloat() / max(decoded.bitmap.width, decoded.bitmap.height)
        val thumb = if (scale < 1f) {
            Bitmap.createScaledBitmap(
                decoded.bitmap,
                max(1, (decoded.bitmap.width * scale).toInt()),
                max(1, (decoded.bitmap.height * scale).toInt()),
                true,
            )
        } else {
            decoded.bitmap
        }
        val file = File(thumbsDir, "${UUID.randomUUID()}.jpg")
        return runCatching {
            FileOutputStream(file).use { out ->
                thumb.compress(Bitmap.CompressFormat.JPEG, 85, out)
            }
            file.absolutePath
        }.getOrNull()
    }

    fun saveQueryImage(decoded: DecodedImage): String {
        val file = File(queriesDir, "${UUID.randomUUID()}.jpg")
        FileOutputStream(file).use { out ->
            decoded.bitmap.compress(Bitmap.CompressFormat.JPEG, 92, out)
        }
        return file.absolutePath
    }

    fun deleteFiles(paths: Iterable<String?>) {
        paths.filterNotNull().forEach { path ->
            runCatching { File(path).takeIf { it.exists() }?.delete() }
        }
    }

    fun deleteThumbnailsOf(imagePath: String) {
        // thumbnails are stored separately; a stale file is harmless but clean up anyway
        thumbsDir.listFiles()?.forEach { if (it.name.startsWith(imagePath.hashCode().toString())) it.delete() }
    }

    fun readBitmap(path: String): Bitmap? =
        runCatching { BitmapFactory.decodeFile(path) }.getOrNull()

    private fun applyOrientation(bitmap: Bitmap, orientation: Int): Bitmap {
        val matrix = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> matrix.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> matrix.postScale(1f, -1f)
            else -> return bitmap
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun downscale(bitmap: Bitmap, maxEdge: Int): Bitmap {
        val longest = max(bitmap.width, bitmap.height)
        if (longest <= maxEdge) return bitmap
        val scale = maxEdge.toFloat() / longest
        return Bitmap.createScaledBitmap(
            bitmap,
            max(1, (bitmap.width * scale).toInt()),
            max(1, (bitmap.height * scale).toInt()),
            true,
        )
    }

    companion object {
        fun toRgbImage(bitmap: Bitmap): RgbImage {
            val w = bitmap.width
            val h = bitmap.height
            val pixels = IntArray(w * h)
            bitmap.getPixels(pixels, 0, w, 0, 0, w, h)
            val bytes = ByteArray(w * h * 3)
            for (i in pixels.indices) {
                val p = pixels[i]
                val o = i * 3
                bytes[o] = (p shr 16 and 0xFF).toByte()
                bytes[o + 1] = (p shr 8 and 0xFF).toByte()
                bytes[o + 2] = (p and 0xFF).toByte()
            }
            return RgbImage(w, h, bytes)
        }

        fun sha256(bytes: ByteArray): String {
            val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
            return digest.joinToString("") { "%02x".format(it) }
        }
    }
}

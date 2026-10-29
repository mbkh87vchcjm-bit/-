package com.auroravision.app.ui.components

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import com.auroravision.app.ui.theme.AuroraColors
import com.auroravision.core.model.Box
import com.auroravision.core.model.Point2
import com.auroravision.core.quality.QualityStatus

/**
 * Draws the detected face geometry on top of the analysed image: corner
 * brackets around the box and a marker on each of the five landmarks.
 *
 * The coordinates come straight from the detector, so what is drawn is exactly
 * what the engine used for alignment and embedding.
 */
@Composable
fun FaceOverlay(
    imageWidth: Int,
    imageHeight: Int,
    faces: List<DetectedFaceOverlay>,
    modifier: Modifier = Modifier,
    showLandmarks: Boolean = true,
    selectedIndex: Int? = null,
) {
    val appear by animateFloatAsState(targetValue = 1f, label = "overlay")
    Canvas(modifier.fillMaxSize()) {
        if (imageWidth <= 0 || imageHeight <= 0) return@Canvas
        val scale = minOf(size.width / imageWidth, size.height / imageHeight)
        val dx = (size.width - imageWidth * scale) / 2f
        val dy = (size.height - imageHeight * scale) / 2f
        faces.forEach { face ->
            val color = when (face.status) {
                QualityStatus.OK -> AuroraColors.Cyan
                QualityStatus.WARNING -> AuroraColors.Amber
                QualityStatus.REJECTED -> AuroraColors.Danger
            }
            val alpha = if (selectedIndex == null || selectedIndex == face.index) 1f else 0.45f
            val left = dx + face.box.x * scale
            val top = dy + face.box.y * scale
            val right = left + face.box.width * scale
            val bottom = top + face.box.height * scale
            drawBracket(left, top, right, bottom, color.copy(alpha = alpha * appear))
            if (showLandmarks) {
                face.landmarks.forEach { p ->
                    val c = Offset(dx + p.x * scale, dy + p.y * scale)
                    drawCircle(color = color.copy(alpha = alpha), radius = 2.5.dp.toPx(), center = c)
                    drawCircle(
                        color = color.copy(alpha = alpha * 0.5f),
                        radius = 5.5.dp.toPx(),
                        center = c,
                        style = Stroke(width = 1f),
                    )
                }
            }
        }
    }
}

class DetectedFaceOverlay(
    val index: Int,
    val box: Box,
    val landmarks: List<Point2>,
    val status: QualityStatus,
)

/** Four corner brackets + a thin full frame. */
private fun DrawScope.drawBracket(left: Float, top: Float, right: Float, bottom: Float, color: Color) {
    val len = minOf(22.dp.toPx(), (right - left) * 0.28f, (bottom - top) * 0.28f)
    val w = 2.dp.toPx()
    drawRoundRect(
        color = color.copy(alpha = 0.22f),
        topLeft = Offset(left, top),
        size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
        cornerRadius = androidx.compose.ui.geometry.CornerRadius(6.dp.toPx()),
        style = Stroke(width = 1f),
    )
    // corners
    drawLine(color, Offset(left, top), Offset(left + len, top), w)
    drawLine(color, Offset(left, top), Offset(left, top + len), w)
    drawLine(color, Offset(right, top), Offset(right - len, top), w)
    drawLine(color, Offset(right, top), Offset(right, top + len), w)
    drawLine(color, Offset(left, bottom), Offset(left + len, bottom), w)
    drawLine(color, Offset(left, bottom), Offset(left, bottom - len), w)
    drawLine(color, Offset(right, bottom), Offset(right - len, bottom), w)
    drawLine(color, Offset(right, bottom), Offset(right, bottom - len), w)
}

/** Wraps an image preview with the overlay; the image slot fills the box. */
@Composable
fun OverlayHost(
    imageWidth: Int,
    imageHeight: Int,
    faces: List<DetectedFaceOverlay>,
    modifier: Modifier = Modifier,
    selectedIndex: Int? = null,
    image: @Composable () -> Unit,
) {
    Box(modifier) {
        image()
        FaceOverlay(
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            faces = faces,
            selectedIndex = selectedIndex,
        )
    }
}

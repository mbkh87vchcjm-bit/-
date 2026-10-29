package com.auroravision.app.ui.components

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.auroravision.app.ui.theme.AuroraColors

/** Dark gradient background with a faint technical grid (drawn, not an image). */
@Composable
fun AuroraBackground(content: @Composable () -> Unit) {
    Box(Modifier.fillMaxSize().background(AuroraColors.backgroundGradient)) {
        Canvas(Modifier.fillMaxSize()) {
            val step = 56.dp.toPx()
            val line = Color(0xFF0E2A3D).copy(alpha = 0.35f)
            var x = 0f
            while (x < size.width) {
                drawLine(line, Offset(x, 0f), Offset(x, size.height), strokeWidth = 1f)
                x += step
            }
            var y = 0f
            while (y < size.height) {
                drawLine(line, Offset(0f, y), Offset(size.width, y), strokeWidth = 1f)
                y += step
            }
            // soft cyan glow at the top
            drawRect(
                brush = Brush.radialGradient(
                    colors = listOf(AuroraColors.Cyan.copy(alpha = 0.10f), Color.Transparent),
                    center = Offset(size.width * 0.5f, size.height * 0.05f),
                    radius = size.width * 0.9f,
                ),
            )
        }
        content()
    }
}

/** A panel with a hairline glowing border, used for cards and sections. */
@Composable
fun AuroraPanel(
    modifier: Modifier = Modifier,
    accent: Color = AuroraColors.Outline,
    content: @Composable () -> Unit,
) {
    Box(
        modifier = modifier
            .background(
                brush = Brush.verticalGradient(
                    listOf(AuroraColors.Panel.copy(alpha = 0.95f), AuroraColors.Deep.copy(alpha = 0.95f)),
                ),
                shape = RoundedCornerShape(18.dp),
            )
    ) {
        Canvas(Modifier.fillMaxSize()) {
            drawRoundRect(
                color = accent.copy(alpha = 0.45f),
                size = size,
                cornerRadius = androidx.compose.ui.geometry.CornerRadius(18.dp.toPx()),
                style = androidx.compose.ui.graphics.drawscope.Stroke(width = 1.dp.toPx()),
            )
        }
        Column(Modifier.padding(16.dp)) { content() }
    }
}

/** Small monospace-ish label in the style of the film HUDs. */
@Composable
fun HudLabel(text: String, color: Color = AuroraColors.TextSecondary, modifier: Modifier = Modifier) {
    Text(
        text = text,
        style = MaterialTheme.typography.labelSmall,
        color = color,
        modifier = modifier,
    )
}

@Composable
fun PrimaryAction(
    text: String,
    icon: ImageVector? = null,
    enabled: Boolean = true,
    modifier: Modifier = Modifier,
    onClick: () -> Unit,
) {
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.fillMaxWidth().height(54.dp),
        shape = RoundedCornerShape(14.dp),
        colors = ButtonDefaults.buttonColors(
            containerColor = AuroraColors.Cyan,
            contentColor = Color(0xFF00222E),
            disabledContainerColor = AuroraColors.PanelLight,
            disabledContentColor = AuroraColors.TextSecondary,
        ),
    ) {
        if (icon != null) {
            Icon(icon, contentDescription = null, modifier = Modifier.size(20.dp))
            Spacer(Modifier.size(10.dp))
        }
        Text(text, style = MaterialTheme.typography.titleMedium)
    }
}

@Composable
fun SecondaryAction(
    text: String,
    icon: ImageVector? = null,
    enabled: Boolean = true,
    modifier: Modifier = Modifier,
    accent: Color = AuroraColors.Violet,
    onClick: () -> Unit,
) {
    OutlinedButton(
        onClick = onClick,
        enabled = enabled,
        modifier = modifier.fillMaxWidth().height(50.dp),
        shape = RoundedCornerShape(14.dp),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = AuroraColors.TextPrimary),
    ) {
        if (icon != null) {
            Icon(icon, contentDescription = null, modifier = Modifier.size(18.dp), tint = accent)
            Spacer(Modifier.size(8.dp))
        }
        Text(text, style = MaterialTheme.typography.bodyMedium)
    }
}

/** Large number + caption, used for the statistics row on the home screen. */
@Composable
fun StatTile(value: String, label: String, modifier: Modifier = Modifier, accent: Color = AuroraColors.Cyan) {
    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(value, style = MaterialTheme.typography.headlineSmall, color = accent)
        Spacer(Modifier.height(2.dp))
        Text(
            label,
            style = MaterialTheme.typography.bodySmall,
            color = AuroraColors.TextSecondary,
            textAlign = TextAlign.Center,
        )
    }
}

/**
 * Animated scan line. Only used while a real phase is running: the caller
 * decides when it is visible, so it never implies progress that is not happening.
 */
@Composable
fun ScanningOverlay(modifier: Modifier = Modifier, color: Color = AuroraColors.Cyan) {
    val transition = rememberInfiniteTransition(label = "scan")
    val progress by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1800), RepeatMode.Restart),
        label = "scanline",
    )
    Canvas(modifier) {
        val y = size.height * progress
        val bandHeight = 60.dp.toPx()
        val top = (y - bandHeight / 2f).coerceIn(0f, (size.height - bandHeight).coerceAtLeast(0f))
        drawRect(
            color = color.copy(alpha = 0.07f),
            topLeft = Offset(0f, top),
            size = Size(size.width, bandHeight),
        )
        drawLine(
            color = color.copy(alpha = 0.8f),
            start = Offset(0f, y),
            end = Offset(size.width, y),
            strokeWidth = 1.5f,
        )
    }
}

@Composable
fun SectionHeader(title: String, trailing: String? = null) {
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(title, style = MaterialTheme.typography.titleMedium, color = AuroraColors.TextPrimary)
        if (trailing != null) HudLabel(trailing, color = AuroraColors.Cyan)
    }
}

package com.auroravision.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalLayoutDirection
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.LayoutDirection
import androidx.compose.ui.unit.sp

/**
 * "Cinematic intelligence" palette: deep night blues, a cyan scanner accent and
 * restrained violet highlights. Deliberately low-chroma backgrounds so that the
 * frame overlays and progress states stay readable for long sessions.
 */
object AuroraColors {
    val Night = Color(0xFF05080F)
    val Deep = Color(0xFF08101C)
    val Panel = Color(0xFF0C1626)
    val PanelLight = Color(0xFF122034)
    val Cyan = Color(0xFF3FD8FF)
    val CyanDim = Color(0xFF1B6C86)
    val Violet = Color(0xFFB68CFF)
    val Mint = Color(0xFF5BE8B0)
    val Amber = Color(0xFFFFB86B)
    val Danger = Color(0xFFFF6B81)
    val TextPrimary = Color(0xFFE9F4FF)
    val TextSecondary = Color(0xFF8FA8C4)
    val Outline = Color(0xFF1E3350)

    val backgroundGradient = Brush.verticalGradient(
        listOf(Color(0xFF04060B), Color(0xFF071120), Color(0xFF04060B)),
    )

    fun verdictColor(verdict: String): Color = when (verdict) {
        "CANDIDATES_FOUND" -> Cyan
        "NO_RELIABLE_MATCH" -> Amber
        "QUALITY_REJECTED" -> Danger
        "NO_FACES_DETECTED" -> Amber
        "NO_REFERENCE_DATA" -> Violet
        else -> TextSecondary
    }
}

private val AuroraTypography = Typography(
    displaySmall = TextStyle(
        fontFamily = FontFamily.SansSerif,
        fontWeight = FontWeight.ExtraBold,
        fontSize = 30.sp,
        letterSpacing = 6.sp,
    ),
    headlineSmall = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 20.sp),
    titleMedium = TextStyle(fontWeight = FontWeight.SemiBold, fontSize = 16.sp),
    bodyMedium = TextStyle(fontSize = 14.sp, lineHeight = 21.sp),
    bodySmall = TextStyle(fontSize = 12.sp, lineHeight = 18.sp),
    labelSmall = TextStyle(
        fontWeight = FontWeight.Medium,
        fontSize = 10.sp,
        letterSpacing = 1.4.sp,
    ),
)

private val AuroraScheme = darkColorScheme(
    primary = AuroraColors.Cyan,
    onPrimary = Color(0xFF00222E),
    secondary = AuroraColors.Violet,
    background = AuroraColors.Night,
    onBackground = AuroraColors.TextPrimary,
    surface = AuroraColors.Panel,
    onSurface = AuroraColors.TextPrimary,
    surfaceVariant = AuroraColors.PanelLight,
    onSurfaceVariant = AuroraColors.TextSecondary,
    outline = AuroraColors.Outline,
    error = AuroraColors.Danger,
)

/**
 * App theme. The layout direction is forced to RTL because the UI is Arabic
 * first: this keeps Arabic typography and alignment correct even on devices
 * configured in a left-to-right language.
 */
@Composable
fun AuroraTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    CompositionLocalProvider(LocalLayoutDirection provides LayoutDirection.Rtl) {
        MaterialTheme(
            colorScheme = AuroraScheme,
            typography = AuroraTypography,
            content = content,
        )
    }
}

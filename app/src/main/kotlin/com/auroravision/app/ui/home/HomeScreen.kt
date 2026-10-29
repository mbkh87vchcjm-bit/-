package com.auroravision.app.ui.home

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Collections
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Storage
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.auroravision.app.data.db.SearchSessionEntity
import com.auroravision.app.engine.EngineState
import com.auroravision.app.ui.Fmt
import com.auroravision.app.ui.components.AuroraBackground
import com.auroravision.app.ui.components.AuroraPanel
import com.auroravision.app.ui.components.HudLabel
import com.auroravision.app.ui.components.PrimaryAction
import com.auroravision.app.ui.components.SecondaryAction
import com.auroravision.app.ui.components.SectionHeader
import com.auroravision.app.ui.components.StatTile
import com.auroravision.app.ui.theme.AuroraColors

/**
 * Home screen: identity, engine state, the three primary actions, live database
 * statistics and the last searches. Every number is read from the database.
 */
@Composable
fun HomeScreen(
    viewModel: HomeViewModel,
    onStartAnalysis: () -> Unit,
    onOpenCamera: () -> Unit,
    onManageLibrary: () -> Unit,
    onOpenHistory: () -> Unit,
    onOpenSettings: () -> Unit,
    onOpenSession: (Long) -> Unit,
) {
    val state by viewModel.state.collectAsState()
    AuroraBackground {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 18.dp, vertical = 24.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top,
            ) {
                Column {
                    Text(
                        "AURORA VISION",
                        style = MaterialTheme.typography.displaySmall,
                        color = AuroraColors.Cyan,
                        fontSize = 26.sp,
                    )
                    Spacer(Modifier.height(6.dp))
                    Text(
                        "Advanced AI Face Matching Laboratory",
                        style = MaterialTheme.typography.bodySmall,
                        color = AuroraColors.TextSecondary,
                    )
                }
                IconButton(onClick = onOpenSettings) {
                    Icon(Icons.Filled.Info, contentDescription = "معلومات", tint = AuroraColors.TextSecondary)
                }
            }

            Spacer(Modifier.height(14.dp))
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(1.dp)
                    .background(
                        androidx.compose.ui.graphics.Brush.horizontalGradient(
                            listOf(AuroraColors.Cyan.copy(alpha = 0.65f), Color.Transparent),
                        ),
                    ),
            )

            Spacer(Modifier.height(18.dp))
            EngineStatusCard(state.engine)

            Spacer(Modifier.height(18.dp))
            PrimaryAction(
                text = "بدء تحليل صورة",
                icon = Icons.Filled.PlayArrow,
                enabled = state.engine is EngineState.Ready,
                onClick = onStartAnalysis,
            )
            Spacer(Modifier.height(10.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Box(Modifier.weight(1f)) {
                    SecondaryAction("من المعرض", Icons.Filled.Collections, onClick = onStartAnalysis)
                }
                Box(Modifier.weight(1f)) {
                    SecondaryAction("الكاميرا", Icons.Filled.CameraAlt, onClick = onOpenCamera)
                }
            }
            Spacer(Modifier.height(10.dp))
            SecondaryAction(
                text = "إدارة قاعدة الصور",
                icon = Icons.Filled.Storage,
                accent = AuroraColors.Cyan,
                onClick = onManageLibrary,
            )

            Spacer(Modifier.height(22.dp))
            AuroraPanel(Modifier.fillMaxWidth()) {
                Row(
                    Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                ) {
                    StatTile(
                        value = state.stats.registeredImages.toString(),
                        label = "صور مسجلة",
                    )
                    StatTile(
                        value = state.stats.registeredFaces.toString(),
                        label = "وجوه مسجلة",
                        accent = AuroraColors.Violet,
                    )
                    StatTile(
                        value = state.stats.comparisons.toString(),
                        label = "عمليات مقارنة",
                        accent = AuroraColors.Mint,
                    )
                }
            }

            Spacer(Modifier.height(22.dp))
            SectionHeader(
                title = "آخر عمليات البحث",
                trailing = if (state.recent.isNotEmpty()) "اضغط للتفاصيل" else null,
            )
            if (state.recent.isEmpty()) {
                AuroraPanel(Modifier.fillMaxWidth()) {
                    Text(
                        "لا توجد عمليات بحث بعد. ابدأ بتحليل صورة لإنشاء سجل حقيقي.",
                        style = MaterialTheme.typography.bodySmall,
                        color = AuroraColors.TextSecondary,
                    )
                }
            } else {
                state.recent.forEach { session ->
                    RecentSessionRow(session) { onOpenSession(session.id) }
                    Spacer(Modifier.height(8.dp))
                }
                Text(
                    "عرض كل السجل ›",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuroraColors.Cyan,
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable(onClick = onOpenHistory)
                        .padding(vertical = 8.dp),
                    textAlign = TextAlign.End,
                )
            }

            Spacer(Modifier.height(18.dp))
            PolicyFooter(
                threshold = state.policy.operatingThreshold,
                dataset = state.policy.provenance.dataset,
                calibrated = state.policy.provenance.isCalibrated,
            )
            Spacer(Modifier.height(28.dp))
        }
    }
}

@Composable
private fun EngineStatusCard(engine: EngineState) {
    val (label, detail, color) = when (engine) {
        is EngineState.Ready -> Triple(
            "محرك الذكاء الاصطناعي جاهز",
            "${engine.info.displayName} · يعمل محليًا بالكامل دون إنترنت",
            AuroraColors.Mint,
        )
        is EngineState.Loading -> Triple("جارٍ تحميل المحرك…", "يتم تحميل النماذج من أصول التطبيق", AuroraColors.Amber)
        is EngineState.Failed -> Triple(
            "المحرك غير متاح",
            "${engine.reason}${engine.detail?.let { " — $it" } ?: ""}",
            AuroraColors.Danger,
        )
    }
    AuroraPanel(Modifier.fillMaxWidth(), accent = color.copy(alpha = 0.5f)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier
                    .size(9.dp)
                    .clip(CircleShape)
                    .background(color),
            )
            Spacer(Modifier.width(10.dp))
            Column {
                Text(label, style = MaterialTheme.typography.bodyMedium, color = AuroraColors.TextPrimary)
                Text(detail, style = MaterialTheme.typography.bodySmall, color = AuroraColors.TextSecondary)
            }
        }
    }
}

@Composable
private fun RecentSessionRow(session: SearchSessionEntity, onClick: () -> Unit) {
    val color = AuroraColors.verdictColor(session.verdict)
    AuroraPanel(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onClick),
        accent = color.copy(alpha = 0.35f),
    ) {
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(
                    Fmt.verdictTitle(session.verdict),
                    style = MaterialTheme.typography.bodyMedium,
                    color = color,
                )
                Spacer(Modifier.height(2.dp))
                Text(
                    "${Fmt.relative(session.createdAtEpochMs)} · ${session.comparedCount} مقارنة · ${Fmt.duration(session.durationMs)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuroraColors.TextSecondary,
                )
            }
            Column(horizontalAlignment = Alignment.End) {
                Text(
                    session.topSimilarity?.let { Fmt.similarity(it) } ?: "—",
                    style = MaterialTheme.typography.titleMedium,
                    color = AuroraColors.TextPrimary,
                )
                HudLabel("أعلى تشابه")
            }
        }
    }
}

@Composable
private fun PolicyFooter(threshold: Float, dataset: String, calibrated: Boolean) {
    Column {
        HudLabel(
            "عتبة المطابقة الحالية: ${Fmt.similarity(threshold)} " +
                if (calibrated) "· مُعايرة" else "· قيمة مرجعية غير مُعايرة",
            color = if (calibrated) AuroraColors.Cyan else AuroraColors.Amber,
        )
        Spacer(Modifier.height(4.dp))
        Text(
            dataset,
            style = MaterialTheme.typography.bodySmall,
            color = AuroraColors.TextSecondary.copy(alpha = 0.8f),
        )
        Spacer(Modifier.height(2.dp))
        Text(
            "درجة التشابه ليست دليلًا قاطعًا على الهوية. تُستخدم فقط لإظهار مرشّحات للمراجعة.",
            style = MaterialTheme.typography.bodySmall,
            color = AuroraColors.TextSecondary.copy(alpha = 0.8f),
            fontWeight = FontWeight.Normal,
        )
    }
}

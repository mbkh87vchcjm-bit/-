package com.auroravision.app.ui.analyze

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.RadioButtonUnchecked
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.auroravision.app.ui.Fmt
import com.auroravision.app.ui.components.AuroraBackground
import com.auroravision.app.ui.components.AuroraPanel
import com.auroravision.app.ui.components.DetectedFaceOverlay
import com.auroravision.app.ui.components.FaceOverlay
import com.auroravision.app.ui.components.HudLabel
import com.auroravision.app.ui.components.MatchRow
import com.auroravision.app.ui.components.PrimaryAction
import com.auroravision.app.ui.components.ScanningOverlay
import com.auroravision.app.ui.theme.AuroraColors
import com.auroravision.core.pipeline.AnalysisStage
import com.auroravision.core.pipeline.MatchVerdict
import com.auroravision.core.quality.IssueSeverity
import com.auroravision.core.quality.QualityStatus

/**
 * Analysis screen: the image with the detected geometry, the seven phases tied
 * to real work, the face selector, and the result summary.
 */
@Composable
fun AnalyzeScreen(
    viewModel: AnalyzeViewModel,
    onOpenReport: (Long) -> Unit,
    onBack: () -> Unit,
) {
    val state by viewModel.state.collectAsState()
    AuroraBackground {
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 16.dp),
        ) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                HudLabel("ANALYSIS CONSOLE", color = AuroraColors.Cyan)
                Text(
                    "رجوع",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuroraColors.TextSecondary,
                    modifier = Modifier.clickable(onClick = onBack).padding(4.dp),
                )
            }
            Spacer(Modifier.height(10.dp))

            // ---------------- image preview + overlay ---------------- //
            Box(
                Modifier
                    .fillMaxWidth()
                    .aspectRatio(1.15f)
                    .clip(RoundedCornerShape(16.dp))
                    .background(AuroraColors.Panel),
            ) {
                val bitmap = state.imageBitmap
                if (bitmap != null) {
                    Image(
                        bitmap = bitmap.asImageBitmap(),
                        contentDescription = "الصورة قيد التحليل",
                        contentScale = ContentScale.Fit,
                        modifier = Modifier.fillMaxSize(),
                    )
                    FaceOverlay(
                        imageWidth = state.imageWidth,
                        imageHeight = state.imageHeight,
                        faces = state.faces.map {
                            DetectedFaceOverlay(it.index, it.box, it.landmarks, it.quality.status)
                        },
                        selectedIndex = state.selectedFaceIndex.takeIf { state.faces.size > 1 },
                    )
                } else {
                    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                        CircularProgressIndicator(color = AuroraColors.Cyan)
                    }
                }
                if (state.running) {
                    ScanningOverlay(Modifier.fillMaxSize())
                }
            }

            Spacer(Modifier.height(14.dp))

            // ---------------- phases ---------------- //
            AuroraPanel(Modifier.fillMaxWidth()) {
                HudLabel("PROCESSING PHASES", color = AuroraColors.Cyan)
                Spacer(Modifier.height(8.dp))
                AnalysisStage.entries.forEach { stage ->
                    PhaseRow(
                        stage = stage,
                        state = state,
                    )
                }
                if (state.running && state.currentStage != null) {
                    Spacer(Modifier.height(8.dp))
                    LinearProgressIndicator(
                        progress = { state.stageProgress },
                        modifier = Modifier.fillMaxWidth().height(3.dp),
                        color = AuroraColors.Cyan,
                        trackColor = AuroraColors.Outline,
                    )
                }
            }

            // ---------------- faces ---------------- //
            if (state.faces.size > 1) {
                Spacer(Modifier.height(14.dp))
                HudLabel("الوجوه المكتشفة — اختر وجهًا للمقارنة")
                Spacer(Modifier.height(8.dp))
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    items(state.faces.size) { i ->
                        val face = state.faces[i]
                        FaceChip(
                            face = face,
                            selected = state.selectedFaceIndex == face.index,
                            onClick = { viewModel.selectFace(face.index) },
                        )
                    }
                }
            }

            // ---------------- quality ---------------- //
            state.selectedFace?.let { face ->
                Spacer(Modifier.height(14.dp))
                QualityCard(face)
            }

            // ---------------- verdict + matches ---------------- //
            AnimatedVisibility(visible = !state.running && state.verdict != null) {
                Column {
                    Spacer(Modifier.height(14.dp))
                    val verdict = state.verdict?.name ?: ""
                    AuroraPanel(Modifier.fillMaxWidth(), accent = AuroraColors.verdictColor(verdict)) {
                        Text(
                            Fmt.verdictTitle(verdict),
                            style = MaterialTheme.typography.titleMedium,
                            color = AuroraColors.verdictColor(verdict),
                        )
                        Spacer(Modifier.height(6.dp))
                        Text(
                            Fmt.verdictExplanation(verdict, state.search?.threshold ?: 0f),
                            style = MaterialTheme.typography.bodySmall,
                            color = AuroraColors.TextSecondary,
                        )
                        state.search?.let { search ->
                            Spacer(Modifier.height(10.dp))
                            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                HudLabel("تمت مقارنة ${search.comparedCount} سجل")
                                HudLabel("${Fmt.duration(search.durationMs)} للبحث")
                            }
                            if (search.skippedModelMismatch > 0) {
                                HudLabel(
                                    "استُثني ${search.skippedModelMismatch} سجلًا لأنها من إصدار نموذج مختلف",
                                    color = AuroraColors.Amber,
                                )
                            }
                        }
                    }

                    if (state.matches.isNotEmpty()) {
                        Spacer(Modifier.height(14.dp))
                        HudLabel("أفضل النتائج (${state.matches.size})")
                        Spacer(Modifier.height(8.dp))
                        state.matches.take(3).forEach { match ->
                            MatchRow(
                                name = match.displayName,
                                similarity = match.similarity,
                                band = match.band.name,
                                sourceLabel = match.sourceLabel,
                                addedAt = match.addedAtEpochMs,
                                thumbnailPath = match.thumbnailPath,
                            )
                            Spacer(Modifier.height(8.dp))
                        }
                    }

                    if (state.sessionId != null) {
                        Spacer(Modifier.height(6.dp))
                        PrimaryAction(
                            text = "عرض التقرير الكامل",
                            icon = Icons.Filled.ArrowForward,
                            onClick = { onOpenReport(state.sessionId!!) },
                        )
                    }
                }
            }

            state.error?.let { error ->
                Spacer(Modifier.height(14.dp))
                AuroraPanel(Modifier.fillMaxWidth(), accent = AuroraColors.Danger) {
                    Text(error, color = AuroraColors.Danger, style = MaterialTheme.typography.bodyMedium)
                }
            }
            Spacer(Modifier.height(28.dp))
        }
    }
}

@Composable
private fun PhaseRow(stage: AnalysisStage, state: AnalyzeUiState) {
    val done = state.completedStages.contains(stage)
    val active = state.running && state.currentStage == stage && !done
    val skipped = !state.running && state.completedStages.isNotEmpty() && !done
    val color = when {
        done -> AuroraColors.Mint
        active -> AuroraColors.Cyan
        skipped -> AuroraColors.TextSecondary
        else -> AuroraColors.TextSecondary.copy(alpha = 0.5f)
    }
    Row(
        Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (active) {
            CircularProgressIndicator(
                modifier = Modifier.size(13.dp),
                strokeWidth = 2.dp,
                color = AuroraColors.Cyan,
            )
        } else {
            Icon(
                if (done) Icons.Filled.CheckCircle else Icons.Filled.RadioButtonUnchecked,
                contentDescription = null,
                tint = color,
                modifier = Modifier.size(14.dp),
            )
        }
        Spacer(Modifier.width(10.dp))
        Text(
            stage.ordinalLabel,
            style = MaterialTheme.typography.bodySmall,
            color = color,
            modifier = Modifier.weight(1f),
        )
        val timing = state.timings.firstOrNull { it.stage == stage }
        if (timing != null) {
            Text(
                "${timing.durationMs} ms",
                style = MaterialTheme.typography.labelSmall,
                color = AuroraColors.TextSecondary.copy(alpha = 0.7f),
            )
        }
    }
}

@Composable
private fun QualityCard(face: FaceUi) {
    val (label, color) = when (face.quality.status) {
        QualityStatus.OK -> "جودة الوجه: جيدة" to AuroraColors.Mint
        QualityStatus.WARNING -> "جودة الوجه: مقبولة مع تحذير" to AuroraColors.Amber
        QualityStatus.REJECTED -> "جودة الوجه: غير صالحة للمقارنة" to AuroraColors.Danger
    }
    AuroraPanel(Modifier.fillMaxWidth(), accent = color.copy(alpha = 0.45f)) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text(label, style = MaterialTheme.typography.bodyMedium, color = color)
            HudLabel("درجة الجودة ${(face.quality.score * 100).toInt()}%")
        }
        Spacer(Modifier.height(8.dp))
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            MetricPill("الحدّة", face.quality.sharpness.toInt().toString())
            MetricPill("السطوع", face.quality.brightness.toInt().toString())
            MetricPill("التباين", face.quality.contrast.toInt().toString())
            MetricPill("حجم الوجه", "${face.quality.faceShortSidePx.toInt()}px")
        }
        if (face.quality.issues.isNotEmpty()) {
            Spacer(Modifier.height(10.dp))
            face.quality.issues.forEach { issue ->
                val issueColor = when (issue.severity) {
                    IssueSeverity.REJECTED -> AuroraColors.Danger
                    IssueSeverity.WARNING -> AuroraColors.Amber
                    IssueSeverity.INFO -> AuroraColors.TextSecondary
                }
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(5.dp).clip(CircleShape).background(issueColor))
                    Spacer(Modifier.width(8.dp))
                    Text(
                        "${issue.messageAr} (${issue.value} ← ${issue.limit})",
                        style = MaterialTheme.typography.bodySmall,
                        color = issueColor,
                    )
                }
            }
        }
    }
}

@Composable
private fun MetricPill(label: String, value: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(value, style = MaterialTheme.typography.bodyMedium, color = AuroraColors.TextPrimary)
        HudLabel(label)
    }
}

@Composable
private fun FaceChip(face: FaceUi, selected: Boolean, onClick: () -> Unit) {
    val border = if (selected) AuroraColors.Cyan else AuroraColors.Outline
    Column(
        Modifier
            .width(78.dp)
            .clip(RoundedCornerShape(12.dp))
            .background(AuroraColors.Panel)
            .clickable(onClick = onClick)
            .padding(4.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Box(
            Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .clip(RoundedCornerShape(9.dp))
                .background(AuroraColors.Deep),
        ) {
            face.alignedPreview?.let {
                Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = "وجه ${face.index + 1}",
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize(),
                )
            }
            Box(
                Modifier
                    .fillMaxSize()
                    .clip(RoundedCornerShape(9.dp))
                    .background(Color.Transparent),
            )
        }
        Spacer(Modifier.height(4.dp))
        Text(
            "وجه ${face.index + 1}",
            style = MaterialTheme.typography.labelSmall,
            color = border,
            fontWeight = FontWeight.Medium,
        )
    }
}

package com.auroravision.app.ui.results

import androidx.compose.foundation.Image
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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import com.auroravision.app.data.repo.SessionDetail
import com.auroravision.app.ui.Fmt
import com.auroravision.app.ui.components.AuroraBackground
import com.auroravision.app.ui.components.AuroraPanel
import com.auroravision.app.ui.components.HudLabel
import com.auroravision.app.ui.components.MatchRow
import com.auroravision.app.ui.components.ReportRow
import com.auroravision.app.ui.components.ReportStat
import com.auroravision.app.ui.components.SecondaryAction
import com.auroravision.app.ui.theme.AuroraColors
import kotlinx.serialization.json.Json
import java.io.File

/**
 * Full report of one analysis: everything the brief asks to show, taken from the
 * stored `AnalysisReport` rather than from freshly guessed values.
 */
@Composable
fun ResultsScreen(
    viewModel: ResultsViewModel,
    onBack: () -> Unit,
    onDeleted: () -> Unit,
) {
    val state by viewModel.state.collectAsState()
    AuroraBackground {
        when {
            state.loading -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                CircularProgressIndicator(color = AuroraColors.Cyan)
            }
            state.detail == null -> Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                Text(state.error ?: "لا توجد بيانات", color = AuroraColors.Danger)
            }
            else -> {
                val detail = state.detail!!
                ReportContent(detail, onBack) { viewModel.deleteSession(onDeleted) }
            }
        }
    }
}

@Composable
private fun ReportContent(detail: SessionDetail, onBack: () -> Unit, onDelete: () -> Unit) {
    val report = detail.report
    val session = detail.session
    val verdictColor = AuroraColors.verdictColor(report.verdict.name)
    val queryBitmap = remember(session.queryImagePath) {
        runCatching {
            android.graphics.BitmapFactory.decodeFile(File(session.queryImagePath).absolutePath)
        }.getOrNull()
    }

    Column(
        Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
    ) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            HudLabel("RESULTS REPORT", color = AuroraColors.Cyan)
            Text(
                "رجوع",
                style = MaterialTheme.typography.bodySmall,
                color = AuroraColors.TextSecondary,
                modifier = Modifier.clickable(onClick = onBack).padding(4.dp),
            )
        }
        Spacer(Modifier.height(12.dp))

        // ---- query image + verdict ----
        AuroraPanel(Modifier.fillMaxWidth(), accent = verdictColor) {
            Row {
                Box(
                    Modifier
                        .size(96.dp)
                        .clip(RoundedCornerShape(10.dp))
                        .background(AuroraColors.Deep),
                ) {
                    queryBitmap?.let {
                        Image(
                            bitmap = it.asImageBitmap(),
                            contentDescription = "الصورة التي جرى البحث عنها",
                            contentScale = ContentScale.Crop,
                            modifier = Modifier.fillMaxSize(),
                        )
                    }
                }
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(
                        Fmt.verdictTitle(report.verdict.name),
                        style = MaterialTheme.typography.titleMedium,
                        color = verdictColor,
                    )
                    Spacer(Modifier.height(4.dp))
                    Text(
                        Fmt.verdictExplanation(report.verdict.name, report.policy.operatingThreshold),
                        style = MaterialTheme.typography.bodySmall,
                        color = AuroraColors.TextSecondary,
                    )
                }
            }
        }

        Spacer(Modifier.height(14.dp))
        // ---- measured facts ----
        AuroraPanel(Modifier.fillMaxWidth()) {
            HudLabel("MEASUREMENTS")
            Spacer(Modifier.height(8.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                ReportStat("وجوه مكتشفة", report.facesDetected.toString())
                ReportStat("قابلة للمقارنة", report.facesComparable.toString(), AuroraColors.Mint)
                ReportStat("صور فُحصت", (report.search?.comparedCount ?: 0).toString())
                ReportStat("زمن التحليل", Fmt.duration(report.totalDurationMs), AuroraColors.Cyan)
            }
            Spacer(Modifier.height(10.dp))
            ReportRow("حالة الجودة", Fmt.qualityLabel(session.qualityStatus))
            ReportRow("درجة الجودة", "${(session.qualityScore * 100).toInt()}%")
            ReportRow("عتبة التشابه", Fmt.similarity(report.policy.operatingThreshold))
            ReportRow("نوع المقياس", "تشابه جيب التمام (cosine)")
            report.search?.let { search ->
                ReportRow("زمن البحث", Fmt.duration(search.durationMs))
                ReportRow("اكتمل البحث", if (search.completed) "نعم" else "لا")
                if (search.skippedModelMismatch > 0) {
                    ReportRow("سجلات مستثناة (نموذج مختلف)", search.skippedModelMismatch.toString())
                }
            }
            ReportRow("أبعاد الصورة", "${report.imageWidth} × ${report.imageHeight}")
            ReportRow("إصدار النموذج", "${report.detectorVersion} + ${report.embedderVersion}")
            ReportRow("محرك التحليل", report.engineVersion)
            ReportRow("تاريخ البحث", Fmt.dateTime(session.createdAtEpochMs))
        }

        // ---- stage timings ----
        Spacer(Modifier.height(14.dp))
        AuroraPanel(Modifier.fillMaxWidth()) {
            HudLabel("TIMINGS")
            Spacer(Modifier.height(6.dp))
            report.timings.forEach { timing ->
                ReportRow(timing.stage.ordinalLabel, "${timing.durationMs} ms")
            }
        }

        // ---- faces ----
        if (report.faces.isNotEmpty()) {
            Spacer(Modifier.height(14.dp))
            AuroraPanel(Modifier.fillMaxWidth()) {
                HudLabel("QUALITY PER FACE")
                Spacer(Modifier.height(6.dp))
                report.faces.forEach { face ->
                    ReportRow(
                        "وجه ${face.index + 1} · ${Fmt.qualityLabel(face.quality.status.name)}",
                        "حدّة ${face.quality.sharpness.toInt()} · سطوع ${face.quality.brightness.toInt()} · " +
                            "${face.quality.faceShortSidePx.toInt()}px",
                    )
                    face.quality.issues.forEach { issue ->
                        Text(
                            "• ${issue.messageAr}",
                            style = MaterialTheme.typography.bodySmall,
                            color = AuroraColors.Amber,
                        )
                    }
                }
            }
        }

        // ---- matches ----
        Spacer(Modifier.height(14.dp))
        HudLabel("BEST MATCHES")
        Spacer(Modifier.height(8.dp))
        if (detail.matchedRecords.isEmpty()) {
            AuroraPanel(Modifier.fillMaxWidth()) {
                Text(
                    "لا توجد نتائج مرتبة لهذا البحث (لم تُجرَ مقارنة أو لا توجد سجلات مؤهلة).",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuroraColors.TextSecondary,
                )
            }
        } else {
            report.search?.hits?.forEach { hit ->
                val record = detail.matchedRecords.firstOrNull { it.id == hit.recordId }
                MatchRow(
                    name = record?.displayName?.takeIf { it.isNotBlank() } ?: "سجل #${hit.recordId}",
                    similarity = hit.similarity,
                    band = hit.band.name,
                    sourceLabel = record?.sourceLabel ?: "—",
                    addedAt = record?.addedAtEpochMs ?: 0L,
                    thumbnailPath = record?.thumbnailPath ?: record?.imagePath,
                    rank = hit.rank,
                )
                Spacer(Modifier.height(8.dp))
                if (record != null) {
                    HudLabel(
                        "المصدر: ${record.sourceDetail ?: record.sourceLabel} · " +
                            "أُضيفت ${Fmt.dateTime(record.addedAtEpochMs)} · جودة ${Fmt.qualityLabel(record.qualityStatus)}",
                        color = AuroraColors.TextSecondary.copy(alpha = 0.75f),
                    )
                    Spacer(Modifier.height(6.dp))
                }
            }
        }

        Spacer(Modifier.height(14.dp))
        AuroraPanel(Modifier.fillMaxWidth(), accent = AuroraColors.Amber.copy(alpha = 0.4f)) {
            Text(
                "تنبيه: درجة التشابه مقياس بصري وليست احتمالًا مؤكدًا للهوية. " +
                    "لا يجوز استخدامها وحدها لتحديد هوية شخص، وأي نتيجة تتطلب مراجعة بشرية " +
                    "ضمن نطاق البيانات المصرح باستخدامها.",
                style = MaterialTheme.typography.bodySmall,
                color = AuroraColors.Amber,
            )
        }

        Spacer(Modifier.height(16.dp))
        SecondaryAction(
            text = "حذف هذا السجل وصورته",
            icon = Icons.Filled.DeleteOutline,
            accent = AuroraColors.Danger,
            onClick = onDelete,
        )
        Spacer(Modifier.height(24.dp))
    }
}

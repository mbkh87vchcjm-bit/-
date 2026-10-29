package com.auroravision.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.auroravision.app.ui.Fmt
import com.auroravision.app.ui.theme.AuroraColors
import java.io.File

/**
 * One ranked result: thumbnail, the record's own metadata, and the measured
 * similarity with the band it falls into. Weak matches are shown too — the app
 * explains *why* something did not qualify instead of hiding it.
 */
@Composable
fun MatchRow(
    name: String,
    similarity: Float,
    band: String,
    sourceLabel: String,
    addedAt: Long,
    thumbnailPath: String?,
    modifier: Modifier = Modifier,
    rank: Int? = null,
) {
    val color = when (band) {
        "MATCH_CANDIDATE" -> AuroraColors.Cyan
        "INCONCLUSIVE" -> AuroraColors.Amber
        else -> AuroraColors.TextSecondary
    }
    AuroraPanel(modifier.fillMaxWidth(), accent = color.copy(alpha = 0.35f)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier
                    .size(56.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(AuroraColors.Deep),
            ) {
                if (thumbnailPath != null) {
                    AsyncImage(
                        model = File(thumbnailPath),
                        contentDescription = "صورة السجل",
                        contentScale = ContentScale.Crop,
                        modifier = Modifier.fillMaxWidth().height(56.dp),
                    )
                }
            }
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (rank != null) {
                        HudLabel("#$rank", color = AuroraColors.Cyan)
                        Spacer(Modifier.width(6.dp))
                    }
                    Text(
                        name,
                        style = MaterialTheme.typography.bodyMedium,
                        color = AuroraColors.TextPrimary,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                }
                Spacer(Modifier.height(2.dp))
                Text(
                    "$sourceLabel · أُضيفت ${Fmt.date(addedAt)}",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuroraColors.TextSecondary,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            Spacer(Modifier.width(10.dp))
            Column(horizontalAlignment = Alignment.End) {
                Text(Fmt.similarity(similarity), style = MaterialTheme.typography.titleMedium, color = color)
                HudLabel(Fmt.bandLabel(band), color = color)
            }
        }
        Spacer(Modifier.height(8.dp))
        // similarity bar: full width equals similarity 1.0
        Box(
            Modifier
                .fillMaxWidth()
                .height(3.dp)
                .clip(RoundedCornerShape(2.dp))
                .background(AuroraColors.Outline),
        ) {
            Box(
                Modifier
                    .fillMaxWidth(similarity.coerceIn(0f, 1f))
                    .height(3.dp)
                    .background(color),
            )
        }
    }
}

/** Compact key/value line used in the report grid. */
@Composable
fun ReportStat(label: String, value: String, color: androidx.compose.ui.graphics.Color = AuroraColors.TextPrimary) {
    Column {
        Text(value, style = MaterialTheme.typography.bodyMedium, color = color)
        HudLabel(label)
    }
}

@Composable
fun ReportRow(label: String, value: String) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
    ) {
        Text(label, style = MaterialTheme.typography.bodySmall, color = AuroraColors.TextSecondary)
        Text(value, style = MaterialTheme.typography.bodySmall, color = AuroraColors.TextPrimary)
    }
}

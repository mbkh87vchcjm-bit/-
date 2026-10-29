package com.auroravision.app.ui.history

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.auroravision.app.ui.Fmt
import com.auroravision.app.ui.components.AuroraBackground
import com.auroravision.app.ui.components.AuroraPanel
import com.auroravision.app.ui.components.HudLabel
import com.auroravision.app.ui.theme.AuroraColors

/** Search history: open a stored report again, or delete it with its query image. */
@Composable
fun HistoryScreen(
    viewModel: HistoryViewModel,
    onOpenSession: (Long) -> Unit,
    onBack: () -> Unit,
) {
    val state by viewModel.state.collectAsState()
    var confirmAll by remember { mutableStateOf(false) }

    AuroraBackground {
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
        ) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                HudLabel("SEARCH HISTORY", color = AuroraColors.Cyan)
                Text(
                    "رجوع",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuroraColors.TextSecondary,
                    modifier = Modifier.clickable(onClick = onBack).padding(4.dp),
                )
            }
            Spacer(Modifier.height(12.dp))
            if (state.sessions.isEmpty()) {
                AuroraPanel(Modifier.fillMaxWidth()) {
                    Text(
                        "السجل فارغ.",
                        style = MaterialTheme.typography.bodySmall,
                        color = AuroraColors.TextSecondary,
                    )
                }
            } else {
                state.sessions.forEach { session ->
                    val color = AuroraColors.verdictColor(session.verdict)
                    AuroraPanel(
                        Modifier
                            .fillMaxWidth()
                            .clickable { onOpenSession(session.id) },
                        accent = color.copy(alpha = 0.3f),
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text(
                                    Fmt.verdictTitle(session.verdict),
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = color,
                                )
                                HudLabel(
                                    "${Fmt.dateTime(session.createdAtEpochMs)} · " +
                                        "${session.comparedCount} مقارنة · ${Fmt.duration(session.durationMs)}",
                                )
                            }
                            Column(horizontalAlignment = Alignment.End) {
                                Text(
                                    session.topSimilarity?.let { Fmt.similarity(it) } ?: "—",
                                    style = MaterialTheme.typography.bodyMedium,
                                    color = AuroraColors.TextPrimary,
                                )
                                HudLabel("أعلى تشابه")
                            }
                            IconButton(onClick = { viewModel.delete(session.id) }) {
                                Icon(
                                    Icons.Filled.DeleteOutline,
                                    contentDescription = "حذف",
                                    tint = AuroraColors.Danger,
                                )
                            }
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                }
                Spacer(Modifier.height(8.dp))
                TextButton(onClick = { confirmAll = true }) {
                    Text("حذف كل السجل", color = AuroraColors.Danger)
                }
            }
            Spacer(Modifier.height(24.dp))
        }
    }

    if (confirmAll) {
        AlertDialog(
            onDismissRequest = { confirmAll = false },
            title = { Text("حذف كل سجل البحث؟") },
            text = { Text("سيُحذف كل سجل مع صور الاستعلام المحفوظة. لا يمكن التراجع.") },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.deleteAll()
                    confirmAll = false
                }) { Text("حذف", color = AuroraColors.Danger) }
            },
            dismissButton = { TextButton(onClick = { confirmAll = false }) { Text("إلغاء") } },
        )
    }
}

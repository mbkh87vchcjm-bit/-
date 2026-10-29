package com.auroravision.app.ui.library

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
import androidx.compose.material.icons.filled.AddPhotoAlternate
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.DeleteOutline
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.auroravision.app.ui.Fmt
import com.auroravision.app.ui.components.AuroraBackground
import com.auroravision.app.ui.components.AuroraPanel
import com.auroravision.app.ui.components.HudLabel
import com.auroravision.app.ui.components.PrimaryAction
import com.auroravision.app.ui.components.SecondaryAction
import com.auroravision.app.ui.theme.AuroraColors
import java.io.File

/**
 * Reference-image manager. Import runs the real pipeline per image; deletion
 * removes the stored image, its thumbnail and the encrypted embedding.
 */
@Composable
fun LibraryScreen(
    viewModel: ImageLibraryViewModel,
    onPickImages: () -> Unit,
    onOpenCamera: () -> Unit,
    onBack: () -> Unit,
) {
    val state by viewModel.state.collectAsState()
    var label by remember { mutableStateOf("") }
    var confirmDeleteAll by remember { mutableStateOf(false) }
    var pendingDeleteGroup by remember { mutableStateOf<LibraryImage?>(null) }

    AuroraBackground {
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
        ) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                HudLabel("REFERENCE IMAGE DATABASE", color = AuroraColors.Cyan)
                Text(
                    "رجوع",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuroraColors.TextSecondary,
                    modifier = Modifier.clickable(onClick = onBack).padding(4.dp),
                )
            }
            Spacer(Modifier.height(10.dp))

            AuroraPanel(Modifier.fillMaxWidth()) {
                Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                    Column {
                        Text(
                            "${state.images.size}",
                            style = MaterialTheme.typography.headlineSmall,
                            color = AuroraColors.Cyan,
                        )
                        HudLabel("صور مسجلة")
                    }
                    Column {
                        Text(
                            "${state.totalFaces}",
                            style = MaterialTheme.typography.headlineSmall,
                            color = AuroraColors.Violet,
                        )
                        HudLabel("وجوه مفهرسة")
                    }
                    Column {
                        Text(
                            state.images.sumOf { it.records.count { r -> r.qualityStatus == "OK" } }.toString(),
                            style = MaterialTheme.typography.headlineSmall,
                            color = AuroraColors.Mint,
                        )
                        HudLabel("جودة جيدة")
                    }
                }
            }

            Spacer(Modifier.height(14.dp))
            OutlinedTextField(
                value = label,
                onValueChange = { label = it },
                label = { Text("اسم اختياري للسجلات (مثال: أحمد)") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
            )
            Spacer(Modifier.height(10.dp))
            PrimaryAction(
                text = if (state.registering) "جارٍ تحليل الصور وإضافتها…" else "إضافة صور إلى القاعدة",
                icon = Icons.Filled.AddPhotoAlternate,
                enabled = !state.registering && !state.busy,
                onClick = onPickImages,
            )
            Spacer(Modifier.height(8.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Box(Modifier.weight(1f)) {
                    SecondaryAction(
                        "التقاط صورة",
                        Icons.Filled.CameraAlt,
                        enabled = !state.registering,
                        onClick = onOpenCamera,
                    )
                }
                Box(Modifier.weight(1f)) {
                    SecondaryAction(
                        "إعادة فهرسة",
                        Icons.Filled.Refresh,
                        enabled = !state.registering && !state.busy && state.images.isNotEmpty(),
                        accent = AuroraColors.Mint,
                        onClick = { viewModel.reindex() },
                    )
                }
            }

            state.progress?.let { progress ->
                Spacer(Modifier.height(12.dp))
                AuroraPanel(Modifier.fillMaxWidth()) {
                    Text(
                        "معالجة ${progress.current}/${progress.total}: ${progress.fileName}",
                        style = MaterialTheme.typography.bodySmall,
                        color = AuroraColors.TextPrimary,
                    )
                    Spacer(Modifier.height(6.dp))
                    LinearProgressIndicator(
                        progress = { progress.current.toFloat() / progress.total.coerceAtLeast(1) },
                        modifier = Modifier.fillMaxWidth().height(3.dp),
                        color = AuroraColors.Cyan,
                        trackColor = AuroraColors.Outline,
                    )
                    HudLabel("هذه ليست مرحلة تجميلية: كل صورة تُحلَّل فعليًا وتُرفض إن لم تصلح")
                }
            }

            state.message?.let { message ->
                Spacer(Modifier.height(12.dp))
                AuroraPanel(Modifier.fillMaxWidth(), accent = AuroraColors.Mint.copy(alpha = 0.4f)) {
                    Text(message, style = MaterialTheme.typography.bodySmall, color = AuroraColors.Mint)
                }
            }
            state.lastReport?.failures?.takeIf { it.isNotEmpty() }?.let { failures ->
                Spacer(Modifier.height(8.dp))
                AuroraPanel(Modifier.fillMaxWidth(), accent = AuroraColors.Amber.copy(alpha = 0.4f)) {
                    HudLabel("صور لم تُضف", color = AuroraColors.Amber)
                    failures.take(6).forEach {
                        Text("• $it", style = MaterialTheme.typography.bodySmall, color = AuroraColors.Amber)
                    }
                }
            }

            Spacer(Modifier.height(18.dp))
            HudLabel("الصور المسجلة")
            Spacer(Modifier.height(8.dp))
            if (state.images.isEmpty()) {
                AuroraPanel(Modifier.fillMaxWidth()) {
                    Text(
                        "لا توجد صور مسجلة. أضف صورًا مصرحًا باستخدامها لتصبح قابلة للبحث.",
                        style = MaterialTheme.typography.bodySmall,
                        color = AuroraColors.TextSecondary,
                    )
                }
            } else {
                state.images.forEach { image ->
                    LibraryRow(
                        image = image,
                        onDelete = { pendingDeleteGroup = image },
                    )
                    Spacer(Modifier.height(8.dp))
                }
                Spacer(Modifier.height(8.dp))
                TextButton(onClick = { confirmDeleteAll = true }) {
                    Text("حذف كل الصور والتمثيلات", color = AuroraColors.Danger)
                }
            }
            Spacer(Modifier.height(28.dp))
        }
    }

    if (confirmDeleteAll) {
        AlertDialog(
            onDismissRequest = { confirmDeleteAll = false },
            title = { Text("حذف كل البيانات؟") },
            text = {
                Text(
                    "سيتم حذف كل الصور المسجلة وتمثيلاتها الرقمية المشفرة نهائيًا. " +
                        "سجلات البحث تبقى في قسم السجل حتى تحذفها.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.deleteAll()
                    confirmDeleteAll = false
                }) { Text("حذف نهائي", color = AuroraColors.Danger) }
            },
            dismissButton = {
                TextButton(onClick = { confirmDeleteAll = false }) { Text("إلغاء") }
            },
        )
    }

    pendingDeleteGroup?.let { image ->
        AlertDialog(
            onDismissRequest = { pendingDeleteGroup = null },
            title = { Text("حذف الصورة والسجلات المرتبطة؟") },
            text = {
                Text(
                    "الصورة: ${image.primary.sourceDetail ?: image.groupId.take(8)}\n" +
                        "الوجوه المرتبطة: ${image.faceCount}\n" +
                        "سيُحذف الملف والتمثيلات الرقمية المشفرة الخاصة به.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.deleteImage(image.groupId)
                    pendingDeleteGroup = null
                }) { Text("حذف", color = AuroraColors.Danger) }
            },
            dismissButton = {
                TextButton(onClick = { pendingDeleteGroup = null }) { Text("إلغاء") }
            },
        )
    }
}

@Composable
private fun LibraryRow(image: LibraryImage, onDelete: () -> Unit) {
    val record = image.primary
    AuroraPanel(Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier
                    .size(64.dp)
                    .clip(RoundedCornerShape(10.dp))
                    .background(AuroraColors.Deep),
            ) {
                AsyncImage(
                    model = File(record.thumbnailPath ?: record.imagePath),
                    contentDescription = null,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize(),
                )
            }
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(
                    image.label,
                    style = MaterialTheme.typography.bodyMedium,
                    color = AuroraColors.TextPrimary,
                )
                HudLabel(
                    "${image.faceCount} وجه · ${record.sourceLabel} · ${Fmt.date(record.addedAtEpochMs)}",
                )
                HudLabel(
                    "أبعاد ${record.imageWidth}×${record.imageHeight} · جودة ${Fmt.qualityLabel(record.qualityStatus)}",
                    color = AuroraColors.TextSecondary.copy(alpha = 0.7f),
                )
            }
            IconButton(onClick = onDelete) {
                Icon(Icons.Filled.DeleteOutline, contentDescription = "حذف", tint = AuroraColors.Danger)
            }
        }
    }
}

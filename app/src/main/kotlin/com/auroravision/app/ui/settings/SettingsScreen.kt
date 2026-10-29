package com.auroravision.app.ui.settings

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
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.auroravision.app.ui.Fmt
import com.auroravision.app.ui.components.AuroraBackground
import com.auroravision.app.ui.components.AuroraPanel
import com.auroravision.app.ui.components.HudLabel
import com.auroravision.app.ui.components.ReportRow
import com.auroravision.app.ui.components.SecondaryAction
import com.auroravision.app.ui.theme.AuroraColors

/**
 * Model licences, calibration provenance, privacy policy and the "delete
 * everything" control. The app is academic software: the numbers and the
 * limitations are stated here, in the product itself.
 */
@Composable
fun SettingsScreen(viewModel: SettingsViewModel, onBack: () -> Unit) {
    val state by viewModel.state.collectAsState()
    var confirmDelete by remember { mutableStateOf(false) }

    AuroraBackground {
        Column(
            Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(16.dp),
        ) {
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                HudLabel("MODEL & PRIVACY", color = AuroraColors.Cyan)
                Text(
                    "رجوع",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuroraColors.TextSecondary,
                    modifier = Modifier.clickable(onClick = onBack).padding(4.dp),
                )
            }
            Spacer(Modifier.height(12.dp))

            AuroraPanel(Modifier.fillMaxWidth()) {
                HudLabel("MODELS")
                Spacer(Modifier.height(6.dp))
                ReportRow("المحرك", state.models.engineVersion)
                ReportRow("كاشف الوجوه", state.models.detectorFile)
                ReportRow("إصدار الكاشف", state.models.detectorVersion)
                ReportRow("ترخيص الكاشف", state.models.detectorLicense)
                ReportRow("المصدر", state.models.detectorSource)
                Spacer(Modifier.height(6.dp))
                ReportRow("نموذج التمثيل", state.models.embedderFile)
                ReportRow("إصدار النموذج", state.models.embedderVersion)
                ReportRow("ترخيص النموذج", state.models.embedderLicense)
                ReportRow("المصدر", state.models.embedderSource)
                ReportRow("أبعاد التمثيل", "${state.models.embeddingDim} بُعدًا")
                ReportRow("مقياس التشابه", "cosine")
                Spacer(Modifier.height(8.dp))
                HudLabel("SHA-256 للكاشف: ${state.models.detectorSha256.take(24)}…")
                HudLabel("SHA-256 للنموذج: ${state.models.embedderSha256.take(24)}…")
            }

            Spacer(Modifier.height(14.dp))
            AuroraPanel(Modifier.fillMaxWidth(), accent = AuroraColors.Cyan.copy(alpha = 0.4f)) {
                HudLabel("MATCHING POLICY")
                Spacer(Modifier.height(6.dp))
                ReportRow("العتبة الفعّالة", Fmt.similarity(state.policy.operatingThreshold))
                ReportRow("حد عدم الحسم", Fmt.similarity(state.policy.inconclusiveFloor))
                ReportRow("FAR المستهدف", (state.policy.targetFar * 100).toString() + "%")
                ReportRow("عدد النتائج", "حتى ${state.policy.defaultTopK} (الحد الأقصى ${state.policy.maxTopK})")
                Spacer(Modifier.height(6.dp))
                if (state.policy.provenance.isCalibrated) {
                    ReportRow("AUC المقاس", state.measuredAuc?.let { Fmt.similarity(it) } ?: "—")
                    ReportRow("EER المقاس", state.measuredEer?.let { Fmt.percent(it) } ?: "—")
                    ReportRow("FAR المقاس", state.measuredFar?.let { Fmt.percent(it) } ?: "—")
                    ReportRow("أزواج الاختبار", state.sampleCount.toString())
                    ReportRow("تاريخ المعايرة", state.calibrationGeneratedAt)
                    Spacer(Modifier.height(4.dp))
                    Text(
                        state.policy.provenance.dataset,
                        style = MaterialTheme.typography.bodySmall,
                        color = AuroraColors.TextSecondary,
                    )
                } else {
                    Text(
                        "لم تُعاير العتبة على هذا الجهاز: تُستخدم قيمة OpenCV Zoo المرجعية (0.363).",
                        style = MaterialTheme.typography.bodySmall,
                        color = AuroraColors.Amber,
                    )
                }
            }

            Spacer(Modifier.height(14.dp))
            AuroraPanel(Modifier.fillMaxWidth(), accent = AuroraColors.Mint.copy(alpha = 0.35f)) {
                HudLabel("PRIVACY", color = AuroraColors.Mint)
                Spacer(Modifier.height(6.dp))
                listOf(
                    "كل عمليات التحليل (اكتشاف الوجه، التمثيل الرقمي، البحث) تُنفَّذ على الجهاز.",
                    "لا تُرسل أي صورة أو تمثيل رقمي إلى أي خادم. التطبيق لا يحتوي على مفاتيح API.",
                    "تُحفظ الصور المضافة في مساحة التطبيق الخاصة فقط، ويمكن حذفها في أي وقت.",
                    "التمثيلات الرقمية محفوظة مشفّرة (AES-256-GCM) بمفتاح داخل Android Keystore.",
                    "لا تُكتب بيانات الوجه في سجلات الأخطاء، ولا توجد أدوات تحليلات أو إعلانات.",
                    "النسخ الاحتياطي السحابي معطّل لهذا التطبيق لحماية البيانات البيومترية.",
                ).forEach {
                    Text("• $it", style = MaterialTheme.typography.bodySmall, color = AuroraColors.TextSecondary)
                    Spacer(Modifier.height(3.dp))
                }
            }

            Spacer(Modifier.height(14.dp))
            AuroraPanel(Modifier.fillMaxWidth()) {
                HudLabel("PERMISSIONS")
                Spacer(Modifier.height(6.dp))
                ReportRow("الكاميرا", "للتقاط صورة للتحليل فقط")
                ReportRow("الإنترنت", "غير مستخدم في التحليل؛ مذكور فقط للتنزيل الاختياري لاحقًا")
                ReportRow("المعرض", "عبر منتقي صور Android بدون صلاحية تخزين عامة")
            }

            Spacer(Modifier.height(14.dp))
            AuroraPanel(Modifier.fillMaxWidth(), accent = AuroraColors.Amber.copy(alpha = 0.35f)) {
                HudLabel("LIMITS", color = AuroraColors.Amber)
                Spacer(Modifier.height(6.dp))
                Text(
                    "درجة التشابه ليست احتمالًا مؤكدًا للهوية. النظام مخصص للبحث داخل مجموعة صور " +
                        "مصرح باستخدامها، ولا يجمع صورًا من الإنترنت ولا يحدد هويات مجهولة. " +
                        "الأداء المقاس على مجموعة LFW: AUC ${state.measuredAuc?.let { Fmt.similarity(it) } ?: "—"}, " +
                        "EER ${state.measuredEer?.let { Fmt.percent(it) } ?: "—"} (تفاصيل كاملة في docs/EVALUATION.md).",
                    style = MaterialTheme.typography.bodySmall,
                    color = AuroraColors.Amber,
                )
            }

            state.message?.let {
                Spacer(Modifier.height(12.dp))
                AuroraPanel(Modifier.fillMaxWidth(), accent = AuroraColors.Mint.copy(alpha = 0.4f)) {
                    Text(it, style = MaterialTheme.typography.bodySmall, color = AuroraColors.Mint)
                }
            }

            Spacer(Modifier.height(16.dp))
            SecondaryAction(
                text = "حذف كل بيانات التطبيق",
                accent = AuroraColors.Danger,
                onClick = { confirmDelete = true },
            )
            Spacer(Modifier.height(24.dp))
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("حذف كل البيانات؟") },
            text = {
                Text(
                    "سيتم حذف كل الصور المسجلة وتمثيلاتها المشفرة وكل سجلات البحث نهائيًا. " +
                        "لا يمكن التراجع عن هذه العملية.",
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    viewModel.deleteAllData()
                    confirmDelete = false
                }) { Text("حذف كل شيء", color = AuroraColors.Danger) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("إلغاء") } },
        )
    }
}

package com.auroravision.app.ui

import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.TimeUnit

/** Formatting helpers shared by the screens (Arabic locale, Western digits). */
object Fmt {

    private val dateTime = SimpleDateFormat("yyyy/MM/dd HH:mm", Locale.US)
    private val time = SimpleDateFormat("HH:mm", Locale.US)
    private val date = SimpleDateFormat("yyyy/MM/dd", Locale.US)

    fun dateTime(epochMs: Long): String = if (epochMs <= 0) "—" else dateTime.format(Date(epochMs))

    fun date(epochMs: Long): String = if (epochMs <= 0) "—" else date.format(Date(epochMs))

    fun time(epochMs: Long): String = if (epochMs <= 0) "—" else time.format(Date(epochMs))

    fun relative(epochMs: Long, now: Long = System.currentTimeMillis()): String {
        if (epochMs <= 0) return "—"
        val diff = (now - epochMs).coerceAtLeast(0)
        val minutes = TimeUnit.MILLISECONDS.toMinutes(diff)
        val hours = TimeUnit.MILLISECONDS.toHours(diff)
        val days = TimeUnit.MILLISECONDS.toDays(diff)
        return when {
            minutes < 1 -> "الآن"
            minutes < 60 -> "قبل $minutes دقيقة"
            hours < 24 -> "قبل $hours ساعة"
            days < 30 -> "قبل $days يوم"
            else -> date(epochMs)
        }
    }

    fun similarity(value: Float): String = String.format(Locale.US, "%.4f", value)

    fun percent(value: Float): String = String.format(Locale.US, "%.1f%%", value * 100f)

    fun duration(ms: Long): String = when {
        ms < 1000 -> "$ms ms"
        else -> String.format(Locale.US, "%.2f s", ms / 1000.0)
    }

    fun verdictTitle(verdict: String): String = when (verdict) {
        "CANDIDATES_FOUND" -> "مرشّحات تحتاج مراجعة بشرية"
        "NO_RELIABLE_MATCH" -> "لم يُعثر على تطابق موثوق"
        "NO_FACES_DETECTED" -> "لم يتم اكتشاف أي وجه"
        "QUALITY_REJECTED" -> "جودة الصورة لا تسمح بمقارنة موثوقة"
        "NO_REFERENCE_DATA" -> "قاعدة الصور فارغة أو لا تحتوي سجلات قابلة للمقارنة"
        "FAILED" -> "تعذر إكمال التحليل"
        else -> verdict
    }

    fun verdictExplanation(verdict: String, threshold: Float): String = when (verdict) {
        "CANDIDATES_FOUND" ->
            "تجاوزت النتائج أدناه عتبة التشابه ${similarity(threshold)}. هذا يعني تشابهًا بصريًا " +
                "يستحق المراجعة، وليس تأكيدًا لهوية الشخص."
        "NO_RELIABLE_MATCH" ->
            "أُجريت المقارنة مع كل السجلات المؤهلة، لكن أعلى درجة تشابه لم تبلغ العتبة " +
                "${similarity(threshold)}. تُعرض أفضل النتائج أدناه للشفافية مع درجاتها الفعلية."
        "NO_FACES_DETECTED" -> "لم يجد المحرك وجهًا في الصورة. جرّب صورة أوضح وبوجه كامل."
        "QUALITY_REJECTED" ->
            "لم تُنفَّذ المقارنة لأن الوجه لم يجتز فحص الجودة (ضبابية أو إضاءة أو حجم أو زاوية). " +
                "لم يتم اختراع أي نتيجة بدلًا من ذلك."
        "NO_REFERENCE_DATA" ->
            "أضف صورًا إلى قاعدة الصور المسجلة أولًا، ثم أعد التحليل."
        else -> "راجع التفاصيل أدناه."
    }

    fun bandLabel(band: String): String = when (band) {
        "MATCH_CANDIDATE" -> "مرشّح"
        "INCONCLUSIVE" -> "غير حاسم"
        else -> "ضعيف"
    }

    fun qualityLabel(status: String): String = when (status) {
        "OK" -> "جيدة"
        "WARNING" -> "مقبولة مع تحذير"
        "REJECTED" -> "غير صالحة للمقارنة"
        else -> status
    }
}

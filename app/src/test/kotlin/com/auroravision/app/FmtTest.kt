package com.auroravision.app

import com.auroravision.app.ui.Fmt
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class FmtTest {

    @Test
    fun `verdict titles never claim a confirmed identity`() {
        val verdicts = listOf(
            "CANDIDATES_FOUND", "NO_RELIABLE_MATCH", "NO_FACES_DETECTED",
            "QUALITY_REJECTED", "NO_REFERENCE_DATA", "FAILED",
        )
        verdicts.forEach { v ->
            val title = Fmt.verdictTitle(v)
            assertTrue("empty title for $v", title.isNotBlank())
            // The product may not present a similarity as proof of identity.
            assertFalse("title claims a confirmed match for $v: $title", title.contains("تطابق مؤكد"))
            assertFalse(title.contains("هوية مؤكدة"))
        }
    }

    @Test
    fun `candidate explanation states that similarity is not identity`() {
        val text = Fmt.verdictExplanation("CANDIDATES_FOUND", 0.3265f)
        assertTrue(text.contains("ليست"))
        assertTrue(text.contains("0.3265"))
    }

    @Test
    fun `no-match explanation mentions the threshold that was not reached`() {
        val text = Fmt.verdictExplanation("NO_RELIABLE_MATCH", 0.3265f)
        assertTrue(text.contains("0.3265"))
        assertTrue(text.contains("العتبة"))
    }

    @Test
    fun `quality rejected explanation says no comparison was invented`() {
        val text = Fmt.verdictExplanation("QUALITY_REJECTED", 0.3f)
        assertTrue(text.contains("لم تُنفَّذ المقارنة"))
    }

    @Test
    fun `similarity is formatted with four decimals`() {
        assertEquals("0.3265", Fmt.similarity(0.3265f))
        assertEquals("1.0000", Fmt.similarity(1f))
    }

    @Test
    fun `duration switches unit above one second`() {
        assertEquals("250 ms", Fmt.duration(250))
        assertEquals("1.50 s", Fmt.duration(1500))
    }

    @Test
    fun `relative time buckets are stable`() {
        val now = 1_700_000_000_000L
        assertEquals("الآن", Fmt.relative(now, now))
        assertEquals("قبل 5 دقيقة", Fmt.relative(now - 5 * 60_000, now))
        assertEquals("قبل 3 ساعة", Fmt.relative(now - 3 * 3_600_000, now))
        assertEquals("قبل 2 يوم", Fmt.relative(now - 2 * 86_400_000, now))
        assertEquals("—", Fmt.relative(0, now))
    }

    @Test
    fun `band and quality labels cover every enum name the engine emits`() {
        assertTrue(Fmt.bandLabel("MATCH_CANDIDATE").isNotBlank())
        assertTrue(Fmt.bandLabel("INCONCLUSIVE").isNotBlank())
        assertTrue(Fmt.bandLabel("DISSIMILAR").isNotBlank())
        assertEquals("جيدة", Fmt.qualityLabel("OK"))
        assertEquals("غير صالحة للمقارنة", Fmt.qualityLabel("REJECTED"))
    }
}

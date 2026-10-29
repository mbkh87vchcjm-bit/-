package com.auroravision.core.search

import com.auroravision.core.model.FaceEmbedding
import com.auroravision.core.pipeline.MatchVerdict
import com.auroravision.core.pipeline.SimilarityBand
import com.auroravision.core.policy.MatchPolicy
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt

class SearchServiceTest {

    private val policy = MatchPolicy(
        operatingThreshold = 0.60f,
        inconclusiveFloor = 0.25f,
        defaultTopK = 3,
        maxTopK = 10,
    )
    private val service = SearchService(policy)
    private val modelKey = "sface/test"

    /** Unit vector along axis [axis] scaled to make cosine similarities predictable. */
    private fun vector(vararg values: Float): FaceEmbedding {
        val v = FloatArray(FaceEmbedding.DIMENSION)
        values.forEachIndexed { i, f -> v[i] = f }
        var norm = 0.0
        for (x in v) norm += x.toDouble() * x
        val n = sqrt(norm).toFloat()
        for (i in v.indices) v[i] /= n
        return FaceEmbedding.fromStored(v)
    }

    private fun indexed(id: Long, emb: FaceEmbedding, key: String = modelKey) =
        IndexedFace(id, emb, key)

    @Test
    fun `ranks by descending similarity and assigns bands`() {
        val query = vector(1f, 0f, 0f)
        val corpus = listOf(
            indexed(1, vector(1f, 0f, 0f)),      // cos = 1.0  -> candidate
            indexed(2, vector(0.8f, 0.6f, 0f)),  // cos = 0.8  -> candidate
            indexed(3, vector(0.4f, 0.9f, 0f)),  // cos ≈ 0.41 -> inconclusive
            indexed(4, vector(0f, 1f, 0f)),      // cos = 0    -> dissimilar
        )
        val out = service.search(query, corpus, modelKey, topK = 4)
        assertEquals(4, out.comparedCount)
        assertEquals(listOf(1L, 2L, 3L, 4L), out.hits.map { it.recordId })
        assertEquals(SimilarityBand.MATCH_CANDIDATE, out.hits[0].band)
        assertEquals(SimilarityBand.MATCH_CANDIDATE, out.hits[1].band)
        assertEquals(SimilarityBand.INCONCLUSIVE, out.hits[2].band)
        assertEquals(SimilarityBand.DISSIMILAR, out.hits[3].band)
        assertEquals(MatchVerdict.CANDIDATES_FOUND, out.verdict)
        assertEquals(1, out.hits[0].rank)
        assertTrue(out.completed)
    }

    @Test
    fun `reports no reliable match instead of inventing one`() {
        val query = vector(1f, 0f, 0f)
        val corpus = listOf(
            indexed(1, vector(0.3f, 0.95f, 0f)), // cos ≈ 0.30 -> inconclusive
            indexed(2, vector(0f, 1f, 0f)),      // cos = 0
        )
        val out = service.search(query, corpus, modelKey, topK = 5)
        assertEquals(MatchVerdict.NO_RELIABLE_MATCH, out.verdict)
        assertEquals(2, out.hits.size) // weak results are shown, not silently dropped
        assertTrue(out.hits.all { it.band != SimilarityBand.MATCH_CANDIDATE })
    }

    @Test
    fun `skips embeddings produced by a different model and reports the count`() {
        val query = vector(1f, 0f, 0f)
        val corpus = listOf(
            indexed(1, vector(1f, 0f, 0f), key = "other-model/2"),
            indexed(2, vector(1f, 0f, 0f), key = modelKey),
        )
        val out = service.search(query, corpus, modelKey, topK = 5)
        assertEquals(1, out.comparedCount)
        assertEquals(1, out.skippedModelMismatch)
        assertEquals(listOf(2L), out.hits.map { it.recordId })
    }

    @Test
    fun `empty corpus yields NO_REFERENCE_DATA`() {
        val out = service.search(vector(1f), emptyList(), modelKey, topK = 5)
        assertEquals(MatchVerdict.NO_REFERENCE_DATA, out.verdict)
        assertEquals(0, out.comparedCount)
        assertTrue(out.hits.isEmpty())
        assertTrue(out.completed)
    }

    @Test
    fun `topK is clamped to the policy maximum and to the corpus size`() {
        val corpus = (1L..20L).map { indexed(it, vector(1f, 0f, 0f)) }
        val out = service.search(vector(1f, 0f, 0f), corpus, modelKey, topK = 99)
        assertEquals(10, out.hits.size) // maxTopK
        assertEquals(20, out.comparedCount)
    }

    @Test
    fun `ties are broken deterministically by record id`() {
        val corpus = listOf(
            indexed(5, vector(1f, 0f, 0f)),
            indexed(2, vector(1f, 0f, 0f)),
            indexed(9, vector(1f, 0f, 0f)),
        )
        val out = service.search(vector(1f, 0f, 0f), corpus, modelKey, topK = 3)
        assertEquals(listOf(2L, 5L, 9L), out.hits.map { it.recordId })
    }

    @Test
    fun `quality-rejected records are never compared`() {
        val corpus = listOf(
            IndexedFace(1, vector(1f, 0f, 0f), modelKey, qualityComparable = false),
            indexed(2, vector(0.2f, 0.9f, 0f)),
        )
        val out = service.search(vector(1f, 0f, 0f), corpus, modelKey, topK = 5)
        assertEquals(1, out.comparedCount)
        assertEquals(1, out.skippedQualityRejected)
        assertEquals(listOf(2L), out.hits.map { it.recordId })
    }

    @Test
    fun `policy rejects an incoherent configuration`() {
        assertThrows(IllegalArgumentException::class.java) {
            MatchPolicy(operatingThreshold = 0.2f, inconclusiveFloor = 0.5f)
        }
    }

    @Test
    fun `flat index search over 20000 records stays fast`() {
        val rnd = java.util.Random(42)
        val corpus = (1L..20_000L).map {
            val v = FloatArray(FaceEmbedding.DIMENSION) { rnd.nextFloat() - 0.5f }
            indexed(it, FaceEmbedding.fromRaw(v))
        }
        val index = EmbeddingIndex.build(modelKey, corpus)
        val query = FaceEmbedding.fromRaw(FloatArray(FaceEmbedding.DIMENSION) { 0.1f })
        // warm up the JIT
        repeat(3) { index.search(query.asList().toFloatArray(), 5) }
        val t0 = System.nanoTime()
        val hits = index.search(query.asList().toFloatArray(), 5)
        val ms = (System.nanoTime() - t0) / 1_000_000.0
        assertEquals(5, hits.size)
        assertTrue("search over 20k records took ${ms}ms", ms < 500.0)
        println("20k-record exhaustive search: %.2f ms".format(ms))
    }
}

package com.auroravision.core.search

import com.auroravision.core.model.FaceEmbedding
import com.auroravision.core.pipeline.MatchVerdict
import com.auroravision.core.pipeline.RankedMatch
import com.auroravision.core.pipeline.SearchOutcome
import com.auroravision.core.pipeline.SimilarityBand
import com.auroravision.core.policy.MatchPolicy
import kotlin.system.measureTimeMillis

/**
 * Phase 05/06 — compares a query embedding against every eligible registered
 * face and ranks the results.
 *
 * Rules that matter for honest reporting:
 *  * records produced by a *different* model are not compared (their vectors
 *    live in an incompatible space) and their count is reported;
 *  * nothing below the threshold is silently dropped — [SearchOutcome.hits]
 *    always contains the top-K with their real similarity and band, and the
 *    verdict says plainly when no candidate reached the operating point.
 */
class SearchService(private val policy: MatchPolicy) {

    fun band(similarity: Float): SimilarityBand = when {
        similarity >= policy.operatingThreshold -> SimilarityBand.MATCH_CANDIDATE
        similarity >= policy.inconclusiveFloor -> SimilarityBand.INCONCLUSIVE
        else -> SimilarityBand.DISSIMILAR
    }

    fun search(
        query: FaceEmbedding,
        corpus: List<IndexedFace>,
        modelKey: String,
        topK: Int = policy.defaultTopK,
        index: EmbeddingIndex? = null,
    ): SearchOutcome {
        val eligible = if (index != null && index.modelKey == modelKey) {
            null // already filtered by the caller
        } else {
            corpus.filter { it.modelKey == modelKey }
        }
        val skippedModel = corpus.size - (eligible?.size ?: corpus.size)
        val skippedQuality = (eligible ?: corpus).count { !it.qualityComparable }
        val comparableCount = (eligible ?: corpus).count { it.qualityComparable }

        val idx = index ?: EmbeddingIndex.build(modelKey, eligible ?: corpus)
        val effectiveTopK = topK.coerceIn(1, maxOf(1, minOf(policy.maxTopK, comparableCount.coerceAtLeast(1))))

        var hits: List<RankedMatch> = emptyList()
        var best: Float? = null
        val elapsed = measureTimeMillis {
            val raw = idx.search(query.asList().toFloatArray(), effectiveTopK)
            hits = raw.mapIndexed { rank, hit ->
                RankedMatch(
                    recordId = hit.recordId,
                    similarity = hit.similarity,
                    rank = rank + 1,
                    band = band(hit.similarity),
                )
            }
            best = hits.firstOrNull()?.similarity
        }

        val verdict = when {
            comparableCount == 0 -> MatchVerdict.NO_REFERENCE_DATA
            hits.any { it.band == SimilarityBand.MATCH_CANDIDATE } -> MatchVerdict.CANDIDATES_FOUND
            else -> MatchVerdict.NO_RELIABLE_MATCH
        }

        return SearchOutcome(
            comparedCount = comparableCount,
            skippedModelMismatch = skippedModel,
            skippedQualityRejected = skippedQuality,
            totalRegistered = corpus.size,
            hits = hits,
            bestSimilarity = best,
            verdict = verdict,
            durationMs = elapsed,
            threshold = policy.operatingThreshold,
            topK = effectiveTopK,
            completed = true,
        )
    }
}

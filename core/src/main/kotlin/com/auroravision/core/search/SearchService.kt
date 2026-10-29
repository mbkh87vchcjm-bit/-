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
        val idx: EmbeddingIndex
        val skippedModel: Int
        val skippedQuality: Int
        val comparableCount: Int
        val totalRegistered: Int

        if (index != null && index.modelKey == modelKey) {
            // Pre-built (cached) index: its entries are already filtered by model
            // key, so the counts must be read from the index itself — reporting 0
            // here would make a non-empty database look empty.
            idx = index
            skippedModel = 0
            skippedQuality = index.count - index.comparableCount
            comparableCount = index.comparableCount
            totalRegistered = index.count
        } else {
            val eligible = corpus.filter { it.modelKey == modelKey }
            skippedModel = corpus.size - eligible.size
            skippedQuality = eligible.count { !it.qualityComparable }
            comparableCount = eligible.count { it.qualityComparable }
            totalRegistered = corpus.size
            idx = EmbeddingIndex.build(modelKey, eligible)
        }

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
            totalRegistered = totalRegistered,
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

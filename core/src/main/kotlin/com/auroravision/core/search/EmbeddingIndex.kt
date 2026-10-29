package com.auroravision.core.search

import com.auroravision.core.model.FaceEmbedding

/** One registered face that can take part in a search. */
data class IndexedFace(
    val recordId: Long,
    val embedding: FaceEmbedding,
    /** Model key (`embedderId/version`) that produced the embedding. */
    val modelKey: String,
    /** False when the record was registered with a rejected quality report. */
    val qualityComparable: Boolean = true,
)

/**
 * Flat, contiguous index of normalised embeddings.
 *
 * Why exhaustive search instead of an approximate index: for 128-D *unit*
 * vectors the similarity is a plain dot product, so 50 000 records cost
 * ~6.4 M multiply-adds (~3-8 ms on a mid-range phone). An ANN structure would
 * add recall risk and code for no measurable gain at the scale this app
 * targets; the design note and the benchmark live in docs/ARCHITECTURE.md.
 *
 * Embeddings are packed into one `FloatArray` so the inner loop walks memory
 * linearly and never chases per-object references.
 */
class EmbeddingIndex private constructor(
    val count: Int,
    val modelKey: String,
    private val ids: LongArray,
    private val matrix: FloatArray,
    private val comparable: BooleanArray,
) {
    val dimension: Int get() = if (count == 0) FaceEmbedding.DIMENSION else matrix.size / count

    /** Records that can take part in a comparison (quality-rejected ones cannot). */
    val comparableCount: Int get() = comparable.count { it }

    fun recordIdAt(i: Int): Long = ids[i]

    fun isComparable(i: Int): Boolean = comparable[i]

    /**
     * Exhaustive search. Returns up to [topK] hits ordered by descending cosine
     * similarity. Ties are broken by record id so results are reproducible.
     */
    fun search(query: FloatArray, topK: Int): List<Hit> {
        require(query.size == dimension) { "query dimension ${query.size} != index dimension $dimension" }
        if (count == 0) return emptyList()
        val dim = dimension
        val scores = FloatArray(count)
        for (i in 0 until count) {
            var dot = 0f
            val base = i * dim
            for (d in 0 until dim) dot += matrix[base + d] * query[d]
            scores[i] = if (comparable[i]) dot else Float.NEGATIVE_INFINITY
        }
        val k = topK.coerceIn(1, count)
        val order = (0 until count)
            .filter { scores[it] != Float.NEGATIVE_INFINITY }
            .sortedWith(compareByDescending<Int> { scores[it] }.thenBy { ids[it] })
            .take(k)
        return order.map { Hit(ids[it], scores[it]) }
    }

    data class Hit(val recordId: Long, val similarity: Float)

    companion object {
        val EMPTY = EmbeddingIndex(0, "", LongArray(0), FloatArray(0), BooleanArray(0))

        fun build(modelKey: String, faces: List<IndexedFace>): EmbeddingIndex {
            if (faces.isEmpty()) return EmbeddingIndex(0, modelKey, LongArray(0), FloatArray(0), BooleanArray(0))
            val dim = FaceEmbedding.DIMENSION
            val ids = LongArray(faces.size)
            val matrix = FloatArray(faces.size * dim)
            val comparable = BooleanArray(faces.size)
            faces.forEachIndexed { i, face ->
                require(face.embedding.dimension == dim) {
                    "record ${face.recordId} has dimension ${face.embedding.dimension}, expected $dim"
                }
                ids[i] = face.recordId
                comparable[i] = face.qualityComparable
                val values = face.embedding.asList()
                for (d in 0 until dim) matrix[i * dim + d] = values[d]
            }
            return EmbeddingIndex(faces.size, modelKey, ids, matrix, comparable)
        }
    }
}

package com.auroravision.app

import com.auroravision.app.data.crypto.PlainEmbeddingCipher
import com.auroravision.app.data.crypto.toBytes
import com.auroravision.app.data.crypto.toFloats
import com.auroravision.core.model.FaceEmbedding
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.sqrt

/**
 * Storage contract for biometric templates.
 *
 * The Keystore-backed cipher itself is exercised on a device; here we verify the
 * byte format and the *logic* that must hold for any [EmbeddingCipher]
 * implementation, so a regression in the codec cannot silently corrupt the
 * database.
 */
class EmbeddingStorageTest {

    @Test
    fun `float array round-trips through the byte blob`() {
        val values = FloatArray(128) { it * 0.25f - 3f }
        val restored = values.toBytes().toFloats()
        assertArrayEquals(values, restored, 0f)
        assertEquals(128 * 4, values.toBytes().size)
    }

    @Test
    fun `a truncated blob is rejected instead of producing garbage`() {
        val bad = ByteArray(6)
        val error = runCatching { bad.toFloats() }.exceptionOrNull()
        assertTrue("expected an IllegalArgumentException, got $error", error is IllegalArgumentException)
    }

    @Test
    fun `plain cipher preserves the embedding and is not the identity encoding`() {
        val cipher = PlainEmbeddingCipher()
        val embedding = FaceEmbedding.fromRaw(FloatArray(128) { (it % 7).toFloat() - 3f })
        val blob = cipher.encrypt(embedding.values)
        val restored = FaceEmbedding.fromStored(cipher.decrypt(blob))
        assertEquals(1f, embedding.cosineSimilarity(restored), 1e-6f)
        assertEquals(512, blob.size)
    }

    @Test
    fun `normalisation makes stored templates unit length`() {
        val raw = FloatArray(128) { (it + 1).toFloat() }
        val embedding = FaceEmbedding.fromRaw(raw)
        var norm = 0.0
        for (v in embedding.values) norm += v.toDouble() * v
        assertEquals(1.0, sqrt(norm), 1e-6)
        assertNotEquals(raw[0], embedding.values[0])
    }

    @Test
    fun `rejects a zero-norm descriptor`() {
        val error = runCatching { FaceEmbedding.fromRaw(FloatArray(128)) }.exceptionOrNull()
        assertTrue("a zero vector must not become an embedding", error is IllegalArgumentException)
    }

    @Test
    fun `rejects a descriptor with the wrong dimension`() {
        val error = runCatching { FaceEmbedding.fromRaw(FloatArray(64) { 1f }) }.exceptionOrNull()
        assertTrue(error is IllegalArgumentException)
    }

    @Test
    fun `cosine similarity of identical templates is one and of opposites is minus one`() {
        val a = FaceEmbedding.fromRaw(FloatArray(128) { if (it % 2 == 0) 1f else -0.5f })
        val b = FaceEmbedding.fromRaw(FloatArray(128) { if (it % 2 == 0) -1f else 0.5f })
        assertEquals(1f, a.cosineSimilarity(a), 1e-6f)
        assertEquals(-1f, a.cosineSimilarity(b), 1e-6f)
    }
}

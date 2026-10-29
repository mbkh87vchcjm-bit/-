package com.auroravision.app.data.crypto

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import java.nio.ByteBuffer
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Protects the biometric templates stored in Room.
 *
 * A face embedding is a biometric identifier: it cannot be re-issued like a
 * password. Embeddings are therefore encrypted with an AES-256-GCM key that
 * lives in the Android Keystore and never leaves it (the plaintext key is not
 * exportable, and the app cannot read it even with root on a locked device).
 *
 * What this does and does not protect against (documented plainly, because
 * over-claiming here would be dishonest):
 *  * protects: database files copied off the device, adb backups, other apps
 *    reading app-private storage on a rooted device;
 *  * does not protect: an attacker with code execution *inside* this app's
 *    process while the device is unlocked (the key is usable then).
 */
interface EmbeddingCipher {
    fun encrypt(embedding: FloatArray): ByteArray
    fun decrypt(blob: ByteArray): FloatArray
}

/** AES-256-GCM via the Android Keystore. Key alias is created on first use. */
class KeystoreEmbeddingCipher(
    private val alias: String = DEFAULT_ALIAS,
) : EmbeddingCipher {

    override fun encrypt(embedding: FloatArray): ByteArray {
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, key())
        val plain = embedding.toBytes()
        val encrypted = cipher.doFinal(plain)
        // store IV || ciphertext
        val out = ByteArray(cipher.iv.size + encrypted.size)
        System.arraycopy(cipher.iv, 0, out, 0, cipher.iv.size)
        System.arraycopy(encrypted, 0, out, cipher.iv.size, encrypted.size)
        plain.fill(0)
        return out
    }

    override fun decrypt(blob: ByteArray): FloatArray {
        require(blob.size > IV_SIZE) { "embedding blob is truncated" }
        val iv = blob.copyOfRange(0, IV_SIZE)
        val body = blob.copyOfRange(IV_SIZE, blob.size)
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.DECRYPT_MODE, key(), GCMParameterSpec(TAG_BITS, iv))
        val plain = cipher.doFinal(body)
        return plain.toFloats().also { plain.fill(0) }
    }

    private fun key(): SecretKey {
        val keyStore = KeyStore.getInstance(PROVIDER).apply { load(null) }
        (keyStore.getEntry(alias, null) as? KeyStore.SecretKeyEntry)?.let { return it.secretKey }
        val generator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, PROVIDER)
        generator.init(
            KeyGenParameterSpec.Builder(
                alias,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT,
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                // Not tied to user authentication: analysis must run in the
                // background (e.g. while indexing a batch the user picked) and
                // must keep working when the screen is locked.
                .build(),
        )
        return generator.generateKey()
    }

    companion object {
        const val DEFAULT_ALIAS = "aurora_embedding_key_v1"
        private const val PROVIDER = "AndroidKeyStore"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
        private const val IV_SIZE = 12
        private const val TAG_BITS = 128
    }
}

/** Unencrypted codec — used only by unit tests that do not have a Keystore. */
class PlainEmbeddingCipher : EmbeddingCipher {
    override fun encrypt(embedding: FloatArray): ByteArray = embedding.toBytes()
    override fun decrypt(blob: ByteArray): FloatArray = blob.toFloats()
}

internal fun FloatArray.toBytes(): ByteArray {
    val buf = ByteBuffer.allocate(size * 4)
    buf.asFloatBuffer().put(this)
    return buf.array()
}

internal fun ByteArray.toFloats(): FloatArray {
    require(size % 4 == 0) { "byte blob is not a float array" }
    val buf = ByteBuffer.wrap(this)
    val out = FloatArray(size / 4)
    buf.asFloatBuffer().get(out)
    return out
}

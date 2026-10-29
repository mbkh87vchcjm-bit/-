package com.auroravision.core.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer

/** A named float tensor returned by a model. */
class TensorOutput(val name: String, val dims: LongArray, val data: FloatArray) {
    val elementCount: Int get() = data.size
}

/**
 * Minimal, thread-safe wrapper around an ONNX Runtime session.
 *
 * The same class is used on Android (`onnxruntime-android`) and in JVM unit
 * tests (`onnxruntime`), because both artifacts expose the identical
 * `ai.onnxruntime` API. That is what allows the *real* models to be verified by
 * JVM tests instead of only on a device.
 */
class OnnxSession private constructor(
    private val session: OrtSession,
    val inputName: String,
    val outputNames: List<String>,
    val inputShape: LongArray,
) : AutoCloseable {

    private val lock = Any()

    fun run(input: FloatArray, shape: LongArray): List<TensorOutput> {
        val expected = shape.fold(1L) { acc, d -> acc * d }
        require(input.size.toLong() == expected) {
            "input has ${input.size} values but shape ${shape.toList()} needs $expected"
        }
        synchronized(lock) {
            val env = OrtEnvironment.getEnvironment()
            OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape).use { tensor ->
                session.run(mapOf(inputName to tensor)).use { result ->
                    // Iterate the result instead of index-based lookups: this keeps
                    // the code independent of Optional-handling differences across
                    // ONNX Runtime releases.
                    val byName = HashMap<String, OnnxTensor>(outputNames.size)
                    for (entry in result) {
                        val value = entry.value
                        if (value is OnnxTensor) byName[entry.key] = value
                    }
                    val out = ArrayList<TensorOutput>(outputNames.size)
                    for (name in outputNames) {
                        val t = byName[name]
                            ?: throw IllegalStateException("model produced no tensor output named '$name'")
                        // ONNX Runtime returns java.nio.FloatBuffer here (not Optional);
                        // a null means the tensor is not a float tensor.
                        val buffer = t.floatBuffer
                        if (buffer == null) {
                            throw IllegalStateException("output '$name' is not a float tensor")
                        }
                        val arr = FloatArray(buffer.remaining())
                        buffer.get(arr)
                        out.add(TensorOutput(name, t.info.shape.clone(), arr))
                    }
                    return out
                }
            }
        }
    }

    override fun close() {
        synchronized(lock) { session.close() }
    }

    companion object {
        /**
         * Loads a model from memory (assets are read into a ByteArray by the
         * caller: this keeps the core free of Android asset APIs).
         *
         * @param threads number of intra-op threads; 2 is a good compromise on
         *   mid-range phones (more threads add wake-up latency on small models).
         */
        fun load(
            modelBytes: ByteArray,
            threads: Int = 2,
            threadsExplicitlySet: Boolean = true,
        ): OnnxSession {
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions()
            opts.setIntraOpNumThreads(threads.coerceAtLeast(1))
            if (threadsExplicitlySet) {
                opts.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            }
            val session = env.createSession(modelBytes, opts)
            return OnnxSession(
                session = session,
                inputName = session.inputNames.first(),
                outputNames = session.outputNames.toList(),
                inputShape = LongArray(0),
            )
        }
    }
}

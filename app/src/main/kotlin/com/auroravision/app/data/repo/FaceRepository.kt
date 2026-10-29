package com.auroravision.app.data.repo

import android.content.Context
import android.net.Uri
import com.auroravision.app.data.ImageStore
import com.auroravision.app.data.crypto.EmbeddingCipher
import com.auroravision.app.data.db.FaceRecordDao
import com.auroravision.app.data.db.FaceRecordEntity
import com.auroravision.app.engine.EngineState
import com.auroravision.app.engine.FaceEngineHolder
import com.auroravision.core.model.FaceEmbedding
import com.auroravision.core.policy.ModelInfo
import com.auroravision.core.quality.QualityAnalyzer
import com.auroravision.core.search.EmbeddingIndex
import com.auroravision.core.search.IndexedFace
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext
import java.util.UUID

/** Outcome of registering one or more images in the reference database. */
data class RegistrationReport(
    val imagesAccepted: Int,
    val imagesDuplicate: Int,
    val imagesUnreadable: Int,
    val facesStored: Int,
    val facesRejected: Int,
    val facesSkippedBecauseDuplicate: Int,
    val failures: List<String>,
    val durationMs: Long,
) {
    val addedAnything: Boolean get() = facesStored > 0
}

/** Per-image progress while registering a batch. */
data class RegistrationProgress(
    val current: Int,
    val total: Int,
    val fileName: String,
    val facesFound: Int,
)

/** Result of re-running the whole pipeline over the stored images. */
data class ReindexReport(
    val imagesProcessed: Int,
    val recordsUpdated: Int,
    val recordsRemoved: Int,
    val failures: List<String>,
    val durationMs: Long,
)

/**
 * The reference-image database ("قاعدة الصور المسجلة").
 *
 * Registration is a *real* pipeline run per image: decode → validate → detect →
 * quality gate → embed → persist. Duplicate detection uses the SHA-256 of the
 * source file, and per-face uniqueness uses (sha, faceIndex).
 */
class FaceRepository(
    private val context: Context,
    private val dao: FaceRecordDao,
    private val imageStore: ImageStore,
    private val cipher: EmbeddingCipher,
    private val engine: FaceEngineHolder,
    private val models: ModelInfo = ModelInfo.DEFAULT,
) {

    fun observeRecords(): Flow<List<FaceRecordEntity>> = dao.observeAll()

    fun observeFaceCount(): Flow<Int> = dao.observeCount()

    fun observeImageCount(): Flow<Int> = dao.observeImageCount()

    suspend fun comparableCount(): Int = dao.comparableCount(models.modelKey)

    suspend fun findByIds(ids: List<Long>): List<FaceRecordEntity> = dao.findByIds(ids)

    /**
     * Registers images from the photo picker.
     *
     * @param displayName optional label applied to every face of these images.
     */
    suspend fun registerUris(
        uris: List<Uri>,
        sourceLabel: String,
        displayName: String? = null,
        onProgress: (RegistrationProgress) -> Unit = {},
    ): RegistrationReport = registerImages(
        items = uris.map { uri ->
            RegistrationInput(
                name = uri.lastPathSegment?.substringAfterLast('/') ?: "image",
                sourceLabel = sourceLabel,
                displayName = displayName,
                loader = { imageStore.decode(uri) },
            )
        },
        onProgress = onProgress,
    )

    suspend fun registerFile(
        file: java.io.File,
        sourceLabel: String,
        displayName: String? = null,
        onProgress: (RegistrationProgress) -> Unit = {},
    ): RegistrationReport = registerImages(
        items = listOf(
            RegistrationInput(file.name, sourceLabel, displayName) { imageStore.decode(file) },
        ),
        onProgress = onProgress,
    )

    private class RegistrationInput(
        val name: String,
        val sourceLabel: String,
        val displayName: String?,
        val loader: suspend () -> com.auroravision.app.data.DecodedImage,
    )

    private suspend fun registerImages(
        items: List<RegistrationInput>,
        onProgress: (RegistrationProgress) -> Unit,
    ): RegistrationReport = withContext(Dispatchers.Default) {
        val started = System.currentTimeMillis()
        if (engine.ensureLoaded() !is EngineState.Ready) {
            return@withContext RegistrationReport(
                0, 0, items.size, 0, 0, 0,
                listOf("محرك الذكاء الاصطناعي غير متاح — تعذر تحميل النماذج"),
                System.currentTimeMillis() - started,
            )
        }
        val detector = engine.requireDetector()
        val embedder = engine.requireEmbedder()
        val qualityPolicy = engine.loadQualityPolicy()

        var accepted = 0
        var duplicates = 0
        var unreadable = 0
        var stored = 0
        var rejected = 0
        var dupFaces = 0
        val failures = ArrayList<String>()

        items.forEachIndexed { index, item ->
            onProgress(RegistrationProgress(index + 1, items.size, item.name, stored))
            val decoded = try {
                item.loader()
            } catch (e: Exception) {
                unreadable++
                failures.add("${item.name}: تعذر فتح الصورة (${e.message ?: "خطأ غير معروف"})")
                return@forEachIndexed
            }
            if (dao.imagesWithSha(decoded.sha256) > 0) {
                duplicates++
                dupFaces++
                return@forEachIndexed
            }

            val detections = detector.detect(decoded.rgb)
            if (detections.isEmpty()) {
                failures.add("${item.name}: لم يتم اكتشاف أي وجه")
                return@forEachIndexed
            }
            val groupId = UUID.randomUUID().toString()
            var imagePath: String? = null
            var thumbPath: String? = null
            val rows = ArrayList<FaceRecordEntity>(detections.size)

            for (det in detections) {
                val aligned = embedder.align(decoded.rgb, det.landmarks)
                val quality = QualityAnalyzer.analyze(
                    aligned, det.landmarks, det.box.width, det.box.height,
                    decoded.width, decoded.height, qualityPolicy,
                )
                if (!quality.comparable) {
                    rejected++
                    continue
                }
                val embedding = embedder.embed(decoded.rgb, det)
                if (imagePath == null) {
                    imagePath = imageStore.saveRegisteredImage(decoded)
                    thumbPath = imageStore.saveThumbnail(decoded)
                }
                rows.add(
                    FaceRecordEntity(
                        groupId = groupId,
                        displayName = item.displayName,
                        sourceLabel = item.sourceLabel,
                        sourceDetail = item.name,
                        addedAtEpochMs = System.currentTimeMillis(),
                        imagePath = imagePath!!,
                        thumbnailPath = thumbPath,
                        imageWidth = decoded.width,
                        imageHeight = decoded.height,
                        faceIndex = det.index,
                        boxX = det.box.x,
                        boxY = det.box.y,
                        boxWidth = det.box.width,
                        boxHeight = det.box.height,
                        landmarks = det.landmarks.points.joinToString(";") { "${it.x},${it.y}" },
                        embeddingEncrypted = cipher.encrypt(embedding.values),
                        modelKey = models.modelKey,
                        qualityStatus = quality.status.name,
                        qualityScore = quality.score,
                        sharpness = quality.sharpness,
                        brightness = quality.brightness,
                        imageSha256 = decoded.sha256,
                    ),
                )
            }
            if (rows.isEmpty()) {
                failures.add("${item.name}: الوجوه المكتشفة لا تستوفي شروط الجودة")
                imageStore.deleteFiles(listOf(imagePath, thumbPath))
                return@forEachIndexed
            }
            dao.insertAll(rows)
            accepted++
            stored += rows.size
        }
        RegistrationReport(
            imagesAccepted = accepted,
            imagesDuplicate = duplicates,
            imagesUnreadable = unreadable,
            facesStored = stored,
            facesRejected = rejected,
            facesSkippedBecauseDuplicate = dupFaces,
            failures = failures,
            durationMs = System.currentTimeMillis() - started,
        )
    }

    /** Deletes every record (and file) belonging to one registered image. */
    suspend fun deleteGroup(groupId: String): Boolean = withContext(Dispatchers.IO) {
        val records = dao.observeAll().let { _ -> dao.all().filter { it.groupId == groupId } }
        if (records.isEmpty()) return@withContext false
        dao.deleteGroup(groupId)
        imageStore.deleteFiles(records.flatMap { listOf(it.imagePath, it.thumbnailPath) }.distinct())
        true
    }

    suspend fun deleteRecord(id: Long): Boolean = withContext(Dispatchers.IO) {
        val record = dao.findById(id) ?: return@withContext false
        dao.deleteById(id)
        val remaining = dao.all().count { it.groupId == record.groupId }
        if (remaining == 0) {
            imageStore.deleteFiles(listOf(record.imagePath, record.thumbnailPath))
        }
        true
    }

    suspend fun deleteAll(): Int = withContext(Dispatchers.IO) {
        val all = dao.all()
        val deleted = dao.deleteAll()
        imageStore.deleteFiles(all.flatMap { listOf(it.imagePath, it.thumbnailPath) }.distinct())
        deleted
    }

    /**
     * Re-runs detection + embedding for every stored image.
     *
     * Needed after a model change (embeddings of different models live in
     * incompatible spaces) and useful after a quality-policy update.
     */
    suspend fun reindexAll(
        onProgress: (RegistrationProgress) -> Unit = {},
    ): ReindexReport = withContext(Dispatchers.Default) {
        val started = System.currentTimeMillis()
        if (engine.ensureLoaded() !is EngineState.Ready) {
            return@withContext ReindexReport(0, 0, 0, listOf("engine unavailable"), 0)
        }
        val detector = engine.requireDetector()
        val embedder = engine.requireEmbedder()
        val qualityPolicy = engine.loadQualityPolicy()
        val groups = dao.all().groupBy { it.groupId }
        var processed = 0
        var updated = 0
        var removed = 0
        val failures = ArrayList<String>()

        groups.entries.forEachIndexed { i, (groupId, records) ->
            val sample = records.first()
            onProgress(RegistrationProgress(i + 1, groups.size, sample.sourceDetail ?: groupId, updated))
            val decoded = try {
                imageStore.decode(java.io.File(sample.imagePath))
            } catch (e: Exception) {
                failures.add("${sample.sourceDetail ?: groupId}: تعذرت قراءة الملف المحفوظ")
                return@forEachIndexed
            }
            val detections = detector.detect(decoded.rgb)
            val newRows = ArrayList<FaceRecordEntity>(detections.size)
            for (det in detections) {
                val aligned = embedder.align(decoded.rgb, det.landmarks)
                val quality = QualityAnalyzer.analyze(
                    aligned, det.landmarks, det.box.width, det.box.height,
                    decoded.width, decoded.height, qualityPolicy,
                )
                if (!quality.comparable) continue
                val embedding = embedder.embed(decoded.rgb, det)
                newRows.add(
                    sample.copy(
                        id = 0,
                        faceIndex = det.index,
                        boxX = det.box.x,
                        boxY = det.box.y,
                        boxWidth = det.box.width,
                        boxHeight = det.box.height,
                        landmarks = det.landmarks.points.joinToString(";") { "${it.x},${it.y}" },
                        embeddingEncrypted = cipher.encrypt(embedding.values),
                        modelKey = models.modelKey,
                        qualityStatus = quality.status.name,
                        qualityScore = quality.score,
                        sharpness = quality.sharpness,
                        brightness = quality.brightness,
                        imageWidth = decoded.width,
                        imageHeight = decoded.height,
                    ),
                )
            }
            dao.deleteGroup(groupId)
            removed += records.size
            if (newRows.isEmpty()) {
                failures.add("${sample.sourceDetail ?: groupId}: لا توجد وجوه صالحة بعد إعادة الفهرسة")
            } else {
                dao.insertAll(newRows)
                updated += newRows.size
            }
            processed++
        }
        ReindexReport(processed, updated, removed, failures, System.currentTimeMillis() - started)
    }

    /**
     * Builds the searchable index: decrypts templates, drops incompatible
     * models and quality-rejected records (already excluded at insert time).
     */
    suspend fun buildIndex(): EmbeddingIndex = withContext(Dispatchers.Default) {
        val rows = dao.comparableFor(models.modelKey)
        val faces = rows.mapNotNull { row ->
            runCatching {
                IndexedFace(
                    recordId = row.id,
                    embedding = FaceEmbedding.fromStored(cipher.decrypt(row.embeddingEncrypted)),
                    modelKey = row.modelKey,
                    qualityComparable = row.qualityStatus != "REJECTED",
                )
            }.getOrNull()
        }
        EmbeddingIndex.build(models.modelKey, faces)
    }
}

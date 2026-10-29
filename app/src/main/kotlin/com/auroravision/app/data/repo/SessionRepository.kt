package com.auroravision.app.data.repo

import com.auroravision.app.data.ImageStore
import com.auroravision.app.data.db.FaceRecordDao
import com.auroravision.app.data.db.FaceRecordEntity
import com.auroravision.app.data.db.SearchMatchEntity
import com.auroravision.app.data.db.SearchSessionDao
import com.auroravision.app.data.db.SearchSessionEntity
import com.auroravision.core.pipeline.AnalysisReport
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import java.io.File

/** A session row plus its decoded report, ready for the results screen. */
data class SessionDetail(
    val session: SearchSessionEntity,
    val report: AnalysisReport,
    val matchedRecords: List<FaceRecordEntity>,
)

/** Aggregate numbers shown on the home screen (all real, all from the database). */
data class AuroraStats(
    val registeredFaces: Int,
    val registeredImages: Int,
    val comparisons: Int,
    val averageDurationMs: Double?,
)

/**
 * Stores search history and resolves the metadata of the matched records.
 *
 * The full `AnalysisReport` is stored as JSON next to the indexed columns, so a
 * historical result can be re-opened (and re-read) without recomputing anything.
 */
class SessionRepository(
    private val dao: SearchSessionDao,
    private val faceDao: FaceRecordDao,
    private val imageStore: ImageStore,
) {
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    fun observeRecent(limit: Int = 5): Flow<List<SearchSessionEntity>> = dao.observeRecent(limit)

    fun observeAll(): Flow<List<SearchSessionEntity>> = dao.observeRecent(1000)

    suspend fun save(
        report: AnalysisReport,
        queryImagePath: String,
        topSimilarity: Float?,
        matches: List<com.auroravision.core.pipeline.RankedMatch>,
        qualityStatus: String,
        qualityScore: Float,
    ): Long = withContext(Dispatchers.IO) {
        val session = SearchSessionEntity(
            createdAtEpochMs = System.currentTimeMillis(),
            queryImagePath = queryImagePath,
            facesDetected = report.facesDetected,
            selectedFaceIndex = report.selectedFaceIndex,
            qualityStatus = qualityStatus,
            qualityScore = qualityScore,
            comparedCount = report.search?.comparedCount ?: 0,
            threshold = report.policy.operatingThreshold,
            verdict = report.verdict.name,
            durationMs = report.totalDurationMs,
            modelKey = report.modelKey,
            topSimilarity = topSimilarity,
            reportJson = json.encodeToString(AnalysisReport.serializer(), report),
        )
        val matchRows = matches.map {
            SearchMatchEntity(
                sessionId = 0,
                rank = it.rank,
                recordId = it.recordId,
                similarity = it.similarity,
                band = it.band.name,
            )
        }
        dao.insertSessionWithMatches(session, matchRows)
    }

    suspend fun detailOf(sessionId: Long): SessionDetail? = withContext(Dispatchers.IO) {
        val session = dao.findById(sessionId) ?: return@withContext null
        val report = runCatching {
            json.decodeFromString(AnalysisReport.serializer(), session.reportJson)
        }.getOrNull() ?: return@withContext null
        val ids = dao.matchesOf(sessionId).map { it.recordId }
        val records = if (ids.isEmpty()) emptyList() else faceDao.findByIds(ids)
        val ordered = ids.mapNotNull { id -> records.firstOrNull { it.id == id } }
        SessionDetail(session, report, ordered)
    }

    suspend fun deleteSession(sessionId: Long) = withContext(Dispatchers.IO) {
        val session = dao.findById(sessionId)
        dao.deleteById(sessionId)
        session?.let { imageStore.deleteFiles(listOf(it.queryImagePath)) }
    }

    suspend fun deleteAll() = withContext(Dispatchers.IO) {
        val sessions = dao.all()
        dao.deleteAll()
        imageStore.deleteFiles(sessions.map { it.queryImagePath })
    }

    suspend fun stats(): AuroraStats = withContext(Dispatchers.IO) {
        AuroraStats(
            registeredFaces = faceDao.all().size,
            registeredImages = faceDao.allGroupIds().size,
            comparisons = dao.all().size,
            averageDurationMs = dao.all().map { it.durationMs }.average().takeIf { !it.isNaN() },
        )
    }

    fun observeSessionCount(): Flow<Int> = dao.observeCount()

    fun readQueryImage(path: String): File = File(path)
}

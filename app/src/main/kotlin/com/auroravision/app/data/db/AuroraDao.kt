package com.auroravision.app.data.db

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Transaction
import kotlinx.coroutines.flow.Flow

@Dao
interface FaceRecordDao {

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertAll(records: List<FaceRecordEntity>): List<Long>

    @Query("SELECT * FROM face_records ORDER BY addedAtEpochMs DESC, id DESC")
    fun observeAll(): Flow<List<FaceRecordEntity>>

    @Query("SELECT * FROM face_records WHERE id = :id")
    suspend fun findById(id: Long): FaceRecordEntity?

    @Query("SELECT * FROM face_records WHERE id IN (:ids)")
    suspend fun findByIds(ids: List<Long>): List<FaceRecordEntity>

    @Query("SELECT * FROM face_records ORDER BY id")
    suspend fun all(): List<FaceRecordEntity>

    @Query("SELECT * FROM face_records WHERE modelKey = :modelKey AND qualityStatus != 'REJECTED' ORDER BY id")
    suspend fun comparableFor(modelKey: String): List<FaceRecordEntity>

    @Query("SELECT COUNT(*) FROM face_records")
    fun observeCount(): Flow<Int>

    @Query("SELECT COUNT(DISTINCT groupId) FROM face_records")
    fun observeImageCount(): Flow<Int>

    @Query("SELECT COUNT(*) FROM face_records WHERE modelKey = :modelKey AND qualityStatus != 'REJECTED'")
    suspend fun comparableCount(modelKey: String): Int

    @Query("SELECT COUNT(DISTINCT imageSha256) FROM face_records WHERE imageSha256 = :sha")
    suspend fun imagesWithSha(sha: String): Int

    @Query("SELECT DISTINCT groupId FROM face_records")
    suspend fun allGroupIds(): List<String>

    @Query("DELETE FROM face_records WHERE groupId = :groupId")
    suspend fun deleteGroup(groupId: String): Int

    @Query("DELETE FROM face_records WHERE id = :id")
    suspend fun deleteById(id: Long): Int

    @Query("DELETE FROM face_records")
    suspend fun deleteAll(): Int
}

@Dao
interface SearchSessionDao {

    @Insert
    suspend fun insertSession(session: SearchSessionEntity): Long

    @Insert
    suspend fun insertMatches(matches: List<SearchMatchEntity>)

    @Transaction
    suspend fun insertSessionWithMatches(
        session: SearchSessionEntity,
        matches: List<SearchMatchEntity>,
    ): Long {
        val id = insertSession(session)
        if (matches.isNotEmpty()) insertMatches(matches.map { it.copy(sessionId = id) })
        return id
    }

    @Query("SELECT * FROM search_sessions ORDER BY createdAtEpochMs DESC LIMIT :limit")
    fun observeRecent(limit: Int): Flow<List<SearchSessionEntity>>

    @Query("SELECT * FROM search_sessions ORDER BY createdAtEpochMs DESC")
    suspend fun all(): List<SearchSessionEntity>

    @Query("SELECT * FROM search_sessions WHERE id = :id")
    suspend fun findById(id: Long): SearchSessionEntity?

    @Query("SELECT * FROM search_matches WHERE sessionId = :sessionId ORDER BY rank")
    suspend fun matchesOf(sessionId: Long): List<SearchMatchEntity>

    @Query("DELETE FROM search_sessions WHERE id = :id")
    suspend fun deleteById(id: Long)

    @Query("DELETE FROM search_sessions")
    suspend fun deleteAll()

    @Query("SELECT COUNT(*) FROM search_sessions")
    fun observeCount(): Flow<Int>

    @Query("SELECT AVG(durationMs) FROM search_sessions")
    fun observeAverageDuration(): Flow<Double?>
}

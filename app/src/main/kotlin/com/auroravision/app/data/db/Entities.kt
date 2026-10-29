package com.auroravision.app.data.db

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * One registered face.
 *
 * An image with several faces produces several records that share a [groupId],
 * because each face carries its own embedding, quality report and box. Search
 * returns faces; the UI groups them back into their source image.
 */
@Entity(
    tableName = "face_records",
    indices = [
        Index(value = ["imageSha256", "faceIndex"], unique = true),
        Index(value = ["groupId"]),
        Index(value = ["modelKey"]),
    ],
)
data class FaceRecordEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    /** Identifier shared by every face detected in the same image. */
    val groupId: String,
    /** Optional human label ("Ahmed", "ID-card photo", ...). */
    val displayName: String? = null,
    /** Where the image came from: "المعرض", "الكاميرا", "استيراد". */
    val sourceLabel: String,
    /** Original file name or a short descriptor, shown in results. */
    val sourceDetail: String? = null,
    val addedAtEpochMs: Long,
    /** Absolute path inside the app's private storage. */
    val imagePath: String,
    val thumbnailPath: String? = null,
    val imageWidth: Int,
    val imageHeight: Int,
    val faceIndex: Int,
    val boxX: Float,
    val boxY: Float,
    val boxWidth: Float,
    val boxHeight: Float,
    /** Five landmarks as "x1,y1;x2,y2;..." — used to redraw overlays. */
    val landmarks: String,
    /** AES-GCM encrypted 128-D embedding (biometric template). */
    @ColumnInfo(typeAffinity = ColumnInfo.BLOB) val embeddingEncrypted: ByteArray,
    /** Which model produced the embedding; only same-model records are comparable. */
    val modelKey: String,
    val qualityStatus: String,
    val qualityScore: Float,
    val sharpness: Float,
    val brightness: Float,
    /** SHA-256 of the source image bytes — used for duplicate detection. */
    val imageSha256: String,
) {
    fun landmarksList(): List<Pair<Float, Float>> = landmarks.split(';').mapNotNull { part ->
        val xy = part.split(',')
        if (xy.size != 2) null
        else xy[0].toFloatOrNull()?.let { x -> xy[1].toFloatOrNull()?.let { y -> x to y } }
    }
}

/** One recorded search / analysis run, including the full serialised report. */
@Entity(
    tableName = "search_sessions",
    indices = [Index(value = ["createdAtEpochMs"])],
)
data class SearchSessionEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val createdAtEpochMs: Long,
    /** Copy of the query image inside app storage (deleted with the session). */
    val queryImagePath: String,
    val facesDetected: Int,
    val selectedFaceIndex: Int?,
    val qualityStatus: String,
    val qualityScore: Float,
    val comparedCount: Int,
    val threshold: Float,
    val verdict: String,
    val durationMs: Long,
    val modelKey: String,
    val topSimilarity: Float?,
    /** Complete `AnalysisReport` as JSON (source of truth for the detail screen). */
    val reportJson: String,
)

/** Individual matches of a session; kept separately so the history can be queried. */
@Entity(
    tableName = "search_matches",
    foreignKeys = [
        ForeignKey(
            entity = SearchSessionEntity::class,
            parentColumns = ["id"],
            childColumns = ["sessionId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index(value = ["sessionId"])],
)
data class SearchMatchEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: Long,
    val rank: Int,
    val recordId: Long,
    val similarity: Float,
    val band: String,
)

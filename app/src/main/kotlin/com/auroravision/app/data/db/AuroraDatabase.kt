package com.auroravision.app.data.db

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(
    entities = [FaceRecordEntity::class, SearchSessionEntity::class, SearchMatchEntity::class],
    version = 1,
    exportSchema = true,
)
abstract class AuroraDatabase : RoomDatabase() {
    abstract fun faceRecords(): FaceRecordDao
    abstract fun searchSessions(): SearchSessionDao

    companion object {
        fun create(context: Context): AuroraDatabase =
            Room.databaseBuilder(context, AuroraDatabase::class.java, "aurora.db")
                // No destructive migration on a biometric store: a schema change
                // must ship an explicit migration so templates are never wiped
                // silently. Version 1 is the initial schema.
                .build()
    }
}

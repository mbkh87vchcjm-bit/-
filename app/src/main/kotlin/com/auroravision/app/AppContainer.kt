package com.auroravision.app

import android.content.Context
import com.auroravision.app.data.ImageStore
import com.auroravision.app.data.crypto.EmbeddingCipher
import com.auroravision.app.data.crypto.KeystoreEmbeddingCipher
import com.auroravision.app.data.db.AuroraDatabase
import com.auroravision.app.data.repo.FaceRepository
import com.auroravision.app.data.repo.SessionRepository
import com.auroravision.app.engine.FaceEngineHolder
import com.auroravision.core.policy.MatchPolicy
import com.auroravision.core.quality.QualityPolicy

/**
 * Manual dependency container (no DI framework: the graph is small and explicit).
 *
 * Everything here is process-scoped. Models are loaded once, the database is a
 * single instance, and the embedding cipher is created lazily on first use so
 * that a missing Keystore entry cannot break app start-up.
 */
class AppContainer(context: Context) {

    private val appContext = context.applicationContext

    /** Application context (only for reading bundled assets such as the calibration report). */
    fun engineContext(): Context = appContext

    val engine: FaceEngineHolder = FaceEngineHolder(appContext)

    val database: AuroraDatabase by lazy { AuroraDatabase.create(appContext) }

    val imageStore: ImageStore by lazy { ImageStore(appContext) }

    val cipher: EmbeddingCipher by lazy { KeystoreEmbeddingCipher() }

    val matchPolicy: MatchPolicy by lazy { engine.loadMatchPolicy() }

    val qualityPolicy: QualityPolicy by lazy { engine.loadQualityPolicy() }

    val faceRepository: FaceRepository by lazy {
        FaceRepository(
            context = appContext,
            dao = database.faceRecords(),
            imageStore = imageStore,
            cipher = cipher,
            engine = engine,
        )
    }

    val sessionRepository: SessionRepository by lazy {
        SessionRepository(
            dao = database.searchSessions(),
            faceDao = database.faceRecords(),
            imageStore = imageStore,
        )
    }
}

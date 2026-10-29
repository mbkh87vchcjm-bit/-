package com.auroravision.app.ui.library

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.auroravision.app.AppContainer
import com.auroravision.app.data.db.FaceRecordEntity
import com.auroravision.app.data.repo.ReindexReport
import com.auroravision.app.data.repo.RegistrationProgress
import com.auroravision.app.data.repo.RegistrationReport
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

/** One registered image (all faces that were detected in it). */
data class LibraryImage(
    val groupId: String,
    val records: List<FaceRecordEntity>,
) {
    val primary: FaceRecordEntity get() = records.first()
    val faceCount: Int get() = records.size
    val label: String get() = primary.displayName?.takeIf { it.isNotBlank() } ?: primary.sourceDetail ?: "سجل"
}

data class LibraryUiState(
    val images: List<LibraryImage> = emptyList(),
    val totalFaces: Int = 0,
    val registering: Boolean = false,
    val progress: RegistrationProgress? = null,
    val lastReport: RegistrationReport? = null,
    val lastReindex: ReindexReport? = null,
    val busy: Boolean = false,
    val message: String? = null,
)

/**
 * Reference-image manager: import, delete, re-index.
 *
 * Import runs the real pipeline for every image (detect → quality → embed), so
 * a photo with no usable face is rejected here rather than silently stored.
 */
class ImageLibraryViewModel(private val container: AppContainer) : ViewModel() {

    private val _state = MutableStateFlow(LibraryUiState())
    val state: StateFlow<LibraryUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            container.faceRepository.observeRecords().collectLatest { records ->
                val grouped = records.groupBy { it.groupId }
                    .map { (groupId, rows) -> LibraryImage(groupId, rows.sortedBy { it.faceIndex }) }
                    .sortedByDescending { it.primary.addedAtEpochMs }
                _state.value = _state.value.copy(
                    images = grouped,
                    totalFaces = records.size,
                )
            }
        }
    }

    fun register(uris: List<Uri>, sourceLabel: String, displayName: String? = null) {
        if (uris.isEmpty() || _state.value.registering) return
        viewModelScope.launch {
            _state.value = _state.value.copy(registering = true, lastReport = null, message = null)
            val report = container.faceRepository.registerUris(
                uris = uris,
                sourceLabel = sourceLabel,
                displayName = displayName,
            ) { progress ->
                _state.value = _state.value.copy(progress = progress)
            }
            _state.value = _state.value.copy(
                registering = false,
                progress = null,
                lastReport = report,
                message = summarize(report),
            )
        }
    }

    fun registerFiles(files: List<java.io.File>, sourceLabel: String) {
        if (files.isEmpty() || _state.value.registering) return
        viewModelScope.launch {
            _state.value = _state.value.copy(registering = true, lastReport = null, message = null)
            var last: RegistrationReport? = null
            for (file in files) {
                last = container.faceRepository.registerFile(file, sourceLabel) { progress ->
                    _state.value = _state.value.copy(progress = progress)
                }
            }
            _state.value = _state.value.copy(
                registering = false,
                progress = null,
                lastReport = last,
                message = last?.let { summarize(it) },
            )
        }
    }

    fun deleteImage(groupId: String) {
        viewModelScope.launch {
            _state.value = _state.value.copy(busy = true)
            val ok = container.faceRepository.deleteGroup(groupId)
            _state.value = _state.value.copy(
                busy = false,
                message = if (ok) "تم حذف الصورة وتمثيلاتها الرقمية" else "لم يتم العثور على السجل",
            )
        }
    }

    fun deleteFace(recordId: Long) {
        viewModelScope.launch {
            _state.value = _state.value.copy(busy = true)
            container.faceRepository.deleteRecord(recordId)
            _state.value = _state.value.copy(busy = false, message = "تم حذف الوجه وسجله")
        }
    }

    fun deleteAll() {
        viewModelScope.launch {
            _state.value = _state.value.copy(busy = true)
            val deleted = container.faceRepository.deleteAll()
            _state.value = _state.value.copy(busy = false, message = "تم حذف $deleted سجلًا")
        }
    }

    fun reindex() {
        if (_state.value.busy || _state.value.registering) return
        viewModelScope.launch {
            _state.value = _state.value.copy(busy = true, lastReindex = null)
            val report = container.faceRepository.reindexAll { progress ->
                _state.value = _state.value.copy(progress = progress)
            }
            _state.value = _state.value.copy(
                busy = false,
                progress = null,
                lastReindex = report,
                message = "إعادة الفهرسة: تحديث ${report.recordsUpdated} سجلًا، حذف ${report.recordsRemoved}",
            )
        }
    }

    fun clearMessage() {
        _state.value = _state.value.copy(message = null)
    }

    private fun summarize(r: RegistrationReport): String = buildString {
        append("أُضيفت ${r.imagesAccepted} صورة (${r.facesStored} وجهًا)")
        if (r.imagesDuplicate > 0) append(" · مكررة: ${r.imagesDuplicate}")
        if (r.imagesUnreadable > 0) append(" · غير مقروءة: ${r.imagesUnreadable}")
        if (r.facesRejected > 0) append(" · رُفضت للجودة: ${r.facesRejected}")
        append(" · ${r.durationMs} ms")
    }

    companion object {
        fun factory(container: AppContainer): ViewModelProvider.Factory =
            object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T =
                    ImageLibraryViewModel(container) as T
            }
    }
}

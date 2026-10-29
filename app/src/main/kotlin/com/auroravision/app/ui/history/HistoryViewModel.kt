package com.auroravision.app.ui.history

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.auroravision.app.AppContainer
import com.auroravision.app.data.db.SearchSessionEntity
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

data class HistoryUiState(
    val sessions: List<SearchSessionEntity> = emptyList(),
    val message: String? = null,
)

/** Search history: every entry can be opened again or deleted with its files. */
class HistoryViewModel(private val container: AppContainer) : ViewModel() {

    private val _state = MutableStateFlow(HistoryUiState())
    val state: StateFlow<HistoryUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            container.sessionRepository.observeAll().collectLatest { list ->
                _state.value = _state.value.copy(sessions = list)
            }
        }
    }

    fun delete(sessionId: Long) {
        viewModelScope.launch {
            container.sessionRepository.deleteSession(sessionId)
            _state.value = _state.value.copy(message = "تم حذف السجل")
        }
    }

    fun deleteAll() {
        viewModelScope.launch {
            container.sessionRepository.deleteAll()
            _state.value = _state.value.copy(message = "تم حذف كل سجلات البحث")
        }
    }

    fun clearMessage() {
        _state.value = _state.value.copy(message = null)
    }

    companion object {
        fun factory(container: AppContainer): ViewModelProvider.Factory =
            object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T =
                    HistoryViewModel(container) as T
            }
    }
}

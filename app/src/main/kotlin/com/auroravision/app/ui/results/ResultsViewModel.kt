package com.auroravision.app.ui.results

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.auroravision.app.AppContainer
import com.auroravision.app.data.repo.SessionDetail
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class ResultsUiState(
    val loading: Boolean = true,
    val detail: SessionDetail? = null,
    val error: String? = null,
)

/**
 * Loads one stored session (the full report, its query image and the metadata of
 * every matched record). Used both right after an analysis and from history.
 */
class ResultsViewModel(
    private val container: AppContainer,
    private val sessionId: Long,
) : ViewModel() {

    private val _state = MutableStateFlow(ResultsUiState())
    val state: StateFlow<ResultsUiState> = _state.asStateFlow()

    init {
        load()
    }

    fun load() {
        viewModelScope.launch {
            val detail = container.sessionRepository.detailOf(sessionId)
            _state.value = ResultsUiState(
                loading = false,
                detail = detail,
                error = if (detail == null) "لم يتم العثور على سجل البحث" else null,
            )
        }
    }

    fun deleteSession(onDeleted: () -> Unit) {
        viewModelScope.launch {
            container.sessionRepository.deleteSession(sessionId)
            onDeleted()
        }
    }

    companion object {
        fun factory(container: AppContainer, sessionId: Long): ViewModelProvider.Factory =
            object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T =
                    ResultsViewModel(container, sessionId) as T
            }
    }
}

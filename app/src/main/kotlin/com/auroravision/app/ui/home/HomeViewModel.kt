package com.auroravision.app.ui.home

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.auroravision.app.AppContainer
import com.auroravision.app.data.db.SearchSessionEntity
import com.auroravision.app.data.repo.AuroraStats
import com.auroravision.app.engine.EngineState
import com.auroravision.core.policy.MatchPolicy
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class HomeUiState(
    val engine: EngineState = EngineState.Loading,
    val stats: AuroraStats = AuroraStats(0, 0, 0, null),
    val recent: List<SearchSessionEntity> = emptyList(),
    val policy: MatchPolicy = MatchPolicy.FALLBACK,
    val topK: Int = 5,
    val loading: Boolean = true,
)

/**
 * Home screen state: engine readiness, real database statistics and the last
 * searches. Nothing here is static text — the numbers come from Room.
 */
class HomeViewModel(private val container: AppContainer) : ViewModel() {

    private val _state = MutableStateFlow(HomeUiState())
    val state: StateFlow<HomeUiState> = _state.asStateFlow()

    init {
        _state.value = HomeUiState(policy = container.matchPolicy, topK = container.matchPolicy.defaultTopK)
        viewModelScope.launch {
            val engine = withContext(Dispatchers.IO) { container.engine.ensureLoaded() }
            _state.value = _state.value.copy(engine = engine)
            refreshStats(engine)
        }
        viewModelScope.launch {
            container.sessionRepository.observeRecent(4).collectLatest { sessions ->
                _state.value = _state.value.copy(recent = sessions, loading = false)
            }
        }
    }

    fun refresh() {
        viewModelScope.launch {
            val engine = withContext(Dispatchers.IO) { container.engine.ensureLoaded() }
            _state.value = _state.value.copy(engine = engine)
            refreshStats(engine)
        }
    }

    private suspend fun refreshStats(engine: EngineState) {
        val stats = withContext(Dispatchers.IO) { container.sessionRepository.stats() }
        _state.value = _state.value.copy(stats = stats, engine = engine)
    }

    companion object {
        fun factory(container: AppContainer): ViewModelProvider.Factory =
            object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T =
                    HomeViewModel(container) as T
            }
    }
}

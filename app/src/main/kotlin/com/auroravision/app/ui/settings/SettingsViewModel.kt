package com.auroravision.app.ui.settings

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import com.auroravision.app.AppContainer
import com.auroravision.core.policy.MatchPolicy
import com.auroravision.core.policy.ModelInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.float
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

data class SettingsUiState(
    val models: ModelInfo = ModelInfo.DEFAULT,
    val policy: MatchPolicy = MatchPolicy.FALLBACK,
    val measuredAuc: Float? = null,
    val measuredEer: Float? = null,
    val measuredFar: Float? = null,
    val dataset: String = "",
    val calibrationGeneratedAt: String = "",
    val sampleCount: Int = 0,
    val message: String? = null,
)

/**
 * Privacy / model information. The calibration numbers shown here are read from
 * the same `calibration_report.json` that `tools/reference` produced, so the app
 * cannot drift from the documented evaluation.
 */
class SettingsViewModel(private val container: AppContainer) : ViewModel() {

    private val _state = MutableStateFlow(
        SettingsUiState(models = container.engine.modelInfo, policy = container.matchPolicy),
    )
    val state: StateFlow<SettingsUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch {
            val loaded = withContext(Dispatchers.IO) { readCalibration(container) }
            _state.value = _state.value.copy(
                measuredAuc = loaded?.auc,
                measuredEer = loaded?.eer,
                measuredFar = loaded?.far,
                dataset = loaded?.dataset ?: "",
                calibrationGeneratedAt = loaded?.generatedAt ?: "",
                sampleCount = loaded?.samples ?: 0,
            )
        }
    }

    fun deleteAllData() {
        viewModelScope.launch {
            val faces = withContext(Dispatchers.IO) { container.faceRepository.deleteAll() }
            val sessions = withContext(Dispatchers.IO) { container.sessionRepository.deleteAll() }
            _state.value = _state.value.copy(
                message = "تم حذف $faces سجل وجه و $sessions سجل بحث، مع الصور والتمثيلات المرتبطة.",
            )
        }
    }

    fun clearMessage() {
        _state.value = _state.value.copy(message = null)
    }

    private fun readCalibration(container: AppContainer): CalibrationFacts? = runCatching {
        val ctx = container.engineContext()
        val text = ctx.assets.open("calibration/calibration_report.json")
            .bufferedReader().use { it.readText() }
        val root = Json.parseToJsonElement(text).jsonObject
        val test = root["test"]!!.jsonObject
        val dataset = root["dataset"]!!.jsonObject
        CalibrationFacts(
            auc = test["auc"]!!.jsonPrimitive.float,
            eer = test["eer"]!!.jsonPrimitive.float,
            far = test["far"]!!.jsonPrimitive.float,
            dataset = dataset["name"]!!.jsonPrimitive.content,
            generatedAt = root["generatedAt"]!!.jsonPrimitive.content,
            samples = (test["samePairs"]!!.jsonPrimitive.content.toIntOrNull() ?: 0) +
                (test["diffPairs"]!!.jsonPrimitive.content.toIntOrNull() ?: 0),
        )
    }.getOrNull()

    private data class CalibrationFacts(
        val auc: Float,
        val eer: Float,
        val far: Float,
        val dataset: String,
        val generatedAt: String,
        val samples: Int,
    )

    companion object {
        fun factory(container: AppContainer): ViewModelProvider.Factory =
            object : ViewModelProvider.Factory {
                @Suppress("UNCHECKED_CAST")
                override fun <T : ViewModel> create(modelClass: Class<T>): T =
                    SettingsViewModel(container) as T
            }
    }
}

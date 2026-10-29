package com.auroravision.app.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.auroravision.app.AppContainer
import com.auroravision.app.ui.analyze.AnalyzeScreen
import com.auroravision.app.ui.analyze.AnalyzeViewModel
import com.auroravision.app.ui.camera.CameraScreen
import com.auroravision.app.ui.history.HistoryScreen
import com.auroravision.app.ui.history.HistoryViewModel
import com.auroravision.app.ui.home.HomeScreen
import com.auroravision.app.ui.home.HomeViewModel
import com.auroravision.app.ui.library.ImageLibraryViewModel
import com.auroravision.app.ui.library.LibraryScreen
import com.auroravision.app.ui.results.ResultsScreen
import com.auroravision.app.ui.results.ResultsViewModel
import com.auroravision.app.ui.settings.SettingsScreen
import com.auroravision.app.ui.settings.SettingsViewModel
import java.io.File

object Routes {
    const val HOME = "home"
    const val ANALYZE = "analyze"
    const val LIBRARY = "library"
    const val HISTORY = "history"
    const val SETTINGS = "settings"
    const val CAMERA = "camera"
    const val RESULTS = "results/{sessionId}"

    fun results(sessionId: Long) = "results/$sessionId"
}

@Composable
fun AuroraNavHost(
    container: AppContainer,
    analyzeViewModel: AnalyzeViewModel,
    onPickImage: () -> Unit,
    onPickImagesForLibrary: () -> Unit,
    navController: NavHostController = rememberNavController(),
) {
    NavHost(navController = navController, startDestination = Routes.HOME) {
        composable(Routes.HOME) {
            val vm: HomeViewModel = androidx.lifecycle.viewmodel.compose.viewModel(
                factory = HomeViewModel.factory(container),
            )
            HomeScreen(
                viewModel = vm,
                onStartAnalysis = onPickImage,
                onOpenCamera = { navController.navigate(Routes.CAMERA) },
                onManageLibrary = { navController.navigate(Routes.LIBRARY) },
                onOpenHistory = { navController.navigate(Routes.HISTORY) },
                onOpenSettings = { navController.navigate(Routes.SETTINGS) },
                onOpenSession = { navController.navigate(Routes.results(it)) },
            )
        }

        composable(Routes.ANALYZE) {
            AnalyzeScreen(
                viewModel = analyzeViewModel,
                onOpenReport = { navController.navigate(Routes.results(it)) },
                onBack = { navController.popBackStack() },
            )
        }

        composable(Routes.LIBRARY) {
            val vm: ImageLibraryViewModel = androidx.lifecycle.viewmodel.compose.viewModel(
                factory = ImageLibraryViewModel.factory(container),
            )
            LibraryScreen(
                viewModel = vm,
                onPickImages = onPickImagesForLibrary,
                onOpenCamera = { navController.navigate(Routes.CAMERA) },
                onBack = { navController.popBackStack() },
            )
        }

        composable(Routes.HISTORY) {
            val vm: HistoryViewModel = androidx.lifecycle.viewmodel.compose.viewModel(
                factory = HistoryViewModel.factory(container),
            )
            HistoryScreen(
                viewModel = vm,
                onOpenSession = { navController.navigate(Routes.results(it)) },
                onBack = { navController.popBackStack() },
            )
        }

        composable(Routes.SETTINGS) {
            val vm: SettingsViewModel = androidx.lifecycle.viewmodel.compose.viewModel(
                factory = SettingsViewModel.factory(container),
            )
            SettingsScreen(viewModel = vm, onBack = { navController.popBackStack() })
        }

        composable(Routes.CAMERA) {
            CameraScreen(
                onCaptured = { file: File ->
                    analyzeViewModel.analyzeFile(file)
                    navController.navigate(Routes.ANALYZE) {
                        popUpTo(Routes.CAMERA) { inclusive = true }
                    }
                },
                onBack = { navController.popBackStack() },
            )
        }

        composable(Routes.RESULTS) { entry ->
            val sessionId = entry.arguments?.getString("sessionId")?.toLongOrNull() ?: -1L
            val vm: ResultsViewModel = androidx.lifecycle.viewmodel.compose.viewModel(
                factory = ResultsViewModel.factory(container, sessionId),
            )
            ResultsScreen(
                viewModel = vm,
                onBack = { navController.popBackStack() },
                onDeleted = { navController.popBackStack() },
            )
        }
    }
}

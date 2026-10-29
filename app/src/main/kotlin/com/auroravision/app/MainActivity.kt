package com.auroravision.app

import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.lifecycle.ViewModelProvider
import androidx.navigation.compose.rememberNavController
import com.auroravision.app.ui.AuroraNavHost
import com.auroravision.app.ui.Routes
import com.auroravision.app.ui.analyze.AnalyzeViewModel
import com.auroravision.app.ui.library.ImageLibraryViewModel
import com.auroravision.app.ui.theme.AuroraTheme

/**
 * Single-activity host.
 *
 * Photo selection uses the Android Photo Picker (`PickVisualMedia`), which grants
 * access to the one image the user chose — no storage permission is requested.
 */
class MainActivity : ComponentActivity() {

    private val container: AppContainer
        get() = (application as AuroraApplication).container

    private val analyzeViewModel: AnalyzeViewModel by lazy {
        ViewModelProvider(this, AnalyzeViewModel.factory(container, container.sessionRepository))
            .get(AnalyzeViewModel::class.java)
    }

    private val libraryViewModel: ImageLibraryViewModel by lazy {
        ViewModelProvider(this, ImageLibraryViewModel.factory(container))
            .get(ImageLibraryViewModel::class.java)
    }

    private var onPicked: (List<Uri>) -> Unit = {}

    private val pickSingle = registerForActivityResult(
        ActivityResultContracts.PickVisualMedia(),
    ) { uri ->
        if (uri != null) onPicked(listOf(uri))
    }

    private val pickMultiple = registerForActivityResult(
        ActivityResultContracts.PickMultipleVisualMedia(20),
    ) { uris ->
        if (uris.isNotEmpty()) onPicked(uris)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            AuroraTheme {
                val navController = rememberNavController()
                AuroraNavHost(
                    container = container,
                    analyzeViewModel = analyzeViewModel,
                    navController = navController,
                    onPickImage = {
                        onPicked = { uris ->
                            uris.firstOrNull()?.let { uri ->
                                analyzeViewModel.analyze(uri)
                                navController.navigate(Routes.ANALYZE)
                            }
                        }
                        pickSingle.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                        )
                    },
                    onPickImagesForLibrary = {
                        onPicked = { uris ->
                            libraryViewModel.register(uris, sourceLabel = "المعرض")
                        }
                        pickMultiple.launch(
                            PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly),
                        )
                    },
                )
            }
        }
    }
}

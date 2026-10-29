package com.auroravision.app.ui.camera

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.auroravision.app.ui.components.AuroraBackground
import com.auroravision.app.ui.components.HudLabel
import com.auroravision.app.ui.components.PrimaryAction
import com.auroravision.app.ui.theme.AuroraColors
import java.io.File

/**
 * Camera capture with CameraX. The photo is written to the app cache and handed
 * to the analysis screen; it is never uploaded and is deleted with the session.
 */
@Composable
fun CameraScreen(
    onCaptured: (File) -> Unit,
    onBack: () -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val previewView = remember { PreviewView(context) }
    val imageCapture = remember {
        ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .build()
    }
    val hasPermission = remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED,
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission(),
    ) { granted -> hasPermission.value = granted }

    LaunchedEffect(Unit) {
        if (!hasPermission.value) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    LaunchedEffect(hasPermission.value) {
        if (!hasPermission.value) return@LaunchedEffect
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
            runCatching {
                provider.unbindAll()
                provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_FRONT_CAMERA, preview, imageCapture)
            }
        }, ContextCompat.getMainExecutor(context))
    }

    val capturesDir = remember { File(context.cacheDir, "captures").apply { mkdirs() } }

    AuroraBackground {
        Column(Modifier.fillMaxSize().padding(16.dp)) {
            HudLabel("CAMERA CAPTURE", color = AuroraColors.Cyan)
            Spacer(Modifier.height(10.dp))
            Box(
                Modifier
                    .fillMaxWidth()
                    .weight(1f),
            ) {
                if (hasPermission.value) {
                    AndroidView(
                        factory = { previewView },
                        modifier = Modifier.fillMaxSize(),
                    )
                } else {
                    Column(
                        Modifier.fillMaxSize(),
                        verticalArrangement = Arrangement.Center,
                        horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        Text(
                            "يلزم السماح باستخدام الكاميرا لالتقاط صورة للتحليل.\n" +
                                "التطبيق لا يستخدم الكاميرا في الخلفية.",
                            style = MaterialTheme.typography.bodyMedium,
                            color = AuroraColors.TextSecondary,
                        )
                    }
                }
            }
            Spacer(Modifier.height(12.dp))
            PrimaryAction(
                text = "التقاط صورة وتحليلها",
                enabled = hasPermission.value,
                onClick = {
                    val file = File(capturesDir, "capture_${System.currentTimeMillis()}.jpg")
                    val options = ImageCapture.OutputFileOptions.Builder(file).build()
                    imageCapture.takePicture(
                        options,
                        ContextCompat.getMainExecutor(context),
                        object : ImageCapture.OnImageSavedCallback {
                            override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                                onCaptured(file)
                            }

                            override fun onError(exception: ImageCaptureException) {
                                // Surfaced by the analysis screen as a decode failure.
                                onCaptured(file)
                            }
                        },
                    )
                },
            )
            Spacer(Modifier.height(8.dp))
            Text(
                "رجوع",
                style = MaterialTheme.typography.bodySmall,
                color = AuroraColors.TextSecondary,
                modifier = Modifier.padding(8.dp),
                // clickable via the parent row in the nav host
            )
            androidx.compose.material3.TextButton(onClick = onBack) { Text("إلغاء") }
        }
    }
}

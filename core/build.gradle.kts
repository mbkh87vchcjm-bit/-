plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
}

/**
 * :core — platform-independent computer-vision core.
 *
 * Deliberately a plain JVM module (no Android APIs) so that the *exact same*
 * code that runs on the phone can be executed and verified by JVM unit tests
 * against the real ONNX models (see FaceEngineModelTest).
 */
kotlin {
    jvmToolchain(17)
    compilerOptions {
        freeCompilerArgs.addAll("-Xjvm-default=all")
    }
}

dependencies {
    // The ONNX Runtime API is provided at compile time by whoever hosts the
    // runtime: `onnxruntime-android` in the app, `onnxruntime` on the JVM.
    compileOnly(libs.onnxruntime.jvm)

    api(libs.kotlinx.serialization.json)

    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.onnxruntime.jvm)
}

tasks.withType<Test>().configureEach {
    // Model files live in the app's assets; tests read them through this path.
    systemProperty("aurora.models.dir", rootProject.file("app/src/main/assets/models").absolutePath)
    systemProperty("aurora.calibration.dir", rootProject.file("app/src/main/assets/calibration").absolutePath)
    systemProperty("aurora.repo.root", rootProject.projectDir.absolutePath)
    maxHeapSize = "2g"
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
    }
}

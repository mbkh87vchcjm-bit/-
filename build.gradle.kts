// Root build file — plugins are declared here and applied in the modules.
plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.android.library) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.compose) apply false
    alias(libs.plugins.kotlin.serialization) apply false
    alias(libs.plugins.ksp) apply false
}

/**
 * Shared verification task: runs every JVM unit test in the project plus the
 * model-parity tests (see :core). Used by CI and by `./gradlew verifyAll`.
 */
tasks.register("verifyAll") {
    group = "verification"
    description = "Runs all JVM unit tests for :core and :app."
    dependsOn(":core:test", ":app:testDebugUnitTest")
}

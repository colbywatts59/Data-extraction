package com.example.gridbuttonapp.data

/**
 * Data class for recording requests sent to the server
 */
data class RecordingRequest(
    val button: Int,
    val rows: Int,
    val cols: Int,
    val timestamp: Long,
    val press_length_ms: Long,
    val network_latency_ms: Double = 0.0
)

/**
 * Data class for server responses
 */
data class RecordingResponse(
    val status: String,
    val message: String? = null,
    val file: String? = null,
    val delay_ms: Int? = null,
    val server_time: Double? = null
)


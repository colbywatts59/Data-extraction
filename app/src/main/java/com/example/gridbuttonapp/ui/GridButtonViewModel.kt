package com.example.gridbuttonapp.ui

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.gridbuttonapp.data.RecordingRequest
import com.example.gridbuttonapp.network.NetworkModule
import com.example.gridbuttonapp.network.RecordingApi
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.util.*

/**
 * ViewModel for managing grid button state and network operations
 */
class GridButtonViewModel(
    initialRows: Int = 3,
    initialCols: Int = 2,
    apiUrl: String = "http://10.0.0.92:5150/",
    recordingDelay: Long = 100L,
    showGraphs: Boolean = false,
    useTestDataset: Boolean = false,
    initialNetworkLatency: Double = 0.0,
    manualTestMode: Boolean = false
) : ViewModel() {
    
    // ===== CONFIGURATION - MODIFY THESE VALUES =====
    // Delay between start_recording and stop_recording (in milliseconds)
    private var recordingDelayMs = recordingDelay
    // ================================================
    
    private val _uiState = MutableStateFlow(
        GridButtonUiState(
            rows = initialRows,
            cols = initialCols,
            buttonStates = List(initialRows * initialCols) { false },
            networkStatus = false,
            clockSyncResult = if (initialNetworkLatency > 0) {
                ClockSyncResult(
                    networkLatencyMs = initialNetworkLatency,
                    serverTimeMs = 0,
                    phoneTimeMs = 0,
                    offsetMs = 0.0,
                    phoneAhead = false
                )
            } else null,
            manualTestMode = manualTestMode
        )
    )
    val uiState: StateFlow<GridButtonUiState> = _uiState.asStateFlow()
    
    // Initialize recording API (will be set after updating base URL)
    private val recordingApi: RecordingApi
    
    init {
        // Update the network module with the provided API URL FIRST
        NetworkModule.updateBaseUrl(apiUrl)
        // THEN get the recording API with the updated URL
        recordingApi = NetworkModule.recordingApi
        println("GridButtonViewModel created with API URL: $apiUrl, delay: $recordingDelayMs ms")
        
        // Initialize AD3 connection
        initializeAd3()
        
        // Set graph visualization preference
        setGraphVisualization(showGraphs)
        setDatasetMode(useTestDataset)
    }
    
    /**
     * Initialize AD3 device
     */
    private fun initializeAd3() {
        viewModelScope.launch {
            try {
                println("Initializing AD3...")
                val response = recordingApi.initialize()
                if (response.isSuccessful && response.body()?.status == "success") {
                    println("✓ AD3 initialized successfully")
                    updateNetworkStatus(true)
                } else {
                    println("✗ Failed to initialize AD3: ${response.body()?.message}")
                    updateNetworkStatus(false)
                }
            } catch (e: Exception) {
                println("✗ Error initializing AD3: ${e.message}")
                e.printStackTrace()
                updateNetworkStatus(false)
            }
        }
    }
    
    /**
     * Measure current network latency by sending a ping
     * Returns latency in milliseconds
     */
    private suspend fun measureNetworkLatency(): Double {
        return try {
            val tSend = System.currentTimeMillis()
            val response = recordingApi.ping()
            val tReceive = System.currentTimeMillis()
            
            if (response.isSuccessful) {
                val roundTripMs = tReceive - tSend
                val latencyMs = roundTripMs / 2.0
                latencyMs
            } else {
                println("⚠️  Ping failed, using 0ms latency")
                0.0
            }
        } catch (e: Exception) {
            println("⚠️  Ping error: ${e.message}, using 0ms latency")
            0.0
        }
    }
    
    /**
     * Core button press logic - suspend function that can be awaited
     */
    private suspend fun performButtonPress(buttonIndex: Int) {
        // In manual test mode, skip all network requests and just flip the button locally
        if (_uiState.value.manualTestMode) {
            println("Manual test mode enabled - skipping network calls for button $buttonIndex")
            updateButtonState(buttonIndex, true)
            delay(recordingDelayMs)
            updateButtonState(buttonIndex, false)
            println("Manual test press completed for button $buttonIndex")
            return
        }
        
        try {
            // Ping API first to get current network latency
            val networkLatency = measureNetworkLatency()
            println("Current network latency: ${networkLatency}ms")
            
            val timestamp = System.currentTimeMillis()
            val request = RecordingRequest(
                button = buttonIndex,
                rows = _uiState.value.rows,
                cols = _uiState.value.cols,
                timestamp = timestamp,
                press_length_ms = recordingDelayMs,
                network_latency_ms = networkLatency
            )
            
            println("Sending recording request for button $buttonIndex...")
            
            // Send start_recording request and WAIT for response
            val startResponse = recordingApi.startRecording(request)
            
            if (startResponse.isSuccessful && startResponse.body()?.status == "success") {
                // TIMESTAMP 1: Response received
                val t1_responseReceived = System.currentTimeMillis()
                println("═══ TIMING LOG ═══")
                println("T1: Response received at ${t1_responseReceived}ms")
                
                // Server responded successfully
                val responseBody = startResponse.body()
                updateNetworkStatus(true)
                
                val delayMs = responseBody?.delay_ms ?: 0
                
                println("Recording will start in ${delayMs}ms")
                
                // TIMESTAMP 2: About to start delay
                val t2_beforeDelay = System.currentTimeMillis()
                val processingTime1 = t2_beforeDelay - t1_responseReceived
                println("T2: Starting delay at ${t2_beforeDelay}ms (processing: ${processingTime1}ms)")
                
                // WAIT for sync delay before flipping button (synchronized with API)
                delay(delayMs.toLong())
                
                // TIMESTAMP 3: Delay complete, about to flip button
                val t3_beforeFlip = System.currentTimeMillis()
                val actualDelayTime = t3_beforeFlip - t2_beforeDelay
                println("T3: Delay complete at ${t3_beforeFlip}ms (actual delay: ${actualDelayTime}ms)")
                
                // NOW update the button to black (synchronized with recording start)
                updateButtonState(buttonIndex, true)
                
                // TIMESTAMP 4: Button actually flipped
                val t4_afterFlip = System.currentTimeMillis()
                val flipTime = t4_afterFlip - t3_beforeFlip
                println("T4: Button flipped at ${t4_afterFlip}ms (flip took: ${flipTime}ms)")
                
                val totalTime = t4_afterFlip - t1_responseReceived
                println("═══ TOTAL: ${totalTime}ms from response to button flip ═══")
                println("  Processing before delay: ${processingTime1}ms")
                println("  Actual delay: ${actualDelayTime}ms")
                println("  Button flip UI update: ${flipTime}ms")
                
                // Wait for press duration then reset button
                delay(recordingDelayMs)
                updateButtonState(buttonIndex, false)
                println("Button $buttonIndex reset after $recordingDelayMs ms")
            } else {
                // Network failed - don't change button state
                updateNetworkStatus(false)
                println("Recording request failed for button $buttonIndex - button stays white")
            }
        } catch (e: Exception) {
            // Network error - don't change button state
            updateNetworkStatus(false)
            println("Network error for button $buttonIndex: ${e.message}")
            e.printStackTrace()
        }
    }
    
    /**
     * Handle button click - sends network requests and updates UI
     */
    fun onButtonClick(buttonIndex: Int) {
        println("Button $buttonIndex clicked!")
        viewModelScope.launch {
            performButtonPress(buttonIndex)
        }
    }
    
    /**
     * Update the pressed state of a specific button
     */
    private fun updateButtonState(buttonIndex: Int, isPressed: Boolean) {
        println("updateButtonState: button=$buttonIndex, isPressed=$isPressed")
        val currentStates = _uiState.value.buttonStates.toMutableList()
        if (buttonIndex in currentStates.indices) {
            currentStates[buttonIndex] = isPressed
            _uiState.value = _uiState.value.copy(buttonStates = currentStates)
            println("State updated - button $buttonIndex is now ${if (isPressed) "BLACK" else "WHITE"}")
            println("Current button states: ${_uiState.value.buttonStates}")
        } else {
            println("ERROR: buttonIndex $buttonIndex out of bounds (0..${currentStates.size - 1})")
        }
    }
    
    /**
     * Update the network status
     */
    private fun updateNetworkStatus(isConnected: Boolean) {
        _uiState.value = _uiState.value.copy(networkStatus = isConnected)
    }
    
    /**
     * Update the recording delay
     */
    fun updateRecordingDelay(newDelay: Long) {
        recordingDelayMs = newDelay
    }
    
    /**
     * Set graph visualization preference on server
     */
    private fun setGraphVisualization(enabled: Boolean) {
        viewModelScope.launch {
            try {
                val request = mapOf("enabled" to enabled)
                val response = recordingApi.toggleGraphs(request)
                if (response.isSuccessful) {
                    println("Graph visualization ${if (enabled) "enabled" else "disabled"}")
                }
            } catch (e: Exception) {
                println("Error setting graph visualization: ${e.message}")
            }
        }
    }
    
    fun updateGraphPreference(enabled: Boolean) {
        setGraphVisualization(enabled)
    }
    
    /**
     * Set dataset mode on server (train/test)
     */
    private fun setDatasetMode(useTestDataset: Boolean) {
        viewModelScope.launch {
            try {
                val mode = if (useTestDataset) "test" else "train"
                val response = recordingApi.setDatasetMode(mapOf("mode" to mode))
                if (response.isSuccessful) {
                    println("Dataset mode set to $mode")
                } else {
                    println("Failed to set dataset mode: ${response.body()?.message}")
                }
            } catch (e: Exception) {
                println("Error setting dataset mode: ${e.message}")
            }
        }
    }
    
    fun updateDatasetMode(useTestDataset: Boolean) {
        setDatasetMode(useTestDataset)
    }
    
    /**
     * Reset all buttons to unpressed state (useful for testing)
     */
    fun resetAllButtons() {
        _uiState.value = _uiState.value.copy(
            buttonStates = List(_uiState.value.rows * _uiState.value.cols) { false }
        )
    }
    
    /**
     * Repeat a button press a specified number of times
     * Each press will be executed sequentially with proper delays
     * 
     * @param buttonIndex The button to press
     * @param count Number of times to repeat
     * @param onProgress Callback with (current, total) progress
     * @param onComplete Callback when sequence finishes
     */
    fun repeatButtonPress(
        buttonIndex: Int, 
        count: Int,
        onProgress: ((current: Int, total: Int) -> Unit)? = null,
        onComplete: (() -> Unit)? = null
    ) {
        if (buttonIndex < 0 || buttonIndex >= _uiState.value.rows * _uiState.value.cols) {
            println("ERROR: Invalid button index $buttonIndex for repeat")
            onComplete?.invoke()
            return
        }
        
        if (count <= 0) {
            println("ERROR: Repeat count must be positive, got $count")
            onComplete?.invoke()
            return
        }
        
        println("Starting repeat sequence: Button $buttonIndex, $count times")
        
        viewModelScope.launch {
            // Update progress to show we're starting
            onProgress?.invoke(0, count)
            
            for (i in 1..count) {
                println("Repeat $i/$count: Pressing button $buttonIndex")
                // Use the awaitable suspend function to ensure each press completes before the next
                performButtonPress(buttonIndex)
                
                // Update progress
                onProgress?.invoke(i, count)
                
                // Wait for server recording to complete before starting next press
                // Server recording takes approximately:
                // - server_wait_ms (~100-200ms)
                // - Recording duration: 32768 samples / 50000 Hz * 1000 = ~655ms
                // Total: ~750-900ms
                // We add a buffer to ensure the server is completely done
                if (i < count) {
                    val serverRecordingBufferMs = 1000L  // Wait 1 second between presses to ensure server is done
                    println("Waiting ${serverRecordingBufferMs}ms for server recording to complete...")
                    delay(serverRecordingBufferMs)
                }
            }
            println("Repeat sequence completed: $count presses of button $buttonIndex")
            onComplete?.invoke()
        }
    }
    
    /**
     * Test clock synchronization with server
     */
    fun testClockSync() {
        viewModelScope.launch {
            try {
                println("\n═══ CLOCK SYNC TEST ═══")
                
                // Record time before sending request
                val tSendPhone = System.currentTimeMillis()
                println("T_send (phone): $tSendPhone ms")
                
                // Send ping request to server
                val response = recordingApi.ping()
                
                // Record time when response is received
                val tReceivePhone = System.currentTimeMillis()
                println("T_receive (phone): $tReceivePhone ms")
                
                if (response.isSuccessful && response.body()?.status == "success") {
                    val serverTime = response.body()?.server_time ?: 0.0
                    val tServerMs = (serverTime * 1000).toLong()
                    
                    println("T_server: $tServerMs ms")
                    
                    // Calculate network latency (round-trip time / 2)
                    val roundTripMs = tReceivePhone - tSendPhone
                    val latencyMs = roundTripMs / 2.0
                    
                    println("Round trip time: $roundTripMs ms")
                    println("Estimated one-way latency: $latencyMs ms")
                    
                    // Estimate what server time would be right now on the phone
                    val estimatedServerTimeMs = tServerMs + latencyMs.toLong()
                    val currentPhoneTimeMs = System.currentTimeMillis()
                    
                    // Calculate offset (positive = phone ahead, negative = phone behind)
                    val offsetMs = currentPhoneTimeMs - estimatedServerTimeMs
                    val phoneAhead = offsetMs > 0
                    
                    println("Estimated server time now: $estimatedServerTimeMs ms")
                    println("Current phone time: $currentPhoneTimeMs ms")
                    println("Clock offset: ${if (phoneAhead) "+" else ""}$offsetMs ms")
                    println("Phone is ${if (phoneAhead) "AHEAD" else "BEHIND"} of server")
                    println("═══════════════════════\n")
                    
                    // Update UI state with results
                    _uiState.value = _uiState.value.copy(
                        clockSyncResult = ClockSyncResult(
                            networkLatencyMs = latencyMs,
                            serverTimeMs = estimatedServerTimeMs,
                            phoneTimeMs = currentPhoneTimeMs,
                            offsetMs = offsetMs.toDouble(),
                            phoneAhead = phoneAhead
                        )
                    )
                    
                    updateNetworkStatus(true)
                } else {
                    println("✗ Clock sync failed: ${response.body()?.message}")
                    updateNetworkStatus(false)
                }
            } catch (e: Exception) {
                println("✗ Clock sync error: ${e.message}")
                e.printStackTrace()
                updateNetworkStatus(false)
            }
        }
    }
}

/**
 * UI state data class
 */
data class GridButtonUiState(
    val rows: Int,
    val cols: Int,
    val buttonStates: List<Boolean>, // true = pressed (black), false = unpressed (white)
    val networkStatus: Boolean, // true = connected, false = disconnected
    val clockSyncResult: ClockSyncResult? = null,
    val manualTestMode: Boolean = false
)

/**
 * Clock sync test result
 */
data class ClockSyncResult(
    val networkLatencyMs: Double,
    val serverTimeMs: Long,
    val phoneTimeMs: Long,
    val offsetMs: Double,
    val phoneAhead: Boolean
)


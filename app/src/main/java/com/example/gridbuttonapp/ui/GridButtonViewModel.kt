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
    private val manualMode: Boolean = false
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
            } else null
        )
    )
    val uiState: StateFlow<GridButtonUiState> = _uiState.asStateFlow()
    
    // Initialize recording API (will be set after updating base URL) when not in manual mode
    private val recordingApi: RecordingApi? =
        if (!manualMode) {
            NetworkModule.updateBaseUrl(apiUrl)
            NetworkModule.recordingApi.also {
                println("GridButtonViewModel created with API URL: $apiUrl, delay: $recordingDelayMs ms, manual=false")
                initializeAd3()
                setGraphVisualization(showGraphs)
                setDatasetMode(useTestDataset)
            }
        } else {
            println("GridButtonViewModel created in MANUAL mode (no API traffic).")
            null
        }
    
    /**
     * Initialize AD3 device
     */
    private fun initializeAd3() {
        val api = recordingApi ?: return
        viewModelScope.launch {
            try {
                println("Initializing AD3...")
                val response = api.initialize()
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
     * Handle button click using a two-tap capacitive-touch flow:
     *  - First tap: "prime" the button and start AD3 capture on the server.
     *  - Second tap on the same button: perform the actual press (flip to black,
     *    hold for recordingDelayMs, then release).
     */
    fun onButtonClick(buttonIndex: Int) {
        println("Button $buttonIndex clicked!")

        // In manual mode, just flip immediately and do not touch the API.
        if (manualMode) {
            viewModelScope.launch {
                updateButtonState(buttonIndex, true)
                delay(recordingDelayMs)
                updateButtonState(buttonIndex, false)
            }
            return
        }

        val currentPrimed = _uiState.value.primedButtonIndex
        when {
            currentPrimed == null -> {
                // First tap: prime this button and start AD3 capture.
                primeButton(buttonIndex)
            }
            currentPrimed == buttonIndex -> {
                // Second tap on the same button: perform the capacitive-touch press.
                performTouchPress(buttonIndex)
            }
            else -> {
                // Ignore taps on other buttons while one is primed.
                println("Ignoring tap on button $buttonIndex while button $currentPrimed is primed.")
            }
        }
    }

    /**
     * Send a priming request to the server to start AD3 capture for this button.
     * The response arrival is the cue for the user to perform the physical press.
     */
    private fun primeButton(buttonIndex: Int) {
        val api = recordingApi ?: return
        viewModelScope.launch {
            try {
                val timestamp = System.currentTimeMillis()
                val request = RecordingRequest(
                    button = buttonIndex,
                    rows = _uiState.value.rows,
                    cols = _uiState.value.cols,
                    timestamp = timestamp,
                    press_length_ms = recordingDelayMs,
                    network_latency_ms = 0.0
                )

                println("Sending capacitive-touch prime for button $buttonIndex...")
                val response = api.startRecording(request)

                if (response.isSuccessful && response.body()?.status == "success") {
                    updateNetworkStatus(true)
                    updatePrimedButton(buttonIndex)
                    println("Button $buttonIndex primed. Cue user to press on second tap.")
                } else {
                    updateNetworkStatus(false)
                    println("Failed to prime button $buttonIndex: ${response.body()?.message}")
                }
            } catch (e: Exception) {
                updateNetworkStatus(false)
                println("Error priming button $buttonIndex: ${e.message}")
                e.printStackTrace()
            }
        }
    }

    /**
     * Perform the actual capacitive-touch press: flip the button to black,
     * hold it for recordingDelayMs, then release back to white.
     */
    private fun performTouchPress(buttonIndex: Int) {
        viewModelScope.launch {
            try {
                println("Second tap for button $buttonIndex: starting capacitive-touch press.")
                // Clear primed state as soon as the press begins.
                updatePrimedButton(null)

                updateButtonState(buttonIndex, true)
                delay(recordingDelayMs)
                updateButtonState(buttonIndex, false)
                println("Button $buttonIndex reset after $recordingDelayMs ms")
            } catch (e: Exception) {
                println("Error during touch press for button $buttonIndex: ${e.message}")
                e.printStackTrace()
            }
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
     * Update which button (if any) is currently primed for capacitive-touch.
     */
    private fun updatePrimedButton(buttonIndex: Int?) {
        _uiState.value = _uiState.value.copy(primedButtonIndex = buttonIndex)
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
        val api = recordingApi ?: return
        viewModelScope.launch {
            try {
                val request = mapOf("enabled" to enabled)
                val response = api.toggleGraphs(request)
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
        val api = recordingApi ?: return
        viewModelScope.launch {
            try {
                val mode = if (useTestDataset) "test" else "train"
                val response = api.setDatasetMode(mapOf("mode" to mode))
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
     * Test clock synchronization with server
     */
    fun testClockSync() {
        val api = recordingApi ?: return
        viewModelScope.launch {
            try {
                println("\n═══ CLOCK SYNC TEST ═══")
                
                // Record time before sending request
                val tSendPhone = System.currentTimeMillis()
                println("T_send (phone): $tSendPhone ms")
                
                // Send ping request to server
                val response = api.ping()
                
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
    val primedButtonIndex: Int? = null // which button is currently primed for capacitive-touch, if any
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


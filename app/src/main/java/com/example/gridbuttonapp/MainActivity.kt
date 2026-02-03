package com.example.gridbuttonapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import com.example.gridbuttonapp.ui.GridButtonScreen
import com.example.gridbuttonapp.ui.GridButtonViewModel
import com.example.gridbuttonapp.ui.MenuScreen
import com.example.gridbuttonapp.ui.GridButtonUiState
import com.example.gridbuttonapp.ui.theme.GridButtonAppTheme
import kotlinx.coroutines.flow.MutableStateFlow

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            GridButtonAppTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    AppNavigation()
                }
            }
        }
    }
}

@Composable
fun AppNavigation() {
    var showMenu by remember { mutableStateOf(true) }
    var gridRows by remember { mutableStateOf(3) }
    var gridCols by remember { mutableStateOf(2) }
    var apiUrl by remember { mutableStateOf("http://10.0.0.92:5150/") }
    var recordingDelay by remember { mutableStateOf(100L) }
    var showGraphs by remember { mutableStateOf(false) }
    var useTestDataset by remember { mutableStateOf(false) }
    var manualTestMode by remember { mutableStateOf(false) }
    
    // Repeat sequence parameters - when set, grid screen will auto-run repeat
    var repeatButtonIndex by remember { mutableStateOf<Int?>(null) }
    var repeatCount by remember { mutableStateOf<Int?>(null) }
    
    // Create a shared viewModel for clock sync testing on menu screen
    // Use key to recreate when API URL changes
    val menuViewModel = remember(apiUrl, showGraphs, useTestDataset) {
        GridButtonViewModel(
            initialRows = gridRows,
            initialCols = gridCols,
            apiUrl = apiUrl,
            recordingDelay = recordingDelay,
            showGraphs = showGraphs,
            useTestDataset = useTestDataset
        )
    }
    
    val menuUiState by menuViewModel.uiState.collectAsState()
    
    if (showMenu) {
        
        MenuScreen(
            onNavigateToGrid = { showMenu = false },
            onGridSizeChange = { rows, cols ->
                gridRows = rows
                gridCols = cols
            },
            onApiUrlChange = { url ->
                apiUrl = url
            },
            onDelayChange = { delay ->
                recordingDelay = delay
                menuViewModel.updateRecordingDelay(delay)
            },
            onToggleGraphs = { enabled ->
                showGraphs = enabled
                menuViewModel.updateGraphPreference(enabled)
            },
            onToggleDataset = { enabled ->
                useTestDataset = enabled
                menuViewModel.updateDatasetMode(enabled)
            },
            onToggleManualTestMode = { enabled ->
                manualTestMode = enabled
            },
            onTestClockSync = {
                menuViewModel.testClockSync()
            },
            onStartRepeat = { buttonIndex, count ->
                println("Starting repeat: Button $buttonIndex, $count times")
                // Set repeat parameters and navigate to grid screen
                repeatButtonIndex = buttonIndex
                repeatCount = count
                showMenu = false  // Navigate to grid screen
            },
            currentRows = gridRows,
            currentCols = gridCols,
            currentApiUrl = apiUrl,
            currentDelay = recordingDelay,
            showGraphs = showGraphs,
            useTestDataset = useTestDataset,
            networkStatus = menuUiState.networkStatus,
            manualTestMode = manualTestMode,
            clockSyncResult = menuUiState.clockSyncResult
        )
    } else {
        println("Navigating to grid with delay: $recordingDelay ms")
        
        // Pass network latency from clock sync test to grid screen
        val networkLatency = menuUiState.clockSyncResult?.networkLatencyMs ?: 0.0
        if (networkLatency > 0) {
            println("Using network latency from clock sync: ${networkLatency}ms")
        }
        
        GridButtonScreen(
            onBackToMenu = { 
                // Clear repeat parameters when going back to menu
                repeatButtonIndex = null
                repeatCount = null
                showMenu = true 
            },
            initialRows = gridRows,
            initialCols = gridCols,
            apiUrl = apiUrl,
            recordingDelay = recordingDelay,
            showGraphs = showGraphs,
            useTestDataset = useTestDataset,
            networkLatencyMs = networkLatency,
            manualTestMode = manualTestMode,
            repeatButtonIndex = repeatButtonIndex,
            repeatCount = repeatCount,
            onRepeatComplete = {
                // Clear repeat parameters when sequence finishes
                repeatButtonIndex = null
                repeatCount = null
            }
        )
    }
}


package com.example.gridbuttonapp.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.itemsIndexed
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.runtime.remember

/**
 * Main screen composable that displays the grid of buttons
 */
@Composable
fun GridButtonScreen(
    onBackToMenu: () -> Unit,
    initialRows: Int = 3,
    initialCols: Int = 2,
    apiUrl: String = "http://10.0.0.92:5150/",
    recordingDelay: Long = 100L,
    showGraphs: Boolean = false,
    useTestDataset: Boolean = false,
    networkLatencyMs: Double = 0.0,
    manualTestMode: Boolean = false,
    repeatButtonIndex: Int? = null,
    repeatCount: Int? = null,
    onRepeatComplete: () -> Unit = {}
) {
    val viewModel = remember(initialRows, initialCols, apiUrl, recordingDelay, showGraphs, useTestDataset, networkLatencyMs, manualTestMode) { 
        println("Creating ViewModel with delay: $recordingDelay ms, graphs: $showGraphs, dataset: ${if (useTestDataset) "test" else "train"}, latency: ${networkLatencyMs}ms, manualTestMode: $manualTestMode")
        GridButtonViewModel(initialRows, initialCols, apiUrl, recordingDelay, showGraphs, useTestDataset, networkLatencyMs, manualTestMode) 
    }
    val uiState by viewModel.uiState.collectAsState()
    
    // Launch repeat sequence if parameters are provided
    LaunchedEffect(repeatButtonIndex, repeatCount) {
        if (repeatButtonIndex != null && repeatCount != null && repeatCount > 0) {
            println("GridButtonScreen: Starting repeat sequence for button $repeatButtonIndex, $repeatCount times")
            viewModel.repeatButtonPress(
                buttonIndex = repeatButtonIndex,
                count = repeatCount,
                onComplete = {
                    onRepeatComplete()
                }
            )
        }
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
    ) {
        // Back button and header
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                text = "← Back",
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                color = Color(0xFF2196F3),
                modifier = Modifier.clickable { onBackToMenu() }
            )
            Spacer(modifier = Modifier.weight(1f))
            Text(
                text = "Grid: ${uiState.rows}×${uiState.cols}",
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium,
                color = Color.Black
            )
        }
        
        // Network status
        Text(
            text = "Network: ${if (uiState.networkStatus) "Connected" else "Disconnected"}",
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 4.dp),
            fontSize = 14.sp,
            fontWeight = FontWeight.Normal,
            color = if (uiState.networkStatus) Color(0xFF00AA00) else Color(0xFFAA0000)
        )
        
        // Manual test mode status
        Text(
            text = "Manual test mode: ${if (uiState.manualTestMode) "On" else "Off"}",
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 2.dp),
            fontSize = 14.sp,
            fontWeight = FontWeight.Normal,
            color = Color.Black
        )
        
        // Grid of buttons - takes remaining space
        Column(
            modifier = Modifier
                .weight(1f)
                .padding(horizontal = 4.dp, vertical = 4.dp)
        ) {
            repeat(uiState.rows) { row ->
                Row(
                    modifier = Modifier.weight(1f)
                ) {
                    repeat(uiState.cols) { col ->
                        val index = row * uiState.cols + col
                        val isPressed = uiState.buttonStates[index]
                        val backgroundColor = if (isPressed) Color(0xFF000000) else Color(0xFFFFFFFF)
                        
                        Box(
                            modifier = Modifier
                                .weight(1f)
                                .fillMaxHeight()
                                .background(backgroundColor)
                                .clickable(
                                    interactionSource = remember { MutableInteractionSource() },
                                    indication = null
                                ) { 
                                    viewModel.onButtonClick(index) 
                                }
                        )
                    }
                }
            }
        }
    }
}



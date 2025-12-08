package com.example.gridbuttonapp.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * Menu screen with network status and grid size controls
 */
@Composable
fun MenuScreen(
    onNavigateToGrid: () -> Unit,
    onGridSizeChange: (rows: Int, cols: Int) -> Unit,
    onApiUrlChange: (String) -> Unit,
    onDelayChange: (Long) -> Unit,
    onToggleGraphs: (Boolean) -> Unit,
    onToggleDataset: (Boolean) -> Unit,
    onTestClockSync: () -> Unit,
    currentRows: Int,
    currentCols: Int,
    currentApiUrl: String,
    currentDelay: Long,
    showGraphs: Boolean,
    useTestDataset: Boolean,
    networkStatus: Boolean,
    clockSyncResult: ClockSyncResult? = null
) {
    var tempRows by remember { mutableStateOf(currentRows.toString()) }
    var tempCols by remember { mutableStateOf(currentCols.toString()) }
    var tempApiUrl by remember { mutableStateOf(currentApiUrl) }
    var tempDelay by remember { mutableStateOf(currentDelay.toString()) }
    var graphsEnabled by remember { mutableStateOf(showGraphs) }
    var datasetTestEnabled by remember { mutableStateOf(useTestDataset) }

    LaunchedEffect(showGraphs) {
        graphsEnabled = showGraphs
    }

    LaunchedEffect(useTestDataset) {
        datasetTestEnabled = useTestDataset
    }
    
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.White)
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Title
        Text(
            text = "Grid Button App",
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold,
            color = Color.Black,
            modifier = Modifier.padding(vertical = 24.dp)
        )
        
        // Network Status Card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp),
            colors = CardDefaults.cardColors(
                containerColor = if (networkStatus) Color(0xFFE8F5E8) else Color(0xFFFFE8E8)
            )
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Network Status",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.Black
                )
                Text(
                    text = if (networkStatus) "Connected" else "Disconnected",
                    fontSize = 16.sp,
                    color = if (networkStatus) Color(0xFF00AA00) else Color(0xFFAA0000),
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }
        
        // Grid Size Controls Card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Grid Size",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.Black,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceEvenly
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "Rows",
                            fontSize = 14.sp,
                            color = Color.Black,
                            modifier = Modifier.padding(bottom = 8.dp)
                        )
                        OutlinedTextField(
                            value = tempRows,
                            onValueChange = { tempRows = it },
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            modifier = Modifier.width(80.dp),
                            singleLine = true
                        )
                    }
                    
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "Columns",
                            fontSize = 14.sp,
                            color = Color.Black,
                            modifier = Modifier.padding(bottom = 8.dp)
                        )
                        OutlinedTextField(
                            value = tempCols,
                            onValueChange = { tempCols = it },
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            modifier = Modifier.width(80.dp),
                            singleLine = true
                        )
                    }
                }
                
                Button(
                    onClick = {
                        val rows = tempRows.toIntOrNull() ?: 3
                        val cols = tempCols.toIntOrNull() ?: 2
                        if (rows > 0 && cols > 0) {
                            onGridSizeChange(rows, cols)
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 16.dp)
                ) {
                    Text("Update Grid Size")
                }
            }
        }
        
        // API Configuration Card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "API Configuration",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.Black,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                
                // API URL Input
                Text(
                    text = "Server URL",
                    fontSize = 14.sp,
                    color = Color.Black,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                OutlinedTextField(
                    value = tempApiUrl,
                    onValueChange = { tempApiUrl = it },
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    placeholder = { Text("http://10.0.2.2:5150/") }
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Recording Delay Input
                Text(
                    text = "Button Press Duration (ms)",
                    fontSize = 14.sp,
                    color = Color.Black,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                OutlinedTextField(
                    value = tempDelay,
                    onValueChange = { tempDelay = it },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    placeholder = { Text("1000") }
                )
                
                Button(
                    onClick = {
                        onApiUrlChange(tempApiUrl)
                        val delay = tempDelay.toLongOrNull() ?: 1000L
                        println("Menu: Updating delay to: $delay ms")
                        onDelayChange(delay)
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 16.dp)
                ) {
                    Text("Update API Settings")
                }
            }
        }
        
        // Visualization Options Card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Visualization",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.Black,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Show graphs on computer after recording",
                        fontSize = 14.sp,
                        color = Color.Black,
                        modifier = Modifier.weight(1f)
                    )
                    Switch(
                        checked = graphsEnabled,
                        onCheckedChange = { enabled ->
                            graphsEnabled = enabled
                            onToggleGraphs(enabled)
                        }
                    )
                }
            }
        }

        // Dataset Selection Card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Dataset",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.Black,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Change recordings to save to test dataset",
                        fontSize = 14.sp,
                        color = Color.Black,
                        modifier = Modifier.weight(1f)
                    )
                    Switch(
                        checked = datasetTestEnabled,
                        onCheckedChange = { enabled ->
                            datasetTestEnabled = enabled
                            onToggleDataset(enabled)
                        }
                    )
                }
            }
        }
        
        // Clock Sync Test Card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Clock Synchronization Test",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.Black,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                
                Button(
                    onClick = onTestClockSync,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text("Test Clock Sync")
                }
                
                // Display results if available
                clockSyncResult?.let { result ->
                    Spacer(modifier = Modifier.height(16.dp))
                    
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        colors = CardDefaults.cardColors(
                            containerColor = Color(0xFFF5F5F5)
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(12.dp)
                        ) {
                            Text(
                                text = "Sync Results",
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold,
                                color = Color.Black,
                                modifier = Modifier.padding(bottom = 8.dp)
                            )
                            
                            // Network latency
                            Text(
                                text = "Network latency: ${"%.1f".format(result.networkLatencyMs)} ms",
                                fontSize = 14.sp,
                                color = Color.Black
                            )
                            
                            // Server time
                            val serverTimeFormatted = java.text.SimpleDateFormat("HH:mm:ss.SSS", java.util.Locale.US)
                                .format(java.util.Date(result.serverTimeMs))
                            Text(
                                text = "Server time: $serverTimeFormatted",
                                fontSize = 14.sp,
                                color = Color.Black,
                                modifier = Modifier.padding(top = 4.dp)
                            )
                            
                            // Phone time
                            val phoneTimeFormatted = java.text.SimpleDateFormat("HH:mm:ss.SSS", java.util.Locale.US)
                                .format(java.util.Date(result.phoneTimeMs))
                            Text(
                                text = "Phone time: $phoneTimeFormatted",
                                fontSize = 14.sp,
                                color = Color.Black,
                                modifier = Modifier.padding(top = 4.dp)
                            )
                            
                            // Clock offset
                            val offsetSign = if (result.phoneAhead) "+" else ""
                            val offsetDirection = if (result.phoneAhead) "ahead" else "behind"
                            Text(
                                text = "Estimated offset: $offsetSign${"%.1f".format(result.offsetMs)} ms",
                                fontSize = 14.sp,
                                fontWeight = FontWeight.Bold,
                                color = if (Math.abs(result.offsetMs) > 100) Color(0xFFFF6600) else Color(0xFF00AA00),
                                modifier = Modifier.padding(top = 4.dp)
                            )
                            Text(
                                text = "Phone is ${"%.1f".format(Math.abs(result.offsetMs))} ms $offsetDirection",
                                fontSize = 12.sp,
                                color = Color.Gray,
                                modifier = Modifier.padding(top = 2.dp)
                            )
                        }
                    }
                }
            }
        }
        
        // Current Grid Info
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Current Settings",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.Black
                )
                Text(
                    text = "Grid: ${currentRows} × ${currentCols} = ${currentRows * currentCols} buttons",
                    fontSize = 16.sp,
                    color = Color.Black,
                    modifier = Modifier.padding(top = 4.dp)
                )
                Text(
                    text = "API: ${currentApiUrl}",
                    fontSize = 14.sp,
                    color = Color.Gray,
                    modifier = Modifier.padding(top = 2.dp)
                )
                Text(
                    text = "Delay: ${currentDelay}ms",
                    fontSize = 14.sp,
                    color = Color.Gray,
                    modifier = Modifier.padding(top = 2.dp)
                )
                Text(
                    text = "Graphs: ${if (showGraphs) "Enabled" else "Disabled"}",
                    fontSize = 14.sp,
                    color = Color.Gray,
                    modifier = Modifier.padding(top = 2.dp)
                )
                Text(
                    text = "Dataset: ${if (useTestDataset) "Test" else "Train"}",
                    fontSize = 14.sp,
                    color = Color.Gray,
                    modifier = Modifier.padding(top = 2.dp)
                )
            }
        }
        
        Spacer(modifier = Modifier.weight(1f))
        
        // Start Button
        Button(
            onClick = onNavigateToGrid,
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF2196F3)
            )
        ) {
            Text(
                text = "Start Grid",
                fontSize = 18.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

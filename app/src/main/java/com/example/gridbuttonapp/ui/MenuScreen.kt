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
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MenuScreen(
    onNavigateToGrid: () -> Unit,
    onGridSizeChange: (rows: Int, cols: Int) -> Unit,
    onApiUrlChange: (String) -> Unit,
    onDelayChange: (Long) -> Unit,
    onToggleGraphs: (Boolean) -> Unit,
    onToggleDataset: (Boolean) -> Unit,
    onToggleManualTestMode: (Boolean) -> Unit,
    onTestClockSync: () -> Unit,
    onStartRepeat: (buttonIndex: Int, count: Int) -> Unit,
    currentRows: Int,
    currentCols: Int,
    currentApiUrl: String,
    currentDelay: Long,
    showGraphs: Boolean,
    useTestDataset: Boolean,
    networkStatus: Boolean,
    manualTestMode: Boolean,
    clockSyncResult: ClockSyncResult? = null
) {
    var tempRows by remember { mutableStateOf(currentRows.toString()) }
    var tempCols by remember { mutableStateOf(currentCols.toString()) }
    var tempApiUrl by remember { mutableStateOf(currentApiUrl) }
    var tempDelay by remember { mutableStateOf(currentDelay.toString()) }
    var graphsEnabled by remember { mutableStateOf(showGraphs) }
    var datasetTestEnabled by remember { mutableStateOf(useTestDataset) }
    var manualTestEnabled by remember { mutableStateOf(manualTestMode) }
    
    // Repeat feature state
    var selectedButtonIndex by remember { mutableStateOf(0) }
    var repeatCount by remember { mutableStateOf("1") }
    var expandedButtonDropdown by remember { mutableStateOf(false) }

    LaunchedEffect(showGraphs) {
        graphsEnabled = showGraphs
    }

    LaunchedEffect(useTestDataset) {
        datasetTestEnabled = useTestDataset
    }
    
    LaunchedEffect(manualTestMode) {
        manualTestEnabled = manualTestMode
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
        
        // Manual Test Mode Card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Manual Test Mode",
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
                        text = "Pressing grid buttons will flip them immediately\nwithout sending any requests to the server.",
                        fontSize = 14.sp,
                        color = Color.Black,
                        modifier = Modifier.weight(1f)
                    )
                    Switch(
                        checked = manualTestEnabled,
                        onCheckedChange = { enabled ->
                            manualTestEnabled = enabled
                            onToggleManualTestMode(enabled)
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
        
        // Repeat Button Presses Card
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Repeat Button Presses",
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.Black,
                    modifier = Modifier.padding(bottom = 16.dp)
                )
                
                // Button Selection Dropdown
                Text(
                    text = "Button to Repeat",
                    fontSize = 14.sp,
                    color = Color.Black,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                
                val totalButtons = currentRows * currentCols
                val buttonOptions = (0 until totalButtons).map { "Button $it" }
                
                ExposedDropdownMenuBox(
                    expanded = expandedButtonDropdown,
                    onExpandedChange = { expandedButtonDropdown = !expandedButtonDropdown }
                ) {
                    OutlinedTextField(
                        value = buttonOptions[selectedButtonIndex],
                        onValueChange = {},
                        readOnly = true,
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expandedButtonDropdown) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .menuAnchor()
                    )
                    ExposedDropdownMenu(
                        expanded = expandedButtonDropdown,
                        onDismissRequest = { expandedButtonDropdown = false }
                    ) {
                        buttonOptions.forEachIndexed { index, label ->
                            DropdownMenuItem(
                                text = { Text(label) },
                                onClick = {
                                    selectedButtonIndex = index
                                    expandedButtonDropdown = false
                                }
                            )
                        }
                    }
                }
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Repeat Count Input
                Text(
                    text = "Number of Times",
                    fontSize = 14.sp,
                    color = Color.Black,
                    modifier = Modifier.padding(bottom = 8.dp)
                )
                OutlinedTextField(
                    value = repeatCount,
                    onValueChange = { repeatCount = it },
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                    modifier = Modifier.fillMaxWidth(),
                    singleLine = true,
                    placeholder = { Text("100") }
                )
                
                Spacer(modifier = Modifier.height(16.dp))
                
                // Start Repeat Button
                Button(
                    onClick = {
                        val count = repeatCount.toIntOrNull() ?: 1
                        if (count > 0 && selectedButtonIndex >= 0 && selectedButtonIndex < totalButtons) {
                            onStartRepeat(selectedButtonIndex, count)
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFFFF9800)
                    )
                ) {
                    Text(
                        text = "Start Repeat Sequence",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
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

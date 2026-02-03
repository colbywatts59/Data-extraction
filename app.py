from flask import Flask, request, jsonify
import os
import json
from datetime import datetime
import logging
import time
import threading
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Digilent WaveForms SDK imports
try:
    from pydwf import DwfLibrary, DwfAcquisitionMode, DwfTriggerSource, DwfState
    AD3_AVAILABLE = True
except ImportError:
    AD3_AVAILABLE = False
    logging.warning("Digilent WaveForms SDK not available. Install with: pip install pydwf")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
recording_state = {}
ad3_device = None
ad3_library = None
ad3_configured = False  # Track if AD3 is connected and configured
trace_counters = {}  # Persistent trace counters for each button
show_graphs = False  # Toggle for showing graphs after recording
dataset_mode = 'train'  # 'train' or 'test'

# Sampling and synchronization configuration
SAMPLE_RATE_HZ = 20000       # Oscilloscope sampling rate (Hz)
CHANNEL_RANGE_V = 0.5        # ~500 mV/div equivalent full-scale range
RECORD_SAMPLES = 32768       # Fixed number of samples per trace (~1638ms at 20kHz, ~655ms at 50kHz)

# ===== TIMING CONFIGURATION =====
# 
# HOW IT WORKS:
#   1. Server receives request and starts recording thread
#   2. Recording thread waits (calculated delay) before starting AD3
#   3. Server waits STARTUP_BUFFER_MS then sends response
#   4. Phone waits PHONE_DELAY_MS then presses
#   5. Press appears at PRESS_POSITION_MS in the recording
#
# YOU ONLY NEED TO CHANGE THESE:
PRESS_POSITION_MS = 200        # Where you want the press to appear in the recording
STARTUP_BUFFER_MS = 100        # Extra buffer to ensure AD3 is ready (increase if presses are missing)

# These are calculated automatically - don't change unless debugging
PHONE_DELAY_MS = 700           # Fixed delay phone waits after receiving response
# server_wait_ms is calculated in the /record endpoint

# Initialize AD3 if available
if AD3_AVAILABLE:
    try:
        ad3_library = DwfLibrary()
        logger.info("WaveForms SDK initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize WaveForms SDK: {e}")
        AD3_AVAILABLE = False


# Create the recording directory structure if it doesn't exist
def create_recording_directory(grid_rows, grid_cols, button):
    directory = os.path.join(dataset_mode, f"{grid_rows}x{grid_cols}", f"button_{button}")
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Created directory: {directory}")
    return directory

# Initialize and connect to AD3 device once at startup
def initialize_ad3():
    global ad3_device, ad3_configured
    
    if not AD3_AVAILABLE:
        logger.warning("AD3 not available - cannot initialize")
        return False
    
    if ad3_configured:
        logger.info("AD3 already initialized and configured")
        return True
    
    try:
        device_enum = ad3_library.deviceEnum
        device_enum.enumerateStart()
        device_count = device_enum.enumerateDevices()
        if device_count == 0:
            device_enum.enumerateStop()
            raise Exception("No Analog Discovery devices found")
        
        # Connect to first device
        ad3_device = ad3_library.deviceControl.open(0)
        device_enum.enumerateStop()
        logger.info("Connected to Analog Discovery 3")
        
        # Configure oscilloscope once
        configure_oscilloscope()
        ad3_configured = True
        logger.info("AD3 initialized and configured successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize AD3: {e}")
        ad3_configured = False
        return False

# Configure oscilloscope for channel 1 recording
def configure_oscilloscope():
    global ad3_device
    
    if not ad3_device:
        raise Exception("AD3 device not connected")
    
    try:
        ad3_device.analogIn.channelEnableSet(0, True)  # Channel 1
        ad3_device.analogIn.channelRangeSet(0, CHANNEL_RANGE_V)
        ad3_device.analogIn.channelOffsetSet(0, 0.0)   # No offset
        
        # Configure acquisition parameters
        ad3_device.analogIn.frequencySet(SAMPLE_RATE_HZ)
        ad3_device.analogIn.bufferSizeSet(RECORD_SAMPLES)
        
        # Use single acquisition mode
        ad3_device.analogIn.acquisitionModeSet(DwfAcquisitionMode.Single)
        ad3_device.analogIn.triggerAutoTimeoutSet(0)  # No trigger timeout
        ad3_device.analogIn.triggerSourceSet(DwfTriggerSource.None_)  # No trigger
        
        logger.info("Oscilloscope configured for channel 1 (single acquisition mode)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to configure oscilloscope: {e}")
        raise


# Start oscilloscope acquisition using fixed buffer size
def start_acquisition():
    global ad3_device
    
    if not ad3_device:
        raise Exception("AD3 device not connected")
    

    # THIS MIGHT BE UNNECESSARY****************************************************
    try:
        ad3_device.analogIn.channelEnableSet(0, True)
        ad3_device.analogIn.channelRangeSet(0, CHANNEL_RANGE_V)
        ad3_device.analogIn.channelOffsetSet(0, 0.0)
        ad3_device.analogIn.frequencySet(SAMPLE_RATE_HZ)
        
        # Use fixed buffer size for each recording
        samples_needed = RECORD_SAMPLES 
        ad3_device.analogIn.bufferSizeSet(samples_needed)
        
        # Set acquisition mode
        ad3_device.analogIn.acquisitionModeSet(DwfAcquisitionMode.Single)
        ad3_device.analogIn.triggerAutoTimeoutSet(0)
        ad3_device.analogIn.triggerSourceSet(DwfTriggerSource.None_)
        
        total_duration_ms = samples_needed / float(SAMPLE_RATE_HZ) * 1000.0
        logger.info(f"Starting acquisition: ~{total_duration_ms:.1f}ms, "
                    f"{samples_needed} samples at {SAMPLE_RATE_HZ}Hz")
        
        # Start single acquisition
        ad3_device.analogIn.configure(False, True)
        
        # Wait for acquisition to complete
        while True:
            status = ad3_device.analogIn.status(True)
            if status == DwfState.Done:
                break
            time.sleep(0.001)  # Check every 1ms
        
        logger.info("Acquisition completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed during acquisition: {e}")
        raise


# Retrieve waveform data from AD3 and construct a correctly scaled time axis
def get_waveform_data():
    global ad3_device
    
    if not ad3_device:
        raise Exception("AD3 device not connected")
    
    try:
        # Get the actual buffer size that was used
        buffer_size = ad3_device.analogIn.bufferSizeGet()
        logger.info(f"Buffer size: {buffer_size}")
        
        # Get channel 1 data - read the full buffer
        channel_data = ad3_device.analogIn.statusData(0, buffer_size)
        
        data_array = np.array(channel_data)
        
        if len(data_array) == 0:
            raise Exception("No data retrieved from AD3 - check signal connection")
        
        # Get timing information from sample count and sample rate
        total_duration_s = len(data_array) / float(SAMPLE_RATE_HZ)
        time_points = np.linspace(
            0,
            total_duration_s,
            len(data_array),
            endpoint=False
        ) * 1000.0  # convert to ms

        # Log some sample data for debugging
        # logger.info(f"Retrieved {len(data_array)} data points")
        # logger.info(f"First 5 voltage values: {data_array[:5]}")
        # logger.info(f"Last 5 voltage values: {data_array[-5:]}")
        # logger.info(f"Voltage range: {np.min(data_array):.6f}V to {np.max(data_array):.6f}V")
        
        return time_points, data_array
        
    except Exception as e:
        logger.error(f"Failed to retrieve waveform data: {e}")
        raise


# Disconnect from AD3 device
def disconnect_ad3():
    global ad3_device
    
    if ad3_device:
        try:
            ad3_device.close()
            ad3_device = None
            logger.info("Disconnected from Analog Discovery 3")
        except Exception as e:
            logger.error(f"Error disconnecting from AD3: {e}")


# Load trace counters from persistent storage
def load_trace_counters():
    global trace_counters
    
    try:
        if os.path.exists('trace_counters.json'):
            with open('trace_counters.json', 'r') as f:
                trace_counters = json.load(f)
            logger.info(f"Loaded trace counters: {trace_counters}")
        else:
            trace_counters = {}
            logger.info("No existing trace counters found, starting fresh")
    except Exception as e:
        logger.error(f"Error loading trace counters: {e}")
        trace_counters = {}

# Save trace counters to persistent storage
def save_trace_counters():
    global trace_counters
    
    try:
        with open('trace_counters.json', 'w') as f:
            json.dump(trace_counters, f)
        logger.info(f"Saved trace counters: {trace_counters}")
    except Exception as e:
        logger.error(f"Error saving trace counters: {e}")

# Get the next trace number for a button with persistent storage
def get_next_trace_number(button_key):
    global trace_counters
    
    # Initialize counter for this button if not exists
    if button_key not in trace_counters:
        trace_counters[button_key] = 0
    
    # Increment and save
    trace_counters[button_key] += 1
    save_trace_counters()
    
    return trace_counters[button_key]

# Record waveform from AD3 for specified duration
def record_waveform(press_length_ms):
    global ad3_configured
    
    if not ad3_configured:
        raise Exception("AD3 not initialized. Call initialize_ad3() first.")
    
    try:
        # Use a fixed-length recording window (matches standalone AD3 script)
        recording_duration_ms = RECORD_SAMPLES / float(SAMPLE_RATE_HZ) * 1000.0
        logger.info(
            f"Starting recording (~{recording_duration_ms:.1f}ms window) "
            f"for press_length_ms={press_length_ms}"
        )
        
        # Start acquisition 
        start_acquisition()
        
        # Wait a moment for data to be processed
        time.sleep(0.1)
        
        # Get waveform data with correctly scaled time axis
        time_points, voltage_data = get_waveform_data()
        
        # Log some data statistics for debugging
        if len(voltage_data) > 0:
            min_voltage = np.min(voltage_data)
            max_voltage = np.max(voltage_data)
            mean_voltage = np.mean(voltage_data)
            logger.info(f"Data stats - Min: {min_voltage:.3f}V, Max: {max_voltage:.3f}V, Mean: {mean_voltage:.3f}V")
        else:
            logger.warning("No voltage data captured!")
        
        logger.info(f"Successfully recorded {len(voltage_data)} data points")
        return time_points, voltage_data
        
    except Exception as e:
        logger.error(f"Error during waveform recording: {e}")
        raise

# Save waveform data to CSV file
def save_waveform_to_csv(time_points, voltage_data, filepath):
    try:
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['time_ms', 'voltage'])
            
            # Write data points
            for time_ms, voltage in zip(time_points, voltage_data):
                writer.writerow([time_ms, voltage])
        
        logger.info(f"Saved waveform data to {filepath}")
        return True
            
    except Exception as e:
        logger.error(f"Error saving CSV file {filepath}: {e}")
        raise

# =============================================================================
# RISING EDGE ALIGNMENT FOR BUTTON PRESS EXTRACTION
# =============================================================================
#
# For training classifiers on button press signals, ALIGNMENT IS CRITICAL.
# If traces are not consistently aligned, the same feature index corresponds
# to different parts of the signal (baseline vs rising edge vs peak), making
# classification much harder.
#
# RISING EDGE ALIGNMENT:
# - Finds the main peak first (highest smoothed value)
# - Works backwards from peak to find where signal crosses threshold
# - Window starts at a fixed offset BEFORE the rising edge
# - Provides the most consistent alignment for classification
# - Rising edge, peak, and falling edge appear at predictable positions
#
# =============================================================================

def align_by_rising_edge(
    times: np.ndarray,
    volts: np.ndarray,
    coarse_start_idx: int,
    coarse_end_idx: int,
    baseline_val: float,
    press_length_ms: float,
    sample_rate_hz: float,
    smoothing_window_ms: float = 10.0,
    threshold_factor: float = 0.2,  # Rising edge = baseline + 20% of (peak - baseline)
    pre_edge_ms: float = 5.0  # Start window this many ms before rising edge
) -> tuple:
    """
    Align button press extraction to the RISING EDGE of the signal.
    
    This provides consistent alignment for training classifiers because:
    - The rising edge is a well-defined, consistent reference point
    - All traces will have rising edge, peak, and falling edge at predictable positions
    - Better feature correspondence across training samples
    
    Parameters:
    -----------
    times : np.ndarray
        Full time array in milliseconds
    volts : np.ndarray
        Full voltage array
    coarse_start_idx : int
        Start index of coarse window containing the button press
    coarse_end_idx : int
        End index of coarse window containing the button press
    baseline_val : float
        Estimated baseline voltage (from pre-press region)
    press_length_ms : float
        Desired output window length in milliseconds
    sample_rate_hz : float
        Sampling rate in Hz
    smoothing_window_ms : float
        Width of moving average smoothing window in ms (default 10ms)
    threshold_factor : float
        Rising edge threshold as fraction of (peak - baseline) above baseline (default 0.2 = 20%)
    pre_edge_ms : float
        How many ms before the rising edge to start the window (default 5ms)
    
    Returns:
    --------
    tuple: (edge_idx, edge_time_ms, aligned_start_idx, aligned_end_idx)
           or None if rising edge cannot be found
    """
    n = len(volts)
    samples_per_ms = sample_rate_hz / 1000.0
    
    # Validate indices
    coarse_start_idx = max(0, coarse_start_idx)
    coarse_end_idx = min(n - 1, coarse_end_idx)
    
    if coarse_end_idx <= coarse_start_idx:
        logger.warning("align_by_rising_edge: invalid coarse window")
        return None
    
    # Step 1: Extract and smooth the coarse window
    coarse_volts = volts[coarse_start_idx:coarse_end_idx + 1]
    
    smoothing_samples = int(smoothing_window_ms * samples_per_ms)
    if smoothing_samples < 3:
        smoothing_samples = 3
    if smoothing_samples % 2 == 0:
        smoothing_samples += 1
    
    kernel = np.ones(smoothing_samples) / smoothing_samples
    smoothed = np.convolve(coarse_volts, kernel, mode='same')
    
    # Step 2: Find the MAIN PEAK first (highest point in smoothed signal)
    # This ensures we find the actual button press, not noise spikes
    peak_local_idx = np.argmax(smoothed)
    peak_val = smoothed[peak_local_idx]
    
    if peak_val <= baseline_val:
        logger.warning("align_by_rising_edge: no signal above baseline")
        return None
    
    # Step 3: Calculate threshold for rising edge detection
    # Threshold = baseline + threshold_factor * (peak - baseline)
    threshold = baseline_val + threshold_factor * (peak_val - baseline_val)
    
    # Step 4: Work BACKWARDS from the peak to find the rising edge
    # This guarantees we find the rising edge of the ACTUAL button press,
    # not some earlier noise spike
    rising_edge_local_idx = 0  # Default to start if not found
    for i in range(peak_local_idx, -1, -1):
        if smoothed[i] <= threshold:
            rising_edge_local_idx = i + 1  # First point above threshold
            break
    
    # Clamp to valid range
    rising_edge_local_idx = max(0, min(rising_edge_local_idx, len(smoothed) - 1))
    rising_edge_idx = coarse_start_idx + rising_edge_local_idx
    rising_edge_time_ms = times[rising_edge_idx]
    
    logger.info(
        f"Rising edge detection: peak at local idx {peak_local_idx}, "
        f"rising edge at local idx {rising_edge_local_idx}, threshold={threshold:.4f}V"
    )
    
    # Step 5: Compute window starting pre_edge_ms before rising edge
    # Add 10% buffer to press_length_ms
    window_duration_ms = press_length_ms * 1.10
    total_window_samples = int(window_duration_ms * samples_per_ms)
    pre_edge_samples = int(pre_edge_ms * samples_per_ms)
    
    aligned_start_idx = rising_edge_idx - pre_edge_samples
    aligned_end_idx = aligned_start_idx + total_window_samples
    
    # Handle edge cases: window bounds
    if aligned_start_idx < 0:
        aligned_start_idx = 0
        aligned_end_idx = min(n - 1, aligned_start_idx + total_window_samples)
    elif aligned_end_idx >= n:
        aligned_end_idx = n - 1
        aligned_start_idx = max(0, aligned_end_idx - total_window_samples)
    
    logger.info(
        f"Rising edge alignment: edge at {rising_edge_time_ms:.2f}ms, "
        f"threshold={threshold:.4f}V ({threshold_factor*100:.0f}% above baseline), "
        f"window=[{times[aligned_start_idx]:.2f}, {times[aligned_end_idx]:.2f}]ms"
    )
    
    return rising_edge_idx, rising_edge_time_ms, aligned_start_idx, aligned_end_idx


# Extract the main peak segment after the button press and save it separately
# Uses rising edge alignment for consistent classifier training
def extract_peak_segment(
    time_points,
    voltage_data,
    button_press_time_ms,
    press_length_ms,  # Expected duration of button press
    pre_search_buffer_ms=50.0,  # Look this much before expected press
    post_search_buffer_ms=50.0,  # Look this much after expected press ends
    baseline_window_ms=200.0,
    baseline_gap_ms=50.0,
    peak_threshold_factor=0.15  # Peak must be at least 15% higher than baseline
):
    """
    Extract button press segment using CENTROID-BASED ALIGNMENT.
    
    Algorithm:
    1. Estimate baseline from pre-press region
    2. Define coarse search window around expected press time
    3. Verify signal is above threshold (has a button press)
    4. Use centroid alignment to find the "center of energy" of the press
    5. Extract a fixed-length window centered on the centroid
    
    Centroid alignment is preferred over peak alignment because:
    - It's robust to noise spikes that can fool peak detection
    - It considers the entire shape/energy of the button press
    - It provides consistent alignment even with variable peak heights
    """
    if len(time_points) == 0 or len(time_points) != len(voltage_data):
        logger.warning("extract_peak_segment: empty or mismatched data, skipping")
        return None
    
    times = np.asarray(time_points)
    volts = np.asarray(voltage_data)
    n = len(volts)
    samples_per_ms = SAMPLE_RATE_HZ / 1000.0
    
    # Step 1: Estimate baseline from a window before the press
    # Using median is more robust to outliers than mean
    baseline_start = button_press_time_ms - baseline_window_ms - baseline_gap_ms
    baseline_end = button_press_time_ms - baseline_gap_ms
    baseline_mask = (times >= baseline_start) & (times <= baseline_end)
    
    if np.any(baseline_mask):
        baseline_val = float(np.median(volts[baseline_mask]))
    else:
        baseline_val = float(np.percentile(volts, 10))
    
    # Step 2: Define coarse search window containing the button press
    # This should be wider than press_length_ms to ensure we capture the full event
    search_start = button_press_time_ms - pre_search_buffer_ms
    search_end = button_press_time_ms + press_length_ms + post_search_buffer_ms
    search_mask = (times >= search_start) & (times <= search_end)
    search_indices = np.where(search_mask)[0]
    
    if len(search_indices) == 0:
        logger.warning("extract_peak_segment: no samples in search window")
        return None
    
    coarse_start_idx = search_indices[0]
    coarse_end_idx = search_indices[-1]
    
    # Step 3: Verify there's a valid signal (quick check using smoothed max)
    smoothing_window_ms = 10.0
    smoothing_samples = max(3, int(smoothing_window_ms * samples_per_ms))
    if smoothing_samples % 2 == 0:
        smoothing_samples += 1
    
    kernel = np.ones(smoothing_samples) / smoothing_samples
    smoothed_volts = np.convolve(volts, kernel, mode='same')
    
    search_smoothed = smoothed_volts[coarse_start_idx:coarse_end_idx + 1]
    max_smoothed_val = np.max(search_smoothed)
    
    peak_threshold = baseline_val * (1.0 + peak_threshold_factor)
    if max_smoothed_val < peak_threshold:
        logger.warning(
            f"extract_peak_segment: signal too low ({max_smoothed_val:.4f}V) compared to baseline "
            f"({baseline_val:.4f}V), threshold={peak_threshold:.4f}V"
        )
        return None
    
    # Step 4: Use RISING EDGE ALIGNMENT for consistent classifier training
    # This ensures all traces have their rising edge, peak, and falling edge
    # at the same positions, which is critical for classification accuracy.
    alignment_result = align_by_rising_edge(
        times=times,
        volts=volts,
        coarse_start_idx=coarse_start_idx,
        coarse_end_idx=coarse_end_idx,
        baseline_val=baseline_val,
        press_length_ms=press_length_ms,
        sample_rate_hz=SAMPLE_RATE_HZ,
        smoothing_window_ms=10.0,
        threshold_factor=0.2,  # Rising edge = 20% of the way from baseline to peak
        pre_edge_ms=5.0  # Start window 5ms before rising edge
    )
    
    if alignment_result is None:
        logger.warning("extract_peak_segment: rising edge alignment failed")
        return None
    
    ref_idx, ref_time_ms, start_idx, end_idx = alignment_result
    
    # Step 5: Extract the segment
    segment_times = times[start_idx:end_idx + 1]
    segment_volts = volts[start_idx:end_idx + 1]
    
    if len(segment_times) == 0:
        logger.warning("extract_peak_segment: empty segment after extraction")
        return None
    
    start_time_ms = float(segment_times[0])
    end_time_ms = float(segment_times[-1])
    duration_ms = end_time_ms - start_time_ms
    
    # Find the actual peak value for logging
    peak_val = float(np.max(segment_volts))
    
    logger.info(
        f"Extracted segment: [{start_time_ms:.2f}ms - {end_time_ms:.2f}ms] "
        f"(duration: {duration_ms:.2f}ms, {len(segment_times)} samples, "
        f"alignment ref: {ref_time_ms:.2f}ms, max voltage: {peak_val:.4f}V, baseline: {baseline_val:.4f}V)"
    )
    
    return segment_times, segment_volts, start_time_ms, end_time_ms

# Visualize the waveform and save it as a PNG image
def visualize_waveform(
    time_points,
    voltage_data,
    title="Waveform",
    filepath=None,
    button_press_time_ms=None,
    highlight_start_ms=None,
    highlight_end_ms=None
):
    try:
        plt.figure(figsize=(14, 7))
        # Add vertical line at expected button press time if provided
        if button_press_time_ms is not None:
            plt.axvline(
                x=button_press_time_ms,
                color='green',
                linestyle='--',
                linewidth=2,
                alpha=0.7,
                label=f'Expected Button Press (~{button_press_time_ms:.0f}ms)'
            )
        
        # Highlight extracted peak segment, if provided
        if highlight_start_ms is not None and highlight_end_ms is not None:
            peak_duration_ms = highlight_end_ms - highlight_start_ms
            plt.axvspan(
                highlight_start_ms,
                highlight_end_ms,
                color='orange',
                alpha=0.2,
                label=f'Saved Peak Window ({peak_duration_ms:.1f}ms)'
            )
            
            # Add a text annotation showing the duration
            mid_point_ms = (highlight_start_ms + highlight_end_ms) / 2.0
            max_voltage = np.max(voltage_data)
            plt.annotate(
                f'{peak_duration_ms:.1f}ms',
                xy=(mid_point_ms, max_voltage * 0.95),
                fontsize=12,
                fontweight='bold',
                color='darkorange',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='orange', alpha=0.8)
            )
        # Plot raw trace
        plt.plot(time_points, voltage_data, linewidth=0.5, alpha=0.5, color='blue', label='Raw Trace')
        
        # Calculate and plot moving average
        window_size = 30  # Adjust this for more/less smoothing
        moving_avg = np.convolve(voltage_data, np.ones(window_size)/window_size, mode='valid')
        # Adjust time points for moving average (center the window)
        ma_time_points = time_points[window_size//2:len(moving_avg)+window_size//2]
        plt.plot(ma_time_points, moving_avg, linewidth=1.5, color='red', label=f'Moving Average (window={window_size})')
        
        plt.xlabel('Time (ms)', fontsize=12)
        plt.ylabel('Voltage (V)', fontsize=12)
        plt.title(title, fontsize=13)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save as PNG image next to the CSV file
        if filepath:
            png_path = filepath.replace('.csv', '.png')
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            logger.info(f"Waveform visualization saved to {png_path}")
            
            # Try to open the image automatically
            # try:
            #     import subprocess
            #     subprocess.Popen(['open', png_path])
            #     logger.info(f"Opened visualization in default image viewer")
            # except Exception as e:
            #     logger.warning(f"Could not auto-open image: {e}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error visualizing waveform: {e}")

# Initialize AD3 connection and configuration
@app.route('/initialize', methods=['POST'])
def initialize():
    try:
        if initialize_ad3():
            return jsonify({
                "status": "success",
                "message": "AD3 initialized and ready for recording"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to initialize AD3"
            }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Initialization failed: {str(e)}"
        }), 500

# Perform the actual recording after waiting for server-side sync delay
def do_recording_async_with_delay(rows, cols, button, press_length_ms, server_wait_ms):
    global show_graphs
    
    try:
        # Wait for sync delay if specified (0 = start immediately)
        if server_wait_ms > 0:
            logger.info(f"Server waiting {server_wait_ms}ms before starting recording...")
            time.sleep(server_wait_ms / 1000.0)
        logger.info("Starting recording NOW")
        
        # Create directory structure
        directory = create_recording_directory(rows, cols, button)
        
        # Get next trace number with persistent storage
        button_key = f"{rows}x{cols}_button_{button}"
        trace_number = get_next_trace_number(button_key)
        
        # Create filename
        filename = f"trace{trace_number}.csv"
        filepath = os.path.join(directory, filename)
        
        logger.info(f"Starting background recording for button {button}")
        
        # Record waveform from AD3
        time_points, voltage_data = record_waveform(press_length_ms)
        
        # Save full recording to CSV
        save_waveform_to_csv(time_points, voltage_data, filepath)
        
        logger.info(f"Recording completed successfully - saved to {filepath}")
        
        # Press should appear at approximately PRESS_POSITION_MS in the recording
        button_press_time_ms = PRESS_POSITION_MS
        
        # Extract the main peak segment after the button press and save it separately
        peak_result = extract_peak_segment(time_points, voltage_data, button_press_time_ms, press_length_ms)
        highlight_start_ms = None
        highlight_end_ms = None
        
        if peak_result is not None:
            peak_times, peak_voltages, highlight_start_ms, highlight_end_ms = peak_result
            peak_filepath = filepath.replace('.csv', '_peak.csv')
            save_waveform_to_csv(peak_times, peak_voltages, peak_filepath)
            logger.info(f"Saved peak segment to {peak_filepath}")
        else:
            logger.warning("No peak segment extracted for this trace")
        
        # Show graph if enabled
        if show_graphs:
            title = f"Button {button} - Trace {trace_number} ({rows}x{cols} Grid)"
            logger.info(
                f"Button press expected at ~{button_press_time_ms:.0f}ms (PRESS_POSITION_MS={PRESS_POSITION_MS}ms). "
                f"Highlight window: {highlight_start_ms}ms to {highlight_end_ms}ms"
            )
            visualize_waveform(
                time_points,
                voltage_data,
                title,
                filepath,
                button_press_time_ms,
                highlight_start_ms,
                highlight_end_ms
            )
        
    except Exception as e:
        logger.error(f"Background recording failed: {str(e)}")

# Start a new oscilloscope recording with AD3
@app.route('/start_recording', methods=['POST'])
def start_recording():
    try:
        # Get JSON payload
        data = request.get_json()
        
        if not data:
            logger.error("No JSON payload provided")
            return jsonify({"status": "error", "message": "No JSON payload provided"}), 400
        
        # Extract required fields
        rows = data.get('rows')
        cols = data.get('cols')
        button = data.get('button')
        press_length_ms = data.get('press_length_ms')
        network_latency_ms = data.get('network_latency_ms', 0) # Optional

        press_length_ms = int(press_length_ms)
        
        # Validate required fields
        if rows is None or cols is None or button is None or press_length_ms is None:
            missing_fields = []
            if rows is None: missing_fields.append('rows')
            if cols is None: missing_fields.append('cols')
            if button is None: missing_fields.append('button')
            if press_length_ms is None: missing_fields.append('press_length_ms')
            
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            logger.error(error_msg)
            return jsonify({"status": "error", "message": error_msg}), 400
        
        # Validate field types and values
        if not isinstance(rows, int) or rows <= 0:
            return jsonify({"status": "error", "message": "rows must be a positive integer"}), 400
        if not isinstance(cols, int) or cols <= 0:
            return jsonify({"status": "error", "message": "cols must be a positive integer"}), 400
        if not isinstance(button, int) or button < 0:
            return jsonify({"status": "error", "message": "button must be a non-negative integer"}), 400
        if not isinstance(press_length_ms, int) or press_length_ms <= 0:
            return jsonify({"status": "error", "message": "press_length_ms must be a positive integer"}), 400
        
        logger.info(f"Recording request - Button: {button}, Grid: {rows}x{cols}, Duration: {press_length_ms}ms")
        
        # Check if AD3 is available and configured
        if not AD3_AVAILABLE:
            return jsonify({"status": "error", "message": "Analog Discovery 3 not available."}), 500
        
        if not ad3_configured:
            return jsonify({"status": "error", "message": "AD3 not initialized. Call /initialize first."}), 500
        
        # TIMING CALCULATION:
        #   - Phone presses at: STARTUP_BUFFER_MS + PHONE_DELAY_MS (from request time)
        #   - Recording starts at: server_wait_ms (from request time)
        #   - Press appears at: (STARTUP_BUFFER_MS + PHONE_DELAY_MS) - server_wait_ms
        #   - We want press at PRESS_POSITION_MS, so:
        #     server_wait_ms = STARTUP_BUFFER_MS + PHONE_DELAY_MS - PRESS_POSITION_MS
        
        server_wait_ms = STARTUP_BUFFER_MS + PHONE_DELAY_MS - PRESS_POSITION_MS
        
        # Sanity check
        recording_duration_ms = (RECORD_SAMPLES / SAMPLE_RATE_HZ) * 1000
        if server_wait_ms < 0:
            logger.warning(f"server_wait_ms={server_wait_ms}ms is negative, clamping to 0")
            server_wait_ms = 0
        if PRESS_POSITION_MS > recording_duration_ms * 0.8:
            logger.warning(f"PRESS_POSITION_MS={PRESS_POSITION_MS}ms is near end of {recording_duration_ms:.0f}ms recording")
        
        logger.info(f"Timing: server_wait={server_wait_ms}ms, phone_delay={PHONE_DELAY_MS}ms, "
                   f"press_position={PRESS_POSITION_MS}ms, recording={recording_duration_ms:.0f}ms")
        
        # Start recording in background thread (waits server_wait_ms before AD3 starts)
        recording_thread = threading.Thread(
            target=do_recording_async_with_delay,
            args=(rows, cols, button, press_length_ms, server_wait_ms)
        )
        recording_thread.daemon = True
        recording_thread.start()
        
        # Wait before responding (ensures thread is running)
        time.sleep(STARTUP_BUFFER_MS / 1000.0)
        
        # Tell phone how long to wait before pressing
        logger.info(f"Responding to phone - phone will wait {PHONE_DELAY_MS}ms before pressing")
        return jsonify({
            "status": "success",
            "message": "Recording started",
            "delay_ms": PHONE_DELAY_MS
        })
        
    except Exception as e:
        error_msg = f"Recording failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500


# Simple endpoint to flip a button WITHOUT recording (for testing)
@app.route('/flip', methods=['POST'])
def flip_button():
    """
    Tell the phone to flip a button without starting AD3 recording.
    Useful for testing with external recording tools like ad3_waveforms_clone.py
    
    Usage:
        curl -X POST http://localhost:5150/flip -H "Content-Type: application/json" \
             -d '{"delay_ms": 500, "press_length_ms": 100}'
    
    Parameters:
        delay_ms: How long phone waits before flipping (default: 100)
        press_length_ms: How long button stays pressed (default: 100)
    """
    try:
        data = request.get_json() or {}
        delay_ms = data.get('delay_ms', 100)
        press_length_ms = data.get('press_length_ms', 100)
        
        logger.info(f"Flip request: delay={delay_ms}ms, press_length={press_length_ms}ms (NO recording)")
        
        return jsonify({
            "status": "success",
            "message": "Flip command sent (no recording)",
            "delay_ms": delay_ms,
            "press_length_ms": press_length_ms
        })
        
    except Exception as e:
        error_msg = f"Flip failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500


# Return server timestamp for clock sync testing
@app.route('/ping', methods=['POST'])
def ping():
    try:
        server_time = time.time()
        return jsonify({
            "status": "success",
            "server_time": server_time
        })
    except Exception as e:
        logger.error(f"Ping failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# Toggle graph visualization on/off
@app.route('/toggle_graphs', methods=['POST'])
def toggle_graphs():
    global show_graphs
    
    data = request.get_json()
    if data and 'enabled' in data:
        show_graphs = data['enabled']
    else:
        show_graphs = not show_graphs  # Toggle if no value provided
    
    logger.info(f"Graph visualization {'enabled' if show_graphs else 'disabled'}")
    return jsonify({
        "status": "success",
        "show_graphs": show_graphs
    })

# Set the dataset mode (train/test)
@app.route('/set_dataset_mode', methods=['POST'])
def set_dataset_mode():
    global dataset_mode
    data = request.get_json() or {}
    mode = data.get('mode')
    
    if mode not in ('train', 'test'):
        return jsonify({
            "status": "error",
            "message": "mode must be 'train' or 'test'"
        }), 400
    
    dataset_mode = mode
    logger.info(f"Dataset mode set to {dataset_mode}")
    return jsonify({
        "status": "success",
        "dataset_mode": dataset_mode
    })

# Get current API status and AD3 availability
@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "running",
        "ad3_available": AD3_AVAILABLE,
        "ad3_configured": ad3_configured,
        "show_graphs": show_graphs,
        "dataset_mode": dataset_mode,
        "trace_counters": trace_counters,
        "server_time": datetime.now().isoformat()
    })

# Simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    # Load trace counters on startup
    load_trace_counters()
    
    logger.info("Starting Flask API server for Analog Discovery 3")
    logger.info("Server will run on 0.0.0.0:5150")
    
    if AD3_AVAILABLE:
        logger.info("Analog Discovery 3 SDK is available")
        logger.info("AD3 will be initialized when /initialize endpoint is called")
    else:
        logger.warning("Analog Discovery 3 SDK not available - install with: pip install pydwf")
    
    logger.info("Ready to receive requests")
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5150, debug=False)

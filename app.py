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
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
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
# Match the standalone AD3 capture script (ad3_waveforms_clone.py)
SAMPLE_RATE_HZ = 5000       # Oscilloscope sampling rate (Hz)
CHANNEL_RANGE_V = 0.5        # ~500 mV/div equivalent full-scale range
RECORD_SAMPLES = 32768       # Fixed number of samples per trace (~1.64s at 20 kHz)
SYNC_DELAY_MS = 250          # Phone waits this long after response before pressing
PRE_ROLL_MS = 500            # How long before the press the recording should already be running

# Initialize AD3 if available
if AD3_AVAILABLE:
    try:
        ad3_library = DwfLibrary()
        logger.info("WaveForms SDK initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize WaveForms SDK: {e}")
        AD3_AVAILABLE = False

def create_recording_directory(grid_rows, grid_cols, button):
    """Create the recording directory structure if it doesn't exist"""
    directory = os.path.join(dataset_mode, f"{grid_rows}x{grid_cols}", f"button_{button}")
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Created/verified directory: {directory}")
    return directory

def initialize_ad3():
    """Initialize and connect to AD3 device once at startup"""
    global ad3_device, ad3_configured
    
    if not AD3_AVAILABLE:
        logger.warning("AD3 not available - cannot initialize")
        return False
    
    if ad3_configured:
        logger.info("AD3 already initialized and configured")
        return True
    
    try:
        # Get device count
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

def configure_oscilloscope():
    """Configure oscilloscope for channel 1 recording"""
    global ad3_device
    
    if not ad3_device:
        raise Exception("AD3 device not connected")
    
    try:
        # Configure channel 1 (analog input) - match standalone AD3 capture script
        ad3_device.analogIn.channelEnableSet(0, True)  # Channel 1
        ad3_device.analogIn.channelRangeSet(0, CHANNEL_RANGE_V)
        ad3_device.analogIn.channelOffsetSet(0, 0.0)   # No offset
        
        # Configure acquisition parameters
        ad3_device.analogIn.frequencySet(SAMPLE_RATE_HZ)
        ad3_device.analogIn.bufferSizeSet(RECORD_SAMPLES)
        
        # Use single acquisition mode - WaveForms style
        ad3_device.analogIn.acquisitionModeSet(DwfAcquisitionMode.Single)
        ad3_device.analogIn.triggerAutoTimeoutSet(0)  # No trigger timeout
        ad3_device.analogIn.triggerSourceSet(DwfTriggerSource.None_)  # No trigger
        
        logger.info("Oscilloscope configured for channel 1 (single acquisition mode)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to configure oscilloscope: {e}")
        raise

def start_acquisition():
    """Start oscilloscope acquisition using fixed buffer size (WaveForms-style)"""
    global ad3_device
    
    if not ad3_device:
        raise Exception("AD3 device not connected")
    
    try:
        # Re-apply oscilloscope configuration to ensure settings are correct
        ad3_device.analogIn.channelEnableSet(0, True)
        ad3_device.analogIn.channelRangeSet(0, CHANNEL_RANGE_V)
        ad3_device.analogIn.channelOffsetSet(0, 0.0)
        ad3_device.analogIn.frequencySet(SAMPLE_RATE_HZ)
        
        # Use fixed buffer size for each recording (matches standalone script)
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

def get_waveform_data():
    """Retrieve waveform data from AD3 and construct a correctly scaled time axis"""
    global ad3_device
    
    if not ad3_device:
        raise Exception("AD3 device not connected")
    
    try:
        # Get the actual buffer size that was used
        buffer_size = ad3_device.analogIn.bufferSizeGet()
        logger.info(f"Buffer size: {buffer_size}")
        
        # Get channel 1 data - read the full buffer
        channel_data = ad3_device.analogIn.statusData(0, buffer_size)
        
        # Convert to numpy array for easier handling
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
        logger.info(f"Retrieved {len(data_array)} data points")
        logger.info(f"First 5 voltage values: {data_array[:5]}")
        logger.info(f"Last 5 voltage values: {data_array[-5:]}")
        logger.info(f"Voltage range: {np.min(data_array):.6f}V to {np.max(data_array):.6f}V")
        
        return time_points, data_array
        
    except Exception as e:
        logger.error(f"Failed to retrieve waveform data: {e}")
        raise

def disconnect_ad3():
    """Disconnect from AD3 device"""
    global ad3_device
    
    if ad3_device:
        try:
            ad3_device.close()
            ad3_device = None
            logger.info("Disconnected from Analog Discovery 3")
        except Exception as e:
            logger.error(f"Error disconnecting from AD3: {e}")

def load_trace_counters():
    """Load trace counters from persistent storage"""
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

def save_trace_counters():
    """Save trace counters to persistent storage"""
    global trace_counters
    
    try:
        with open('trace_counters.json', 'w') as f:
            json.dump(trace_counters, f)
        logger.info(f"Saved trace counters: {trace_counters}")
    except Exception as e:
        logger.error(f"Error saving trace counters: {e}")

def get_next_trace_number(button_key):
    """Get the next trace number for a button with persistent storage"""
    global trace_counters
    
    # Initialize counter for this button if not exists
    if button_key not in trace_counters:
        trace_counters[button_key] = 0
    
    # Increment and save
    trace_counters[button_key] += 1
    save_trace_counters()
    
    return trace_counters[button_key]

def record_waveform(press_length_ms):
    """
    Record waveform from AD3 for specified duration
    AD3 must already be initialized and configured
    Returns (time_points, voltage_data) or raises exception
    """
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
        
        # Start acquisition (AD3 already configured)
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

def save_waveform_to_csv(time_points, voltage_data, filepath):
    """Save waveform data to CSV file"""
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

def visualize_waveform(time_points, voltage_data, title="Waveform", filepath=None, button_press_time_ms=None):
    """Save waveform visualization as PNG image with moving average and optional press marker"""
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
            
            # Try to open the image automatically on Mac
            try:
                import subprocess
                subprocess.Popen(['open', png_path])
                logger.info(f"Opened visualization in default image viewer")
            except Exception as e:
                logger.warning(f"Could not auto-open image: {e}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error visualizing waveform: {e}")

@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize AD3 connection and configuration"""
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

def do_recording_async_with_delay(rows, cols, button, press_length_ms, server_wait_ms):
    """Perform the actual recording after waiting for server-side sync delay"""
    global show_graphs
    
    try:
        # WAIT for sync delay before starting recording
        logger.info(f"Server will wait {server_wait_ms}ms before starting recording (pre-roll {PRE_ROLL_MS}ms).")
        time.sleep(server_wait_ms / 1000.0)
        logger.info("Sync delay complete - starting recording NOW")
        
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
        
        # Show graph if enabled
        if show_graphs:
            title = f"Button {button} - Trace {trace_number} ({rows}x{cols} Grid)"
            # By design, recording starts PRE_ROLL_MS before the expected button press,
            # so the press should occur PRE_ROLL_MS after recording start.
            button_press_time_ms = PRE_ROLL_MS
            logger.info(
                f"Button press marker at {button_press_time_ms:.1f}ms after recording start "
                f"(SYNC_DELAY_MS={SYNC_DELAY_MS}ms, PRE_ROLL_MS={PRE_ROLL_MS}ms, "
                f"server_wait_ms={server_wait_ms}ms)."
            )
            visualize_waveform(time_points, voltage_data, title, filepath, button_press_time_ms)
        
    except Exception as e:
        logger.error(f"Background recording failed: {str(e)}")

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start a new oscilloscope recording with AD3 - synchronized delay approach"""
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
        network_latency_ms = data.get('network_latency_ms', 0)  # Optional: network latency from clock sync test

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
        
        # Define synchronization behavior:
        #  - Phone waits SYNC_DELAY_MS after receiving the response, then presses the button.
        #  - We want the AD3 recording to start PRE_ROLL_MS before that press.
        #  - We approximate that by having the server wait:
        #        server_wait_ms = SYNC_DELAY_MS + network_latency_ms - PRE_ROLL_MS
        #    after sending the response, before starting acquisition.
        server_wait_ms = SYNC_DELAY_MS + network_latency_ms - PRE_ROLL_MS
        if server_wait_ms < 0:
            logger.warning(
                f"Computed negative server_wait_ms={server_wait_ms}ms; clamping to 0ms. "
                f"(SYNC_DELAY_MS={SYNC_DELAY_MS}, PRE_ROLL_MS={PRE_ROLL_MS}, "
                f"network_latency_ms={network_latency_ms})"
            )
            server_wait_ms = 0
        
        logger.info(f"Sync delay for phone (client wait): {SYNC_DELAY_MS}ms")
        logger.info(f"Network latency compensation (one-way estimate): {network_latency_ms}ms")
        logger.info(
            f"Server will wait {server_wait_ms}ms before starting recording "
            f"(target pre-roll {PRE_ROLL_MS}ms)."
        )
        
        # Start recording in background thread with compensated delay (server_wait_ms)
        recording_thread = threading.Thread(
            target=do_recording_async_with_delay,
            args=(rows, cols, button, press_length_ms, server_wait_ms)
        )
        recording_thread.daemon = True
        recording_thread.start()

        time.sleep(1)
        
        # Respond immediately with sync delay (phone uses this; server uses server_wait_ms)
        logger.info(
            f"Responded to phone - phone will wait {SYNC_DELAY_MS}ms, "
            f"server will wait {server_wait_ms}ms before acquisition."
        )
        return jsonify({
            "status": "success",
            "message": "Recording will start after delay",
            "delay_ms": SYNC_DELAY_MS
        })
        
    except Exception as e:
        error_msg = f"Recording failed: {str(e)}"
        logger.error(error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500

@app.route('/ping', methods=['POST'])
def ping():
    """Return server timestamp for clock sync testing"""
    try:
        server_time = time.time()
        return jsonify({
            "status": "success",
            "server_time": server_time
        })
    except Exception as e:
        logger.error(f"Ping failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/toggle_graphs', methods=['POST'])
def toggle_graphs():
    """Toggle graph visualization on/off"""
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

@app.route('/set_dataset_mode', methods=['POST'])
def set_dataset_mode():
    """Set the dataset mode (train/test)"""
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

@app.route('/status', methods=['GET'])
def get_status():
    """Get current API status and AD3 availability"""
    return jsonify({
        "status": "running",
        "ad3_available": AD3_AVAILABLE,
        "ad3_configured": ad3_configured,
        "show_graphs": show_graphs,
        "dataset_mode": dataset_mode,
        "trace_counters": trace_counters,
        "server_time": datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
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

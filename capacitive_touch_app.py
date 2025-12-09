from flask import Flask, request, jsonify
import os
import json
from datetime import datetime, timezone
import logging
import time
import threading
import csv

import numpy as np
from typing import Optional

try:
    from pydwf import DwfLibrary, DwfAcquisitionMode, DwfTriggerSource, DwfState
    AD3_AVAILABLE = True
except ImportError:
    AD3_AVAILABLE = False


# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


app = Flask(__name__)


# ----------------------------------------------------------------------
# AD3 configuration (match WaveForms `default4` / ad3_waveforms_clone.py)
# ----------------------------------------------------------------------

SAMPLE_RATE_HZ = 20000          # 20 kHz
BUFFER_SAMPLES = 44005          # ~2.2 s window
CHANNEL_RANGE_V = 0.5           # 500 mV full-scale

# Delay between prime tap and when the AD3 capture actually starts.
# This gives you time to prepare your finger while the host panel counts down.
ARM_DELAY_MS = 800

# Root directories for new capacitive-touch datasets
TOUCH_DATASET_ROOTS = {
    "train": "touch_train",
    "test": "touch_test",
}

TRACE_COUNTERS_FILE = "touch_trace_counters.json"


# Global state
ad3_library = None
ad3_device = None
ad3_configured = False

touch_trace_counters = {}
dataset_mode = "train"  # "train" or "test"
show_graphs = False

# Track the most recent capture so the host panel can show status.
# Example structure:
# {
#   "rows": 2, "cols": 1, "button": 0, "press_length_ms": 100,
#   "trace_number": 5, "csv_path": "...", "dataset_mode": "train",
#   "state": "armed" | "capturing" | "completed" | "error",
#   "armed_at": <unix_time>, "completed_at": <unix_time, optional>
# }
current_capture_state: Optional[dict] = None
capture_state_lock = threading.Lock()


# ----------------------------------------------------------------------
# AD3 helpers
# ----------------------------------------------------------------------

def initialize_ad3():
    """Initialize WaveForms SDK and connect to the first AD3 device."""
    global ad3_library, ad3_device, ad3_configured

    if not AD3_AVAILABLE:
        logger.warning("AD3 not available - install pydwf to enable hardware support.")
        return False

    if ad3_configured and ad3_device is not None:
        logger.info("AD3 already initialized and configured.")
        return True

    try:
        ad3_library = DwfLibrary()

        device_enum = ad3_library.deviceEnum
        device_enum.enumerateStart()
        device_count = device_enum.enumerateDevices()

        if device_count == 0:
            device_enum.enumerateStop()
            raise RuntimeError("No Analog Discovery devices found.")

        ad3_device = ad3_library.deviceControl.open(0)
        name = device_enum.deviceName(0)
        serial = device_enum.serialNumber(0)
        device_enum.enumerateStop()

        logger.info(f"Connected to AD3 device 0: {name} (SN: {serial})")

        configure_analog_in()
        ad3_configured = True
        logger.info("AD3 initialized and configured for capacitive-touch capture.")
        return True

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to initialize AD3: %s", exc)
        ad3_configured = False
        return False


def configure_analog_in():
    """Configure analog input to mimic WaveForms `default4` capture."""
    if ad3_device is None:
        raise RuntimeError("AD3 device not connected.")

    # Clean state
    ad3_device.analogIn.reset()

    # Channel 1 configuration
    ad3_device.analogIn.channelEnableSet(0, True)
    ad3_device.analogIn.channelRangeSet(0, CHANNEL_RANGE_V)
    ad3_device.analogIn.channelOffsetSet(0, 0.0)

    # Sampling config
    ad3_device.analogIn.frequencySet(SAMPLE_RATE_HZ)
    ad3_device.analogIn.bufferSizeSet(BUFFER_SAMPLES)

    # Acquisition mode: single-shot, free-running (no trigger)
    ad3_device.analogIn.acquisitionModeSet(DwfAcquisitionMode.Single)
    ad3_device.analogIn.triggerSourceSet(DwfTriggerSource.None_)
    ad3_device.analogIn.triggerAutoTimeoutSet(0)

    logger.info(
        "Configured AD3 analogIn: sample_rate=%d Hz, buffer=%d samples, "
        "range=%.3f V, no trigger",
        SAMPLE_RATE_HZ,
        BUFFER_SAMPLES,
        CHANNEL_RANGE_V,
    )


def start_capture_nonblocking():
    """Arm the scope and start a single acquisition without blocking."""
    if ad3_device is None:
        raise RuntimeError("AD3 device not connected.")

    # Ensure configuration is current
    configure_analog_in()

    # Arm acquisition. This returns immediately; capture runs in hardware.
    ad3_device.analogIn.configure(False, True)

    logger.info(
        "Armed non-blocking acquisition: sample_rate=%d Hz, samples=%d "
        "(~%.3f s window).",
        SAMPLE_RATE_HZ,
        BUFFER_SAMPLES,
        BUFFER_SAMPLES / float(SAMPLE_RATE_HZ),
    )


def wait_for_capture_and_get_data():
    """Wait for current acquisition to complete, then return (time_ms, voltage_v)."""
    if ad3_device is None:
        raise RuntimeError("AD3 device not connected.")

    # Wait until acquisition is done
    while True:
        status = ad3_device.analogIn.status(True)
        if status == DwfState.Done:
            break
        time.sleep(0.01)

    # Retrieve buffer size and data
    buffer_size = ad3_device.analogIn.bufferSizeGet()
    channel_data = ad3_device.analogIn.statusData(0, buffer_size)
    data_array = np.array(channel_data, dtype=float)

    if data_array.size == 0:
        raise RuntimeError("No data retrieved from AD3 - check signal connection.")

    # Build time axis in milliseconds
    sample_rate = ad3_device.analogIn.frequencyGet()
    total_duration_s = data_array.size / float(sample_rate)
    time_points_s = np.linspace(
        0.0,
        total_duration_s,
        data_array.size,
        endpoint=False,
    )

    time_ms = time_points_s * 1000.0

    logger.info(
        "Captured %d samples at %.1f Hz (%.3f s). Voltage range: %.6f V .. %.6f V",
        data_array.size,
        float(sample_rate),
        total_duration_s,
        float(np.min(data_array)),
        float(np.max(data_array)),
    )

    return time_ms, data_array


def disconnect_ad3():
    """Disconnect from AD3 device."""
    global ad3_device, ad3_configured

    if ad3_device is not None:
        try:
            ad3_device.close()
            logger.info("Disconnected from AD3.")
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Error while disconnecting AD3: %s", exc)
        finally:
            ad3_device = None
            ad3_configured = False


# ----------------------------------------------------------------------
# Trace counter + filesystem helpers
# ----------------------------------------------------------------------

def load_touch_trace_counters():
    """Load trace counters for the capacitive-touch dataset from disk."""
    global touch_trace_counters

    if os.path.exists(TRACE_COUNTERS_FILE):
        try:
            with open(TRACE_COUNTERS_FILE, "r", encoding="utf-8") as fh:
                touch_trace_counters = json.load(fh)
            logger.info("Loaded touch trace counters from %s", TRACE_COUNTERS_FILE)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to load touch trace counters: %s", exc)
            touch_trace_counters = {}
    else:
        touch_trace_counters = {}
        logger.info("No existing touch trace counters found; starting fresh.")


def save_touch_trace_counters():
    """Persist trace counters for the capacitive-touch dataset."""
    try:
        with open(TRACE_COUNTERS_FILE, "w", encoding="utf-8") as fh:
            json.dump(touch_trace_counters, fh, indent=2)
        logger.info("Saved touch trace counters to %s", TRACE_COUNTERS_FILE)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to save touch trace counters: %s", exc)


def get_next_touch_trace_number(button_key: str) -> int:
    """Increment and return the next trace number for a given button key."""
    global touch_trace_counters

    if button_key not in touch_trace_counters:
        touch_trace_counters[button_key] = 0

    touch_trace_counters[button_key] += 1
    save_touch_trace_counters()
    return touch_trace_counters[button_key]


def get_dataset_root() -> str:
    """Return the root directory for the current dataset mode."""
    root = TOUCH_DATASET_ROOTS.get(dataset_mode)
    if root is None:
        raise ValueError(f"Unsupported dataset_mode: {dataset_mode}")
    return root


def create_touch_directory(rows: int, cols: int, button: int) -> str:
    """
    Create the directory structure for a capacitive-touch recording.

    Layout example:
        touch_train/2x1/button_0/trace<N>.csv
    """
    root = get_dataset_root()
    directory = os.path.join(root, f"{rows}x{cols}", f"button_{button}")
    os.makedirs(directory, exist_ok=True)
    logger.info("Created / verified touch directory: %s", directory)
    return directory


def save_waveform_csv(time_ms, voltage_v, filepath: str):
    """Save waveform to CSV with headers: time_ms, voltage_v."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time_ms", "voltage_v"])
        for t, v in zip(time_ms, voltage_v):
            writer.writerow([float(t), float(v)])

    logger.info("Saved capacitive-touch waveform to %s", filepath)


def save_metadata_json(meta: dict, filepath: str):
    """Save metadata JSON next to the CSV."""
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Saved metadata to %s", filepath)


def visualize_waveform(time_ms, voltage_v, filepath: str, meta: Optional[dict] = None):
    """Optionally save a PNG visualization of the waveform."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pylint: disable=import-error
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("matplotlib not available; skipping visualization: %s", exc)
        return

    try:
        plt.figure(figsize=(12, 6))
        plt.plot(time_ms, voltage_v, linewidth=0.7, alpha=0.7, label="Raw trace")

        # Simple moving average for smoothing
        window = 50
        if len(voltage_v) > window:
            kernel = np.ones(window) / float(window)
            smoothed = np.convolve(voltage_v, kernel, mode="valid")
            smoothed_t = time_ms[window // 2 : window // 2 + len(smoothed)]
            plt.plot(smoothed_t, smoothed, color="red", linewidth=1.0, label="Moving avg")

        title = "Capacitive-touch waveform"
        if meta:
            rows = meta.get("rows")
            cols = meta.get("cols")
            button = meta.get("button_index")
            title = f"Touch {rows}x{cols}, button {button}"

        plt.title(title)
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (V)")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")
        plt.tight_layout()

        png_path = os.path.splitext(filepath)[0] + ".png"
        plt.savefig(png_path, dpi=150)
        plt.close()

        logger.info("Saved waveform visualization to %s", png_path)
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Failed to visualize waveform: %s", exc)


# ----------------------------------------------------------------------
# Background recording worker
# ----------------------------------------------------------------------

def record_touch_trace_async(
    rows: int,
    cols: int,
    button: int,
    press_length_ms: int,
    trace_number: int,
    csv_path: str,
    meta_path: str,
    capture_start_server_time: float,
    extra_meta: Optional[dict] = None,
    save_png: bool = True,
):
    """
    Background worker that waits for the current capture to finish, fetches
    data, and writes CSV + metadata (and optional PNG).
    """
    global show_graphs, current_capture_state

    try:
        logger.info(
            "Background touch recorder started for %dx%d button %d trace %d",
            rows,
            cols,
            button,
            trace_number,
        )

        # Mark state as actively capturing (for host panel)
        with capture_state_lock:
            if current_capture_state and current_capture_state.get("trace_number") == trace_number:
                current_capture_state["state"] = "capturing"

        time_ms, voltage_v = wait_for_capture_and_get_data()

        # Persist CSV
        save_waveform_csv(time_ms, voltage_v, csv_path)

        # Build metadata
        now_utc = datetime.now(timezone.utc)
        meta = {
            "rows": rows,
            "cols": cols,
            "button_index": button,
            "press_length_ms": press_length_ms,
            "dataset_mode": dataset_mode,
            "dataset_root": get_dataset_root(),
            "trace_number": trace_number,
            "csv_path": csv_path,
            "created_at_utc": now_utc.isoformat(),
            "capture_start_server_time": capture_start_server_time,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "buffer_samples": BUFFER_SAMPLES,
            "touch_mode": "capacitive_two_tap",
        }

        if extra_meta:
            meta.update(extra_meta)

        save_metadata_json(meta, meta_path)

        if save_png or show_graphs:
            visualize_waveform(time_ms, voltage_v, csv_path, meta)

        # Mark capture as completed
        with capture_state_lock:
            if current_capture_state and current_capture_state.get("trace_number") == trace_number:
                current_capture_state["state"] = "completed"
                current_capture_state["completed_at"] = time.time()

        logger.info(
            "Background touch recording complete for trace %d (rows=%d, cols=%d, button=%d).",
            trace_number,
            rows,
            cols,
            button,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Background touch recording failed: %s", exc)
        with capture_state_lock:
            if current_capture_state and current_capture_state.get("trace_number") == trace_number:
                current_capture_state["state"] = "error"
                current_capture_state["error_message"] = str(exc)


def arm_and_record_touch_trace_async(
    rows: int,
    cols: int,
    button: int,
    press_length_ms: int,
    trace_number: int,
    csv_path: str,
    meta_path: str,
    extra_meta: Optional[dict],
    save_png: bool,
    scheduled_start_time: float,
) -> None:
    """
    Wait until the scheduled arm time, then start the AD3 capture and
    delegate to record_touch_trace_async to fetch and persist data.
    """
    global current_capture_state  # noqa: PLW0603

    try:
        # Sleep until it's time to arm the capture.
        delay = scheduled_start_time - time.time()
        if delay > 0:
            time.sleep(delay)

        capture_start_server_time = time.time()

        # Start the capture now.
        start_capture_nonblocking()

        # Mark as armed so the host panel can cue "Press Now".
        with capture_state_lock:
            if current_capture_state and current_capture_state.get("trace_number") == trace_number:
                current_capture_state["state"] = "armed"
                current_capture_state["armed_at"] = capture_start_server_time

        # Now block until capture completes and save the data.
        record_touch_trace_async(
            rows=rows,
            cols=cols,
            button=button,
            press_length_ms=press_length_ms,
            trace_number=trace_number,
            csv_path=csv_path,
            meta_path=meta_path,
            capture_start_server_time=capture_start_server_time,
            extra_meta=extra_meta,
            save_png=save_png,
        )

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("arm_and_record_touch_trace_async failed: %s", exc)
        with capture_state_lock:
            if current_capture_state and current_capture_state.get("trace_number") == trace_number:
                current_capture_state["state"] = "error"
                current_capture_state["error_message"] = str(exc)


# ----------------------------------------------------------------------
# Flask routes
# ----------------------------------------------------------------------

@app.route("/touch/initialize", methods=["POST"])
def touch_initialize():
    """Initialize AD3 for capacitive-touch captures."""
    try:
        if initialize_ad3():
            return jsonify(
                {
                    "status": "success",
                    "message": "AD3 initialized and ready for capacitive-touch recording.",
                }
            )
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Failed to initialize AD3. Check hardware / pydwf.",
                }
            ),
            500,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("touch_initialize failed: %s", exc)
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Initialization failed: {exc}",
                }
            ),
            500,
        )


@app.route("/touch/start", methods=["POST"])
def touch_start():
    """
    Start a capacitive-touch recording.

    Intended flow:
      1) User taps once in the Android app ("prime") -> app POSTs here.
      2) This endpoint immediately arms a free-running AD3 capture and
         returns as soon as the capture has started.
      3) A separate host panel (on the laptop) shows when the capture is
         armed so the experimenter can cue the user to press and hold on
         the second tap. Alignment is handled offline from the waveform.
    """
    try:
        data = request.get_json() or {}

        rows = data.get("rows")
        cols = data.get("cols")
        button = data.get("button")
        press_length_ms = data.get("press_length_ms")
        save_png = bool(data.get("save_png", True))

        user_id = data.get("user_id")
        session_id = data.get("session_id")
        note = data.get("note")

        # Basic validation
        missing = []
        if rows is None:
            missing.append("rows")
        if cols is None:
            missing.append("cols")
        if button is None:
            missing.append("button")
        if press_length_ms is None:
            missing.append("press_length_ms")

        if missing:
            msg = f"Missing required fields: {', '.join(missing)}"
            logger.error(msg)
            return jsonify({"status": "error", "message": msg}), 400

        try:
            rows = int(rows)
            cols = int(cols)
            button = int(button)
            press_length_ms = int(press_length_ms)
        except (TypeError, ValueError) as exc:
            msg = f"rows, cols, button, and press_length_ms must be integers: {exc}"
            logger.error(msg)
            return jsonify({"status": 'error', "message": msg}), 400

        if rows <= 0 or cols <= 0 or button < 0 or press_length_ms <= 0:
            msg = "rows/cols must be >0, button >=0, press_length_ms >0."
            logger.error(msg)
            return jsonify({"status": "error", "message": msg}), 400

        if not AD3_AVAILABLE:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "AD3 hardware not available (pydwf not installed?).",
                    }
                ),
                500,
            )

        if not ad3_configured:
            # Try auto-initialize if user forgot to call /touch/initialize
            if not initialize_ad3():
                return (
                    jsonify(
                        {
                            "status": "error",
                            "message": "AD3 not initialized and auto-init failed.",
                        }
                    ),
                    500,
                )

        logger.info(
            "Touch start request: rows=%d cols=%d button=%d press=%d ms",
            rows,
            cols,
            button,
            press_length_ms,
        )

        # Determine directory + trace number
        directory = create_touch_directory(rows, cols, button)
        button_key = f"{rows}x{cols}_button_{button}"
        trace_number = get_next_touch_trace_number(button_key)

        csv_filename = f"trace{trace_number}.csv"
        csv_path = os.path.join(directory, csv_filename)
        meta_path = os.path.join(directory, f"trace{trace_number}.json")

        # Extra metadata from the client
        extra_meta = {}
        if user_id is not None:
            extra_meta["user_id"] = user_id
        if session_id is not None:
            extra_meta["session_id"] = session_id
        if note is not None:
            extra_meta["note"] = note

        # Compute when we want to actually arm the capture.
        scheduled_start_time = time.time() + ARM_DELAY_MS / 1000.0

        # Update current capture state for the host panel.
        global current_capture_state  # noqa: PLW0603  (explicitly updating global)
        with capture_state_lock:
            current_capture_state = {
                "rows": rows,
                "cols": cols,
                "button": button,
                "press_length_ms": press_length_ms,
                "trace_number": trace_number,
                "csv_path": csv_path,
                "dataset_mode": dataset_mode,
                "state": "waiting",
                "scheduled_start_time": scheduled_start_time,
            }

        # Spawn background worker to arm later and then record.
        worker = threading.Thread(
            target=arm_and_record_touch_trace_async,
            args=(
                rows,
                cols,
                button,
                press_length_ms,
                trace_number,
                csv_path,
                meta_path,
                extra_meta,
                save_png,
                scheduled_start_time,
            ),
            daemon=True,
        )
        worker.start()

        # Respond right away so the experimenter knows a capture is scheduled.
        window_ms = BUFFER_SAMPLES / float(SAMPLE_RATE_HZ) * 1000.0
        return jsonify(
            {
                "status": "success",
                "message": "AD3 capture scheduled; watch the host panel for the 'Press Now' cue.",
                "rows": rows,
                "cols": cols,
                "button": button,
                "press_length_ms": press_length_ms,
                "dataset_mode": dataset_mode,
                "trace_number": trace_number,
                "csv_path": csv_path,
                "expected_capture_window_ms": window_ms,
                "arm_delay_ms": ARM_DELAY_MS,
            }
        )

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("touch_start failed: %s", exc)
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Touch recording failed: {exc}",
                }
            ),
            500,
        )


@app.route("/touch/set_dataset_mode", methods=["POST"])
def touch_set_dataset_mode():
    """Set train/test mode for capacitive-touch recordings."""
    global dataset_mode

    data = request.get_json() or {}
    mode = data.get("mode")

    if mode not in ("train", "test"):
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "mode must be 'train' or 'test'",
                }
            ),
            400,
        )

    dataset_mode = mode
    logger.info("Capacitive-touch dataset_mode set to %s", dataset_mode)
    return jsonify({"status": "success", "dataset_mode": dataset_mode})


@app.route("/touch/toggle_graphs", methods=["POST"])
def touch_toggle_graphs():
    """Toggle server-side PNG generation / visualization."""
    global show_graphs

    data = request.get_json() or {}
    if "enabled" in data:
        show_graphs = bool(data["enabled"])
    else:
        show_graphs = not show_graphs

    logger.info("Touch waveform visualization %s", "enabled" if show_graphs else "disabled")
    return jsonify({"status": "success", "show_graphs": show_graphs})


@app.route("/touch/status", methods=["GET"])
def touch_status():
    """Return current status of the capacitive-touch API."""
    with capture_state_lock:
        capture_snapshot = dict(current_capture_state) if current_capture_state else None

    return jsonify(
        {
            "status": "running",
            "ad3_available": AD3_AVAILABLE,
            "ad3_configured": ad3_configured,
            "dataset_mode": dataset_mode,
            "dataset_roots": TOUCH_DATASET_ROOTS,
            "touch_trace_counters": touch_trace_counters,
            "show_graphs": show_graphs,
            "current_capture": capture_snapshot,
            "server_time": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.route("/touch/ping", methods=["POST"])
def touch_ping():
    """
    Simple ping endpoint used by the Android app to estimate latency and
    (optionally) compare clocks. Mirrors the legacy /ping shape:
        { "status": "success", "server_time": <unix_seconds_float> }
    """
    try:
        server_time = time.time()
        return jsonify({"status": "success", "server_time": server_time})
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("touch_ping failed: %s", exc)
        return (
            jsonify(
                {
                    "status": "error",
                    "message": str(exc),
                }
            ),
            500,
        )


@app.route("/touch/health", methods=["GET"])
def touch_health():
    """Simple health check for capacitive-touch service."""
    return jsonify({"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()})


@app.route("/touch/disconnect", methods=["POST"])
def touch_disconnect():
    """Explicitly disconnect from the AD3 device."""
    try:
        disconnect_ad3()
        return jsonify(
            {
                "status": "success",
                "message": "AD3 disconnected.",
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.error("touch_disconnect failed: %s", exc)
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Failed to disconnect AD3: {exc}",
                }
            ),
            500,
        )


PANEL_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AD3 Capacitive-Touch Host Panel</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0b1020;
      color: #f5f7ff;
      margin: 0;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .card {
      background: #151a2c;
      border-radius: 16px;
      padding: 20px 24px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.45);
    }
    h1 {
      margin: 0 0 4px 0;
      font-size: 24px;
      font-weight: 650;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      color: #8ab4ff;
    }
    .subtitle {
      opacity: 0.8;
      font-size: 13px;
    }
    .status-row {
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 12px;
      font-size: 14px;
    }
    .pill {
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 600;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .pill-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }
    .pill-ok {
      background: rgba(76, 175, 80, 0.15);
      color: #b9ffbf;
    }
    .pill-ok .pill-dot {
      background: #4caf50;
      box-shadow: 0 0 12px rgba(76,175,80,0.7);
    }
    .pill-bad {
      background: rgba(239, 83, 80, 0.12);
      color: #ffb7b3;
    }
    .pill-bad .pill-dot {
      background: #ef5350;
      box-shadow: 0 0 12px rgba(239,83,80,0.7);
    }
    .capture-state {
      margin-top: 12px;
      padding: 14px 16px;
      border-radius: 12px;
      background: radial-gradient(circle at 0% 0%, rgba(67, 160, 71, 0.32), transparent 55%);
      border: 1px solid rgba(129, 199, 132, 0.3);
    }
    .capture-state.idle {
      background: radial-gradient(circle at 0% 0%, rgba(96, 125, 139, 0.32), transparent 55%);
      border-color: rgba(144, 164, 174, 0.4);
    }
    .capture-state.error {
      background: radial-gradient(circle at 0% 0%, rgba(244, 67, 54, 0.32), transparent 55%);
      border-color: rgba(244, 67, 54, 0.5);
    }
    .capture-title {
      font-size: 15px;
      font-weight: 600;
      margin-bottom: 4px;
    }
    .capture-detail {
      font-size: 13px;
      opacity: 0.9;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
      font-size: 13px;
    }
    .label {
      opacity: 0.7;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 2px;
    }
    .value {
      font-weight: 500;
    }
    button {
      background: linear-gradient(120deg, #4caf50, #00c853);
      border: none;
      color: white;
      padding: 10px 16px;
      border-radius: 999px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      box-shadow: 0 10px 25px rgba(0, 200, 83, 0.4);
    }
    button:disabled {
      opacity: 0.5;
      cursor: default;
      box-shadow: none;
    }
    .controls {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-top: 12px;
      font-size: 12px;
      opacity: 0.85;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Capacitive-Touch Panel</h1>
    <div class="subtitle">Watch this screen for the cue. When it says “Press Now”, do the second tap and hold on the phone.</div>

    <div class="status-row">
      <div id="ad3-status" class="pill pill-bad">
        <div class="pill-dot"></div>
        <span>AD3: UNKNOWN</span>
      </div>
      <div id="api-status" class="pill pill-bad">
        <div class="pill-dot"></div>
        <span>API: DISCONNECTED</span>
      </div>
    </div>

    <div id="capture-box" class="capture-state idle">
      <div id="capture-title" class="capture-title">Waiting for arm…</div>
      <div id="capture-detail" class="capture-detail">
        Prime a button on the phone. When this box turns bright and says “Press Now”, perform the second tap and hold.
      </div>
      <div class="grid">
        <div>
          <div class="label">Grid</div>
          <div id="grid-value" class="value">–</div>
        </div>
        <div>
          <div class="label">Button</div>
          <div id="button-value" class="value">–</div>
        </div>
        <div>
          <div class="label">Trace</div>
          <div id="trace-value" class="value">–</div>
        </div>
        <div>
          <div class="label">Press (ms)</div>
          <div id="press-value" class="value">–</div>
        </div>
        <div>
          <div class="label">Dataset</div>
          <div id="dataset-value" class="value">–</div>
        </div>
        <div>
          <div class="label">State</div>
          <div id="state-value" class="value">Idle</div>
        </div>
      </div>
      <div class="controls">
        <div>
          <button id="sound-btn">Enable Beep</button>
        </div>
        <div id="hint-text">
          Beep + bright box = press and hold on the phone immediately.
        </div>
      </div>
    </div>
  </div>

  <script>
    let audioCtx = null;
    let lastCaptureKey = null;

    document.getElementById('sound-btn').addEventListener('click', () => {
      if (!audioCtx) {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (Ctx) {
          audioCtx = new Ctx();
        }
      }
      document.getElementById('sound-btn').disabled = true;
      document.getElementById('sound-btn').textContent = 'Beep Enabled';
    });

    function playBeep() {
      if (!audioCtx) return;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.value = 1000;
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      gain.gain.setValueAtTime(0.001, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.4, audioCtx.currentTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.25);
      osc.start();
      osc.stop(audioCtx.currentTime + 0.3);
    }

    async function pollStatus() {
      try {
        const res = await fetch('/touch/status');
        const data = await res.json();

        const ad3Ok = data.ad3_available && data.ad3_configured;
        const apiStatusEl = document.getElementById('api-status');
        const ad3StatusEl = document.getElementById('ad3-status');

        if (data.status === 'running') {
          apiStatusEl.className = 'pill pill-ok';
          apiStatusEl.innerHTML = '<div class="pill-dot"></div><span>API: CONNECTED</span>';
        } else {
          apiStatusEl.className = 'pill pill-bad';
          apiStatusEl.innerHTML = '<div class="pill-dot"></div><span>API: ERROR</span>';
        }

        if (ad3Ok) {
          ad3StatusEl.className = 'pill pill-ok';
          ad3StatusEl.innerHTML = '<div class="pill-dot"></div><span>AD3: READY</span>';
        } else {
          ad3StatusEl.className = 'pill pill-bad';
          ad3StatusEl.innerHTML = '<div class="pill-dot"></div><span>AD3: NOT READY</span>';
        }

        const capture = data.current_capture;
        const box = document.getElementById('capture-box');
        const title = document.getElementById('capture-title');
        const detail = document.getElementById('capture-detail');

        const gridEl = document.getElementById('grid-value');
        const btnEl = document.getElementById('button-value');
        const traceEl = document.getElementById('trace-value');
        const pressEl = document.getElementById('press-value');
        const datasetEl = document.getElementById('dataset-value');
        const stateEl = document.getElementById('state-value');

        if (!capture) {
          box.className = 'capture-state idle';
          title.textContent = 'Waiting for arm…';
          detail.textContent = 'Prime a button on the phone. When this box turns bright and says “Press Now”, perform the second tap and hold.';
          gridEl.textContent = '–';
          btnEl.textContent = '–';
          traceEl.textContent = '–';
          pressEl.textContent = '–';
          datasetEl.textContent = data.dataset_mode || '–';
          stateEl.textContent = 'Idle';
          return;
        }

        const key = capture.trace_number + ':' + capture.button;
        const state = capture.state || 'waiting';

        gridEl.textContent = (capture.rows || '–') + '×' + (capture.cols || '–');
        btnEl.textContent = capture.button ?? '–';
        traceEl.textContent = capture.trace_number ?? '–';
        pressEl.textContent = capture.press_length_ms ?? '–';
        datasetEl.textContent = capture.dataset_mode || data.dataset_mode || '–';

        if (state === 'waiting') {
          box.className = 'capture-state idle';
          title.textContent = 'Get Ready';
          detail.textContent = 'Capture will arm in a moment. Move your finger into position.';
          stateEl.textContent = 'Waiting';
        } else if (state === 'armed') {
          box.className = 'capture-state';
          title.textContent = 'Press Now';
          detail.textContent = 'AD3 is armed. Do the second tap and hold on the phone immediately.';
          stateEl.textContent = 'Armed';

          if (key !== lastCaptureKey) {
            lastCaptureKey = key;
            playBeep();
          }
        } else if (state === 'capturing') {
          box.className = 'capture-state';
          title.textContent = 'Recording…';
          detail.textContent = 'Hold the press steady until the configured duration is complete.';
          stateEl.textContent = 'Recording';
        } else if (state === 'completed') {
          box.className = 'capture-state idle';
          title.textContent = 'Done';
          detail.textContent = 'Capture finished. You can move to the next sample.';
          stateEl.textContent = 'Completed';
        } else {
          box.className = 'capture-state error';
          title.textContent = 'Error';
          detail.textContent = capture.error_message || 'Capture failed. Check the logs and hardware.';
          stateEl.textContent = 'Error';
        }
      } catch (err) {
        const apiStatusEl = document.getElementById('api-status');
        apiStatusEl.className = 'pill pill-bad';
        apiStatusEl.innerHTML = '<div class="pill-dot"></div><span>API: OFFLINE</span>';
      }
    }

    setInterval(pollStatus, 400);
    pollStatus();
  </script>
</body>
</html>
"""


@app.route("/touch/panel", methods=["GET"])
def touch_panel():
    """Serve a simple host panel UI for guiding capacitive-touch recordings."""
    return PANEL_HTML


if __name__ == "__main__":
    load_touch_trace_counters()

    logger.info("Starting capacitive-touch Flask API server for AD3.")
    logger.info("Server will run on 0.0.0.0:5250")

    if AD3_AVAILABLE:
        logger.info("AD3 SDK is available. Call /touch/initialize to connect.")
    else:
        logger.warning("AD3 SDK not available. Install with: pip install pydwf")

    app.run(host="0.0.0.0", port=5250, debug=False)



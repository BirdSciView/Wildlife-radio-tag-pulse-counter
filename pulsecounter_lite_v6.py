import numpy as np
import datetime
import csv
import argparse
import matplotlib.pyplot as plt
import logging
import os
from rtlsdr import RtlSdr
from scipy.signal import butter, lfilter, find_peaks

# Default parameters
DEFAULT_CENTER_FREQ = 163.082e6
SAMPLE_RATE = 2.4e6
GAIN = 'auto'
BLOCK_SIZE = 262144
DEFAULT_CSV_FILE = "pulse_count_test2.csv"
LOG_FILE = "pulse_logger_debug.log"

# Configure logging
logging.basicConfig(filename=LOG_FILE,
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# Band-pass filter
def design_bandpass(sr, low_hz, high_hz, order=4):
    nyq = 0.5 * sr
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def bandpass_filter(x, b, a):
    return lfilter(b, a, x)

# Manual width estimation
def estimate_width(env, peak_idx, sr):
    half_height = env[peak_idx] / 2
    left = peak_idx
    while left > 0 and env[left] > half_height:
        left -= 1
    right = peak_idx
    while right < len(env) and env[right] > half_height:
        right += 1
    return (right - left) / sr  # seconds

def main(csv_file, center_freq, live_plot=False):
    # Log start info
    logging.info("=== Pulse Logger Started ===")
    logging.info(f"CSV file: {csv_file}")
    logging.info(f"Center frequency: {center_freq/1e6:.3f} MHz")
    logging.info(f"Sample rate: {SAMPLE_RATE/1e6:.3f} Msps")
    logging.info(f"Gain: {GAIN}")
    logging.info(f"Flags: live_plot={live_plot}")

    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.center_freq = center_freq
    sdr.gain = GAIN

    b_bp, a_bp = design_bandpass(SAMPLE_RATE, 2000, 25000)

    print("Starting continuous logging... Press Ctrl-C to stop.")
    print(f"Logging pulses to {csv_file}")
    if live_plot:
        print("Live plotting ENABLED (use --plot to enable)")
        plt.ion()
        fig, ax_env = plt.subplots(figsize=(12, 6))

    # Open CSV in append mode
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header only if file is new/empty
        if not file_exists or os.path.getsize(csv_file) == 0:
            writer.writerow(["Date", "Time (microseconds)", "Amplitude", "Width (ms)", "Frequency (MHz)"])

        try:
            while True:
                samples = sdr.read_samples(BLOCK_SIZE * 10)
                bp = bandpass_filter(samples, b_bp, a_bp)
                bp = bp - np.mean(bp)
                env = np.abs(bp)

                threshold = np.median(env) * 5
                raw_peaks, _ = find_peaks(env, height=threshold)

                # Merge nearby detections
                merged_peaks = []
                if len(raw_peaks) > 0:
                    current = raw_peaks[0]
                    for p in raw_peaks[1:]:
                        if (p - current) < int(0.05 * SAMPLE_RATE):
                            if env[p] > env[current]:
                                current = p
                        else:
                            merged_peaks.append(current)
                            current = p
                    merged_peaks.append(current)

                if len(merged_peaks) == 0:
                    logging.debug("No peaks detected in this chunk")
                else:
                    for i, p in enumerate(merged_peaks):
                        now = datetime.datetime.now()
                        timestamp = now.time().isoformat(timespec='microseconds')
                        amp = env[p]
                        width_s = estimate_width(env, p, SAMPLE_RATE)
                        width_ms = width_s * 1e3

                        # Reject pulses shorter than 1 ms
                        if width_ms < 1.0:
                            logging.debug(f"Rejected short pulse at {timestamp}, width={width_ms:.2f} ms")
                            continue

                        writer.writerow([now.date().isoformat(),
                                         timestamp,
                                         f"{amp:.3f}",
                                         f"{width_ms:.2f}",
                                         f"{center_freq/1e6:.3f}"])
                        print(f"Pulse logged at {timestamp}, amp={amp:.3f}, width={width_ms:.2f} ms, freq={center_freq/1e6:.3f} MHz")
                        logging.debug(f"Pulse: time={timestamp}, amp={amp:.3f}, width={width_ms:.2f} ms, freq={center_freq/1e6:.3f} MHz")

                # Live plot update
                if live_plot:
                    ax_env.clear()
                    ax_env.plot(env, label="Envelope")
                    ax_env.axhline(threshold, color='red', linestyle='--', label="Threshold")
                    if len(merged_peaks) > 0:
                        ax_env.scatter(merged_peaks, env[merged_peaks], color='green', label="Detected pulses")
                    ax_env.set_title("Live Envelope with Detected Pulses")
                    ax_env.set_xlabel("Sample index")
                    ax_env.set_ylabel("Amplitude")
                    ax_env.legend()
                    plt.pause(0.01)

        except KeyboardInterrupt:
            print("\nStopping continuous logging...")
            logging.info("=== Pulse Logger Stopped ===")

    sdr.close()
    if live_plot:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous wildlife tag pulse logger")
    parser.add_argument("--plot", action="store_true", help="Enable live plotting of envelope")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_FILE, help="Custom CSV output file name")
    parser.add_argument("--freq", type=float, default=DEFAULT_CENTER_FREQ/1e6, help="Center frequency in MHz")
    args = parser.parse_args()

    print("Usage:")
    print("  python pulse_logger.py                 # log pulses only (default CSV, default freq)")
    print("  python pulse_logger.py --plot          # log pulses AND show live plot")
    print("  python pulse_logger.py --csv myrun.csv # log to custom CSV file")
    print("  python pulse_logger.py --freq 164.000  # set center frequency in MHz")

    main(csv_file=args.csv, center_freq=args.freq*1e6, live_plot=args.plot)

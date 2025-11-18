import numpy as np
import datetime
import csv
import argparse
import matplotlib.pyplot as plt
import logging
import os
import time
import math
from rtlsdr import RtlSdr
from scipy.signal import butter, lfilter, find_peaks

# Defaults
DEFAULT_FREQS_MHZ = [163.082]        # center frequencies in MHz
SAMPLE_RATE = 2.4e6                  # Hz
GAIN = 'auto'
BLOCK_SIZE = 262144
LOG_FILE = "pulse_logger_debug.log"
DEFAULT_INTERVAL = 10.0              # seconds

logging.basicConfig(filename=LOG_FILE,
                    level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

def design_bandpass(sr, low_hz, high_hz, order=4):
    nyq = 0.5 * sr
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def bandpass_filter(x, b, a):
    return lfilter(b, a, x)

def estimate_width(env, peak_idx, sr):
    half_height = env[peak_idx] / 2
    left = peak_idx
    while left > 0 and env[left] > half_height:
        left -= 1
    right = peak_idx
    while right < len(env) and env[right] > half_height:
        right += 1
    return (right - left) / sr

def build_csv_filename(freqs_hz, date=None):
    if date is None:
        date = datetime.datetime.now().date()
    date_str = date.isoformat()
    freqs_str = "-".join([f"{f/1e6:.3f}" for f in freqs_hz])
    return f"pulse_log_{date_str}_{freqs_str}.csv"

def process_frequency(sdr, freq_hz, writer, live_plot=False, ax_env=None,
                      snr_history=None, last_peak_time=None, prev_tail=None,
                      merge_window_ms=10, overlap_fraction=0.05):
    sdr.center_freq = freq_hz
    b_bp, a_bp = design_bandpass(SAMPLE_RATE, 2000, 25000)

    # Read exactly BLOCK_SIZE samples
    samples = sdr.read_samples(BLOCK_SIZE)

    # Software overlap: prepend tail from previous buffer
    if prev_tail is not None:
        samples = np.concatenate([prev_tail, samples])

    # Save new tail for next call
    new_tail = samples[-int(overlap_fraction * BLOCK_SIZE):]

    bp = bandpass_filter(samples, b_bp, a_bp)
    bp = bp - np.mean(bp)
    env = np.abs(bp)

    threshold_main = np.median(env) * 5
    threshold_low = np.median(env) * 3

    raw_peaks_main, _ = find_peaks(env, height=threshold_main)
    raw_peaks_low, _ = find_peaks(env, height=threshold_low)

    # Merge nearby detections for main peaks
    merged_peaks = []
    if len(raw_peaks_main) > 0:
        current = raw_peaks_main[0]
        for p in raw_peaks_main[1:]:
            if (p - current) < int(0.05 * SAMPLE_RATE):
                if env[p] > env[current]:
                    current = p
            else:
                merged_peaks.append(current)
                current = p
        merged_peaks.append(current)

    # Suspected missed peaks = those found at low threshold but not in main set
    missed_candidates = [p for p in raw_peaks_low if p not in raw_peaks_main]

    noise_floor = np.median(env)
    avg_env = np.mean(env)

    # Log confirmed peaks
    for p in merged_peaks:
        now = datetime.datetime.now()
        date_str = now.date().isoformat()
        time_str = now.time().isoformat(timespec='microseconds')

        amp = env[p]
        width_s = estimate_width(env, p, SAMPLE_RATE)
        width_ms = width_s * 1e3
        if width_ms < 1.0:
            continue

        snr_db = 20 * math.log10(amp / noise_floor) if noise_floor > 0 else 0
        par = amp / avg_env if avg_env > 0 else 0

        # Absolute peak time in ms
        peak_time_ms = (p / SAMPLE_RATE) * 1000.0
        if last_peak_time is not None and abs(peak_time_ms - last_peak_time) < merge_window_ms:
            continue  # skip double count

        if snr_history is not None:
            snr_history.setdefault(freq_hz, []).append(snr_db)
            avg_snr = np.mean(snr_history[freq_hz])
        else:
            avg_snr = snr_db

        writer.writerow([
            date_str, time_str,
            f"{amp:.3f}", f"{width_ms:.2f}", f"{freq_hz/1e6:.3f}",
            f"{snr_db:.2f}", f"{par:.2f}", f"{noise_floor:.3f}", "CONFIRMED"
        ])
        print(f"Pulse logged at {time_str}, amp={amp:.3f}, width={width_ms:.2f} ms, "
              f"freq={freq_hz/1e6:.3f} MHz, SNR={snr_db:.2f} dB, PAR={par:.2f}, NoiseFloor={noise_floor:.3f}, "
              f"AvgSNR={avg_snr:.2f} dB")

        last_peak_time = peak_time_ms

    # Log suspected missed peaks (only if PAR > 4)
    for p in missed_candidates:
        amp = env[p]
        par = amp / avg_env if avg_env > 0 else 0
        if par <= 4.0:
            continue

        now = datetime.datetime.now()
        date_str = now.date().isoformat()
        time_str = now.time().isoformat(timespec='microseconds')

        width_s = estimate_width(env, p, SAMPLE_RATE)
        width_ms = width_s * 1e3
        snr_db = 20 * math.log10(amp / noise_floor) if noise_floor > 0 else 0

        writer.writerow([
            date_str, time_str,
            f"{amp:.3f}", f"{width_ms:.2f}", f"{freq_hz/1e6:.3f}",
            f"{snr_db:.2f}", f"{par:.2f}", f"{noise_floor:.3f}", "SUSPECTED"
        ])
        print(f"Suspected missed peak at {time_str}, amp={amp:.3f}, width={width_ms:.2f} ms, "
              f"freq={freq_hz/1e6:.3f} MHz, SNR={snr_db:.2f} dB, PAR={par:.2f}, NoiseFloor={noise_floor:.3f}")

    # Live plot update
    if live_plot and ax_env is not None:
        ax_env.clear()
        ax_env.plot(env, label="Envelope")
        ax_env.axhline(threshold_main, color='red', linestyle='--', label="Main Threshold")
        ax_env.axhline(threshold_low, color='orange', linestyle='--', label="Low Threshold")
        if len(merged_peaks) > 0:
            ax_env.scatter(merged_peaks, env[merged_peaks], color='green', label="Confirmed pulses")
        strong_suspects = [p for p in missed_candidates if (env[p]/avg_env if avg_env>0 else 0) > 3]
        if strong_suspects:
            ax_env.scatter(strong_suspects, env[strong_suspects], color='purple', marker='x', label="Suspected missed")
        ax_env.set_title(f"Envelope at {freq_hz/1e6:.3f} MHz")
        ax_env.set_xlabel("Sample index")
        ax_env.set_ylabel("Amplitude")
        ax_env.legend()
        plt.pause(0.01)

    return last_peak_time, new_tail
def main(freqs_hz, interval, live_plot=False):
    logging.info("=== Multi-Frequency Pulse Logger Started ===")
    logging.info(f"Frequencies: {', '.join([f'{f/1e6:.3f} MHz' for f in freqs_hz])}")
    logging.info(f"Sample rate: {SAMPLE_RATE/1e6:.3f} Msps, Gain: {GAIN}, live_plot={live_plot}")
    logging.info(f"Interval between frequencies: {interval:.1f} seconds")

    sdr = RtlSdr()
    sdr.sample_rate = SAMPLE_RATE
    sdr.gain = GAIN

    if live_plot:
        plt.ion()
        fig, ax_env = plt.subplots(figsize=(12, 6))
    else:
        ax_env = None

    snr_history = {}
    current_date = datetime.datetime.now().date()
    csv_file = build_csv_filename(freqs_hz, current_date)

    f = open(csv_file, mode='a', newline='')
    writer = csv.writer(f)
    if os.path.getsize(csv_file) == 0:
        writer.writerow([
            "Date", "Time (microseconds)", "Amplitude", "Width (ms)",
            "Frequency (MHz)", "SNR (dB)", "Peak/Average", "NoiseFloor", "Flag"
        ])
    logging.info(f"Using CSV file: {csv_file}")

    try:
        last_peak_time = None
        prev_tail = None
        while True:
            # Rotate file if date changed
            today = datetime.datetime.now().date()
            if today != current_date:
                f.close()
                current_date = today
                csv_file = build_csv_filename(freqs_hz, current_date)
                f = open(csv_file, mode='a', newline='')
                writer = csv.writer(f)
                writer.writerow([
                    "Date", "Time (microseconds)", "Amplitude", "Width (ms)",
                    "Frequency (MHz)", "SNR (dB)", "Peak/Average", "NoiseFloor", "Flag"
                ])
                logging.info(f"Rotated to new CSV file: {csv_file}")

            for freq_hz in freqs_hz:
                last_peak_time, prev_tail = process_frequency(
                    sdr, freq_hz, writer, live_plot, ax_env, snr_history,
                    last_peak_time=last_peak_time, prev_tail=prev_tail,
                    merge_window_ms=10, overlap_fraction=0.05
                )
                time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping multi-frequency logging...")
        logging.info("=== Pulse Logger Stopped ===")
    finally:
        f.close()
        sdr.close()
        if live_plot:
            plt.ioff()
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-frequency wildlife tag pulse logger with missed peak flag, PAR filter, software overlap, and daily rotation"
    )
    parser.add_argument("--plot", action="store_true", help="Enable live plotting of envelope")
    parser.add_argument("--freqs", type=float, nargs="+", default=DEFAULT_FREQS_MHZ,
                        help="List of center frequencies in MHz (space-separated)")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                        help="Interval between frequency scans in seconds (default 10)")
    args = parser.parse_args()

    freqs_hz = [f * 1e6 for f in args.freqs]

    print("Usage:")
    print("  python pulse_logger.py --freqs 163.082 164.000 --interval 15   # scan freqs every 15s")
    print("  python pulse_logger.py --plot                                  # enable live plot")

    main(freqs_hz=freqs_hz, interval=args.interval, live_plot=args.plot)

import numpy as np
import datetime
import csv
import argparse
from rtlsdr import RtlSdr
from scipy.signal import butter, lfilter, find_peaks

# Default parameters
DEFAULT_CENTER_FREQ = 163.082e6
DEFAULT_SAMPLE_RATE = 2.4e6
DEFAULT_GAIN = 'auto'
DEFAULT_BLOCK_SIZE = 262144
DEFAULT_THRESHOLD_MULT = 5.0
DEFAULT_MIN_WIDTH_MS = 1.0
PRI_WINDOW = 10  # number of pulses to average over

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Pulse counter SDR logger")
parser.add_argument("-f", "--freq", type=float, default=DEFAULT_CENTER_FREQ,
                    help=f"Center frequency in Hz (default: {DEFAULT_CENTER_FREQ})")
parser.add_argument("-r", "--rate", type=float, default=DEFAULT_SAMPLE_RATE,
                    help=f"Sample rate in Hz (default: {DEFAULT_SAMPLE_RATE})")
parser.add_argument("-g", "--gain", default=DEFAULT_GAIN,
                    help=f"Gain setting (default: {DEFAULT_GAIN})")
parser.add_argument("-b", "--block", type=int, default=DEFAULT_BLOCK_SIZE,
                    help=f"Block size (default: {DEFAULT_BLOCK_SIZE})")
parser.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD_MULT,
                    help=f"Threshold multiplier (default: {DEFAULT_THRESHOLD_MULT})")
parser.add_argument("-m", "--minwidth", type=float, default=DEFAULT_MIN_WIDTH_MS,
                    help=f"Minimum pulse width in ms (default: {DEFAULT_MIN_WIDTH_MS})")
args = parser.parse_args()

CENTER_FREQ = args.freq
SAMPLE_RATE = args.rate
GAIN = args.gain
BLOCK_SIZE = args.block
THRESHOLD_MULT = args.threshold
MIN_WIDTH_MS = args.minwidth

print("=== Pulse Counter SDR Logger ===")
print("You can override parameters from the command line, e.g.:")
print("    python pulsecounter.py --freq 162000000 --rate 1200000 --gain 20 --block 65536 --threshold 3 --minwidth 2.0")
print(f"Using center frequency: {CENTER_FREQ/1e6:.3f} MHz")
print(f"Sample rate: {SAMPLE_RATE}")
print(f"Gain: {GAIN}")
print(f"Block size: {BLOCK_SIZE}")
print(f"Threshold multiplier: {THRESHOLD_MULT}")
print(f"Minimum pulse width: {MIN_WIDTH_MS} ms")
print("Press Ctrl-C to stop logging.\n")

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

# Setup SDR
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN

b_bp, a_bp = design_bandpass(SAMPLE_RATE, 2000, 25000)

# Initialize CSV files for today
current_date = datetime.date.today()
data_filename = f"pulsecounter-data-{current_date.isoformat()}.csv"
meta_filename = f"pulsecounter-meta-{current_date.isoformat()}.csv"
start_time = datetime.datetime.now()

# Metadata file
with open(meta_filename, mode='w', newline='') as mf:
    meta_writer = csv.writer(mf)
    meta_writer.writerow(["Logging Metadata"])
    meta_writer.writerow(["Start Time", start_time.isoformat(timespec='seconds')])
    meta_writer.writerow(["Sample Rate", SAMPLE_RATE])
    meta_writer.writerow(["Center Frequency", CENTER_FREQ])
    meta_writer.writerow(["Gain", GAIN])
    meta_writer.writerow(["Block Size", BLOCK_SIZE])
    meta_writer.writerow(["Threshold Multiplier", THRESHOLD_MULT])
    meta_writer.writerow(["Minimum Width (ms)", MIN_WIDTH_MS])
    meta_writer.writerow(["Data File", data_filename])

# Data file
f = open(data_filename, mode='w', newline='')
writer = csv.writer(f)
writer.writerow([
    "Date", "Time (microseconds)", "Amplitude", "Width (ms)",
    "SNR (dB)", "PAR (dB)", "Noise Floor",
    "Time Since Last Peak (ms)", "Avg PRI (ms)"
])

last_peak_time = None
pri_list = []

try:
    while True:
        # Rotate files if date changes
        today = datetime.date.today()
        if today != current_date:
            f.close()
            stop_time = datetime.datetime.now()
            with open(meta_filename, mode='a', newline='') as mf:
                meta_writer = csv.writer(mf)
                meta_writer.writerow(["Stop Time", stop_time.isoformat(timespec='seconds')])

            current_date = today
            data_filename = f"pulsecounter-data-{current_date.isoformat()}.csv"
            meta_filename = f"pulsecounter-meta-{current_date.isoformat()}.csv"
            start_time = datetime.datetime.now()

            with open(meta_filename, mode='w', newline='') as mf:
                meta_writer = csv.writer(mf)
                meta_writer.writerow(["Logging Metadata"])
                meta_writer.writerow(["Start Time", start_time.isoformat(timespec='seconds')])
                meta_writer.writerow(["Sample Rate", SAMPLE_RATE])
                meta_writer.writerow(["Center Frequency", CENTER_FREQ])
                meta_writer.writerow(["Gain", GAIN])
                meta_writer.writerow(["Block Size", BLOCK_SIZE])
                meta_writer.writerow(["Threshold Multiplier", THRESHOLD_MULT])
                meta_writer.writerow(["Minimum Width (ms)", MIN_WIDTH_MS])
                meta_writer.writerow(["Data File", data_filename])

            f = open(data_filename, mode='w', newline='')
            writer = csv.writer(f)
            writer.writerow([
                "Date", "Time (microseconds)", "Amplitude", "Width (ms)",
                "SNR (dB)", "PAR (dB)", "Noise Floor",
                "Time Since Last Peak (ms)", "Avg PRI (ms)"
            ])

        samples = sdr.read_samples(BLOCK_SIZE * 10)
        bp = bandpass_filter(samples, b_bp, a_bp)
        bp = bp - np.mean(bp)
        env = np.abs(bp)

        noise_floor = np.median(env)
        threshold = noise_floor * THRESHOLD_MULT

        raw_peaks, _ = find_peaks(env, height=threshold)

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

        for p in merged_peaks:
            now = datetime.datetime.now()
            timestamp = now.time().isoformat(timespec='microseconds')
            amp = env[p]
            width_ms = estimate_width(env, p, SAMPLE_RATE) * 1e3

            # Skip pulses narrower than minwidth
            if width_ms < MIN_WIDTH_MS:
                continue

            snr = 20 * np.log10(amp / noise_floor) if noise_floor > 0 else 0
            par = 20 * np.log10(amp / np.mean(env)) if np.mean(env) > 0 else 0

            if last_peak_time is None:
                delta_ms = 0.0
            else:
                delta = now - last_peak_time
                delta_ms = delta.total_seconds() * 1e3
                pri_list.append(delta_ms)
                if len(pri_list) > PRI_WINDOW:
                    pri_list.pop(0)
            last_peak_time = now

            avg_pri = np.mean(pri_list) if len(pri_list) > 0 else 0.0

            writer.writerow([
                now.date().isoformat(), timestamp,
                f"{amp:.3f}", f"{width_ms:.2f}",
                f"{snr:.2f}", f"{par:.2f}", f"{noise_floor:.3f}",
                f"{delta_ms:.2f}", f"{avg_pri:.2f}"
            ])
            print(f"Pulse @ {timestamp}, amp={amp:.3f}, width={width_ms:.2f} ms, "
                  f"SNR={snr:.2f} dB, PAR={par:.2f} dB, "
                  f"NF={noise_floor:.3f}, Δt={delta_ms:.2f} ms, Avg PRI={avg_pri:.2f} ms")

except KeyboardInterrupt:
    stop_time = datetime.datetime.now()
    with open(meta_filename, mode='a', newline='') as mf:
        meta_writer = csv.writer(mf)
        meta_writer.writerow(["Stop Time", stop_time.isoformat(timespec='seconds')])
    print("\nStopping continuous logging...")
    f.close()

sdr.close()

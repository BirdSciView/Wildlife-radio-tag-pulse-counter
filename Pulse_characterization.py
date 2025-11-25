import numpy as np
import datetime
from rtlsdr import RtlSdr
from scipy.signal import butter, lfilter, find_peaks

# Parameters
CENTER_FREQ = 163.082e6   # Hz (adjust to your tag frequency)
SAMPLE_RATE = 2.4e6
GAIN = 'auto'
BLOCK_SIZE = 262144
BP_LOW, BP_HIGH = 2000, 25000   # Hz band-pass around baseband
THRESH_FACTOR = 6.0             # threshold multiplier
DECIM_FACTOR = 4                # decimation factor

# Band-pass filter
def design_bandpass(sr, low_hz, high_hz, order=4):
    nyq = 0.5 * sr
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def bandpass_filter(x, b, a):
    return lfilter(b, a, x)

def decimate_signal(x, factor):
    from scipy.signal import decimate
    if factor > 1:
        return decimate(x, factor, ftype='fir', zero_phase=True)
    return x

# Setup SDR
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN

b_bp, a_bp = design_bandpass(SAMPLE_RATE, BP_LOW, BP_HIGH)

print("Collecting samples...")
samples = sdr.read_samples(BLOCK_SIZE * 50)  # grab a chunk
sdr.close()

# Filter, DC removal, decimation
bp = bandpass_filter(samples, b_bp, a_bp)
bp = bp - np.mean(bp)  # remove DC offset
bp = decimate_signal(bp, DECIM_FACTOR)
effective_sr = SAMPLE_RATE // DECIM_FACTOR

# Envelope and smoothing
env = np.abs(bp)
win = max(1, int(0.003 * effective_sr))  # 3 ms smoothing window
env_s = np.convolve(env, np.ones(win)/win, mode='same')

# Threshold
mean_env = np.mean(env_s)
threshold = THRESH_FACTOR * mean_env

# Peak detection constraints
expected_interval_s = 1.0  # adjust if you know your tag’s repetition rate
min_distance = int(0.8 * expected_interval_s * effective_sr)  # 80% of expected interval
prom = 0.5 * threshold
w_min = int(0.008 * effective_sr)   # ~8 ms
w_max = int(0.080 * effective_sr)   # ~80 ms

peaks, props = find_peaks(env_s, height=threshold, distance=min_distance,
                          prominence=prom, width=(w_min, w_max))

# Characterize pulses
pulse_times = peaks / effective_sr
pulse_intervals = np.diff(pulse_times)
pulse_heights = props['peak_heights']

print("\n--- Pulse Characterization ---")
print(f"Detected {len(peaks)} pulses")
if len(pulse_intervals) > 0:
    print(f"Mean interval: {np.mean(pulse_intervals):.3f} s")
    print(f"Std interval: {np.std(pulse_intervals):.3f} s")
print(f"Mean amplitude: {np.mean(pulse_heights):.2f}")

# crude width estimate from props
if "widths" in props:
    widths = props["widths"] / effective_sr
    print(f"Mean width: {np.mean(widths)*1e3:.2f} ms")
    print(f"Std width: {np.std(widths)*1e3:.2f} ms")

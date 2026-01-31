#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PulseCounter SDR Logger — Normal mode + --zero-mode (Pi Zero 2 W optimized)

Normal mode (default):
- Uses SciPy FIR decimator (kaiserord/firwin/upfirdn) + find_peaks

Zero mode (--zero-mode):
- Uses CIC decimator + lightweight threshold-crossing peak detector
- Uses EMA smoothing (cheap) and keeps amplitude-domain logging consistent
- Auto-adjusts defaults unless user explicitly provided those args
- Does NOT batch CSV writes (writes each row immediately)
"""

import numpy as np
import datetime
import csv
import argparse
import time
import statistics
import subprocess
from collections import deque
from rtlsdr import RtlSdr

# -----------------------------
# Default configuration values
# -----------------------------
DEFAULT_CENTER_FREQ = 163.557e6  # Wildlife telemetry frequency

# "Full" defaults (original-ish)
FULL_DEFAULT_SAMPLE_RATE = 2.4e6
FULL_DEFAULT_BLOCK_SIZE = 262144
FULL_DEFAULT_TARGET_FS_DEC = 30_000.0

# "Zero-mode" defaults (Pi-friendly)
ZERO_DEFAULT_SAMPLE_RATE = 1.024e6
ZERO_DEFAULT_BLOCK_SIZE = 131072
ZERO_DEFAULT_TARGET_FS_DEC = 25_600.0

DEFAULT_GAIN = None
DEFAULT_THRESHOLD_MULT = 5.0
DEFAULT_MIN_WIDTH_MS = 1.0
DEFAULT_SCAN_TIME = 30.0
PRI_WINDOW = 10

# FIR decimation design defaults (normal mode)
PB_FRAC = 0.50
TRANS_FRAC = 0.15
RIPPLE_DB = 60.0

# Envelope smoothing
ENV_SMOOTH_MS = 3.0

# Read multiplier (how many blocks per read_samples call)
FULL_READ_MULT = 10
ZERO_READ_MULT = 4


# -----------------------------
# Command-line argument parser
# -----------------------------
parser = argparse.ArgumentParser(
    description="Pulse counter SDR logger with scanning, bias-T, decimation, and overload detection"
)

parser.add_argument("-f", "--freq", type=float, nargs="+", default=[DEFAULT_CENTER_FREQ],
                    help="Base frequencies in Hz.")
parser.add_argument("--offsets", type=float, nargs="*", default=[0],
                    help="Manual frequency offsets in Hz.")
parser.add_argument("--autodiscover", action="store_true",
                    help="Auto-discover strongest offset (+/- 10 kHz, 1 kHz steps).")
parser.add_argument("--biast", action="store_true",
                    help="Enable bias-T (antenna power). No hardware detection performed.")

# Important: default=None so we can detect what user explicitly set
parser.add_argument("-r", "--rate", type=float, default=None,
                    help="Input sample rate in Hz (normal default: 2.4e6; zero-mode default: 1.024e6)")
parser.add_argument("-g", "--gain", type=float, default=DEFAULT_GAIN,
                    help="Gain in dB (float). Omit for automatic gain (~38.6 dB).")
parser.add_argument("-b", "--block", type=int, default=None,
                    help="Block size at input sample rate (normal default: 262144; zero-mode default: 131072)")
parser.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD_MULT,
                    help="Threshold multiplier on the decimated envelope (e.g., 5.0)")
parser.add_argument("-m", "--minwidth", type=float, default=DEFAULT_MIN_WIDTH_MS,
                    help="Minimum pulse width in ms")
parser.add_argument("-s", "--scantime", type=float, default=DEFAULT_SCAN_TIME,
                    help="Seconds to spend on each frequency")

# Decimation control (default=None so we can apply mode defaults)
parser.add_argument("--target-fs-dec", type=float, default=None,
                    help="Target decimated sample rate in Hz (normal default: 30000; zero-mode default: 25600). Ignored if --decim is set.")
parser.add_argument("--decim", type=int, default=None,
                    help="Explicit integer decimation factor (overrides --target-fs-dec).")

# Overload detection options
parser.add_argument("--overload-stepdown", action="store_true",
                    help="Automatically step the tuner gain down one notch on overload.")
parser.add_argument("--overload-debug", action="store_true",
                    help="Print overload metrics every block (verbose).")

# Zero-mode switch
parser.add_argument("--zero-mode", action="store_true",
                    help="Pi Zero 2 W optimized mode: CIC decimation + lightweight peak detector; auto-adjusts defaults unless user set them.")

args = parser.parse_args()
ZERO_MODE = args.zero_mode

# Conditional SciPy import: only needed in normal mode
if not ZERO_MODE:
    from scipy.signal import kaiserord, firwin, upfirdn, find_peaks


# -----------------------------
# Apply mode defaults (only for args the user did NOT set)
# -----------------------------
if args.rate is None:
    args.rate = ZERO_DEFAULT_SAMPLE_RATE if ZERO_MODE else FULL_DEFAULT_SAMPLE_RATE

if args.block is None:
    args.block = ZERO_DEFAULT_BLOCK_SIZE if ZERO_MODE else FULL_DEFAULT_BLOCK_SIZE

if args.target_fs_dec is None:
    args.target_fs_dec = ZERO_DEFAULT_TARGET_FS_DEC if ZERO_MODE else FULL_DEFAULT_TARGET_FS_DEC

# Assign parsed values
FREQ_LIST = args.freq
OFFSETS = args.offsets
SAMPLE_RATE = float(args.rate)
GAIN = args.gain
BLOCK_SIZE = int(args.block)
THRESHOLD_MULT = float(args.threshold)
MIN_WIDTH_MS = float(args.minwidth)
SCAN_TIME = float(args.scantime)
AUTO_STEPDOWN = args.overload_stepdown
OVERLOAD_DEBUG = args.overload_debug
TARGET_FS_DEC_ARG = float(args.target_fs_dec)
DECIM_ARG = args.decim

READ_MULT = ZERO_READ_MULT if ZERO_MODE else FULL_READ_MULT


# ---------------------------------------------------
# Bias-T control helpers
# ---------------------------------------------------
def bias_t_on():
    subprocess.run(["rtl_biast", "-b", "1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Bias-T ENABLED (antenna power ON)")

def bias_t_off():
    subprocess.run(["rtl_biast", "-b", "0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Bias-T DISABLED (antenna power OFF)")


# ---------------------------------------------------
# Pulse width at half-maximum (works at any sample rate)
# ---------------------------------------------------
def estimate_width(env, peak_idx, sr):
    if peak_idx <= 0 or peak_idx >= len(env):
        return 0.0
    half_height = env[peak_idx] / 2.0
    left = peak_idx
    while left > 0 and env[left] > half_height:
        left -= 1
    right = peak_idx
    while right < len(env) and env[right] > half_height:
        right += 1
    return (right - left) / sr


# ---------------------------------------------------
# Automatic offset discovery (simple max envelope scan)
# ---------------------------------------------------
def discover_best_offset(base_freq, sdr, block_size, sample_rate):
    OFFSET_RANGE = 10000
    OFFSET_STEP = 1000

    best_offset = 0
    best_score = 0

    print(f"  Auto-discovering offset for {base_freq/1e6:.6f} MHz...")

    for off in range(-OFFSET_RANGE, OFFSET_RANGE + 1, OFFSET_STEP):
        test_freq = base_freq + off
        sdr.center_freq = test_freq

        samples = sdr.read_samples(block_size * 2).astype(np.complex64, copy=False)
        env = np.abs(samples)
        score = float(np.max(env)) if env.size else 0.0

        if score > best_score:
            best_score = score
            best_offset = off

    print(f"  Best offset: {best_offset/1000:.1f} kHz -> using {(base_freq + best_offset)/1e6:.6f} MHz")
    return base_freq + best_offset


# ---------------------------------------------------
# FIR decimator (normal mode)
# ---------------------------------------------------
class FIRDecimator:
    def __init__(self, fs_in, target_fs, pb_frac=PB_FRAC, trans_frac=TRANS_FRAC, ripple_db=RIPPLE_DB, explicit_decim=None):
        self.fs_in = float(fs_in)

        if explicit_decim is not None and explicit_decim >= 1:
            self.decim = int(explicit_decim)
            self.fs_dec = self.fs_in / self.decim
        else:
            self.decim = max(1, int(round(self.fs_in / float(target_fs))))
            self.fs_dec = self.fs_in / self.decim

        nyq_in = self.fs_in / 2.0
        nyq_dec = self.fs_dec / 2.0

        fp = pb_frac * nyq_dec
        tw = trans_frac * nyq_dec
        width_norm = max(min(tw / nyq_in, 0.999), 1e-6)

        N, beta = kaiserord(ripple_db, width_norm)
        if N % 2 == 0:
            N += 1
        cutoff_norm = fp / nyq_in
        self.taps = firwin(N, cutoff=cutoff_norm, window=('kaiser', beta))

        self._in_tail = np.zeros(len(self.taps) - 1, dtype=np.complex64)

    def process(self, x: np.ndarray) -> np.ndarray:
        x_in = np.concatenate((self._in_tail, x.astype(np.complex64, copy=False)))
        y = upfirdn(self.taps, x_in, up=1, down=self.decim)
        trim = (len(self.taps) - 1) // self.decim
        if trim > 0 and y.size > trim:
            y = y[trim:]
        self._in_tail = x_in[-(len(self.taps) - 1):].copy()
        return y


# ---------------------------------------------------
# CIC decimator (zero-mode)
# ---------------------------------------------------
class CICDecimator:
    """
    Fast decimator for Pi Zero 2 W.
    Order=2 CIC is typically sufficient for envelope/pulse detection.
    """
    def __init__(self, fs_in, target_fs, explicit_decim=None, order=2):
        self.fs_in = float(fs_in)
        if explicit_decim is not None and explicit_decim >= 1:
            self.decim = int(explicit_decim)
        else:
            self.decim = max(1, int(round(self.fs_in / float(target_fs))))
        self.fs_dec = self.fs_in / self.decim

        self.order = int(max(1, order))
        self.int_state = np.zeros(self.order, dtype=np.complex64)
        self.comb_state = np.zeros(self.order, dtype=np.complex64)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.complex64, copy=False)

        y = x
        for k in range(self.order):
            y = np.cumsum(y, dtype=np.complex64) + self.int_state[k]
            self.int_state[k] = y[-1]

        y = y[::self.decim]

        for k in range(self.order):
            prev = self.comb_state[k]
            out = np.empty_like(y)
            out[0] = y[0] - prev
            out[1:] = y[1:] - y[:-1]
            self.comb_state[k] = y[-1]
            y = out

        return y


class EMAFilter:
    """Cheap envelope smoothing with a 1-pole EMA, applied to power then sqrt back to amplitude."""
    def __init__(self, fs, smooth_ms=3.0):
        tau = max(1e-6, smooth_ms / 1000.0)
        self.alpha = 1.0 - np.exp(-1.0 / max(1.0, fs * tau))
        self.state = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x, dtype=np.float32)
        s = float(self.state)
        a = float(self.alpha)
        for i in range(x.size):
            s = s + a * (float(x[i]) - s)
            y[i] = s
        self.state = s
        return y


class PeakTracker:
    """
    Threshold-crossing pulse detector:
    returns peak indices (one per above-threshold region),
    with refractory samples to merge close pulses.
    """
    def __init__(self, fs_dec, merge_guard_ms=50.0):
        self.fs_dec = float(fs_dec)
        self.merge_guard = int((merge_guard_ms / 1000.0) * self.fs_dec)
        self.in_pulse = False
        self.peak_idx = 0
        self.peak_val = 0.0
        self.refractory = 0

    def find_peaks(self, env: np.ndarray, thr: float):
        peaks = []
        above = env > thr
        n = env.size
        i = 0

        while i < n:
            if self.refractory > 0:
                skip = min(self.refractory, n - i)
                self.refractory -= skip
                i += skip
                continue

            if not self.in_pulse:
                if above[i]:
                    self.in_pulse = True
                    self.peak_idx = i
                    self.peak_val = float(env[i])
                i += 1
            else:
                v = float(env[i])
                if v > self.peak_val:
                    self.peak_val = v
                    self.peak_idx = i

                if not above[i]:
                    peaks.append(self.peak_idx)
                    self.in_pulse = False
                    self.refractory = self.merge_guard
                i += 1

        return peaks


# ---------------------------------------------------
# Overload detection (keep your original monitor; it works in both modes)
# ---------------------------------------------------
class OverloadMonitor:
    """
    Detects ADC clipping and RF gain compression from IQ amplitude statistics.
    Designed for pyrtlsdr complex64 streams scaled ~[-1, 1].
    """
    def __init__(self,
                 clip_thr=0.98,
                 crest_min=2.2,
                 crest_relax=2.6,
                 p99_min=0.85,
                 p999_min=0.985,
                 kurt_min=-0.5,
                 kurt_relax=-0.2,
                 clip_ratio_min=1e-4,
                 clip_ratio_relax=5e-6,
                 rms_hist_len=50):
        self.clip_thr = float(clip_thr)
        self.crest_min = float(crest_min)
        self.crest_relax = float(crest_relax)
        self.p99_min = float(p99_min)
        self.p999_min = float(p999_min)
        self.kurt_min = float(kurt_min)
        self.kurt_relax = float(kurt_relax)
        self.clip_ratio_min = float(clip_ratio_min)
        self.clip_ratio_relax = float(clip_ratio_relax)

        self.rms_hist = deque(maxlen=int(rms_hist_len))
        self.overloaded = False

    def update(self, iq_block: np.ndarray):
        I = iq_block.real
        Q = iq_block.imag
        mag = np.abs(iq_block)

        rms = float(np.sqrt(np.mean(mag * mag)) + 1e-20)
        peak = float(np.max(mag) + 1e-20)
        crest = float(peak / rms)

        clip_hits = np.logical_or(np.abs(I) >= self.clip_thr, np.abs(Q) >= self.clip_thr)
        clip_ratio = float(np.mean(clip_hits))

        p99 = float(np.percentile(mag, 99.0))
        p999 = float(np.percentile(mag, 99.9))

        mu = float(np.mean(mag))
        sigma = float(np.std(mag) + 1e-20)
        kurt = float(np.mean(((mag - mu) / sigma) ** 4) - 3.0)

        self.rms_hist.append(rms)
        rms_floor = float(np.median(self.rms_hist)) if self.rms_hist else rms

        if not self.overloaded:
            conds = 0
            conds += int(clip_ratio > self.clip_ratio_min)
            conds += int((crest < self.crest_min) and (rms > 1.5 * rms_floor))
            conds += int((p999 > self.p999_min) and (p99 > self.p99_min))
            conds += int(kurt < self.kurt_min)
            if conds >= 2:
                self.overloaded = True
        else:
            if (clip_ratio < self.clip_ratio_relax and
                crest > self.crest_relax and
                p999 < 0.97 and
                kurt > self.kurt_relax):
                self.overloaded = False

        return {
            "rms": rms,
            "peak": peak,
            "crest": crest,
            "clip_ratio": clip_ratio,
            "p99": p99,
            "p999": p999,
            "kurt": kurt,
            "rms_floor": rms_floor,
            "overloaded": self.overloaded
        }


def maybe_stepdown_gain(sdr: RtlSdr, verbose=True):
    """Step the tuner gain down one notch (if possible)."""
    try:
        current = float(sdr.gain)
    except Exception:
        return None

    new_gain = None
    try:
        gains = sorted(set(sdr.get_gains()))
        lower = [g for g in gains if g < current]
        if lower:
            new_gain = max(lower)
    except Exception:
        possible = [0.0, 9.9, 14.4, 19.7, 22.9, 25.4, 28.0, 32.8, 37.2, 38.6, 42.1, 49.6]
        lower = [g for g in possible if g < current]
        if lower:
            new_gain = max(lower)

    if new_gain is not None and new_gain < current:
        try:
            sdr.gain = new_gain
            if verbose:
                print(f"[OVERLOAD] Reducing tuner gain: {current:.1f} dB -> {new_gain:.1f} dB")
            return new_gain
        except Exception:
            pass
    return None


# -----------------------------
# Initialize SDR
# -----------------------------
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE

# Bias-T
if args.biast:
    print("Bias-T requested -> enabling")
    bias_t_on()
    biast_status = "ON"
else:
    print("Bias-T not requested -> disabling")
    bias_t_off()
    biast_status = "OFF"

# Gain handling
if GAIN is None:
    print("Auto gain requested -> using default gain of 38.6 dB")
    sdr.gain = 38.6
else:
    sdr.gain = float(GAIN)

# Build decimator
if ZERO_MODE:
    decimator = CICDecimator(fs_in=SAMPLE_RATE, target_fs=TARGET_FS_DEC_ARG, explicit_decim=DECIM_ARG, order=2)
    ema = EMAFilter(fs=decimator.fs_dec, smooth_ms=ENV_SMOOTH_MS)
    peak_tracker = PeakTracker(fs_dec=decimator.fs_dec, merge_guard_ms=50.0)
    FS_DEC = decimator.fs_dec
    print(f"[Zero-Mode] Fs_in={SAMPLE_RATE/1e6:.3f} MS/s, decim={decimator.decim} -> Fs_dec={FS_DEC/1e3:.1f} kS/s")
else:
    decimator = FIRDecimator(fs_in=SAMPLE_RATE, target_fs=TARGET_FS_DEC_ARG,
                             pb_frac=PB_FRAC, trans_frac=TRANS_FRAC,
                             ripple_db=RIPPLE_DB, explicit_decim=DECIM_ARG)
    FS_DEC = decimator.fs_dec
    print(f"[Decimator] Fs_in={SAMPLE_RATE/1e6:.3f} MS/s, decim={decimator.decim} -> Fs_dec={FS_DEC/1e3:.1f} kS/s")

    ENV_SMOOTH_WIN = max(1, int((ENV_SMOOTH_MS / 1000.0) * FS_DEC))
    ENV_SMOOTH_KERNEL = np.ones(ENV_SMOOTH_WIN, dtype=float) / float(ENV_SMOOTH_WIN)

    def smooth_envelope(env_dec: np.ndarray) -> np.ndarray:
        if ENV_SMOOTH_WIN <= 1:
            return env_dec
        return np.convolve(env_dec, ENV_SMOOTH_KERNEL, mode='same')

if FS_DEC < 8000:
    print(f"[WARN] Very low decimated rate ({FS_DEC:.0f} Hz). Ensure your tag is stable and threshold is tuned.")

# Overload monitor
ol = OverloadMonitor()


# ---------------------------------------------------
# Build final frequency list
# ---------------------------------------------------
EXPANDED_FREQ_LIST = []
if args.autodiscover:
    print("=== Automatic Offset Discovery Enabled ===")
    for base in FREQ_LIST:
        real_freq = discover_best_offset(base, sdr, BLOCK_SIZE, SAMPLE_RATE)
        EXPANDED_FREQ_LIST.append(real_freq)
else:
    for base in FREQ_LIST:
        for off in OFFSETS:
            EXPANDED_FREQ_LIST.append(base + off)

print("Expanded scan frequencies (MHz):", [round(f/1e6, 6) for f in EXPANDED_FREQ_LIST])


# ---------------------------------------------------
# Prepare CSV output files
# ---------------------------------------------------
current_date = datetime.date.today()
data_filename = f"pulsecounter-data-{current_date.isoformat()}.csv"
meta_filename = f"pulsecounter-meta-{current_date.isoformat()}.csv"
start_time = datetime.datetime.now()

with open(meta_filename, mode='w', newline='') as mf:
    meta_writer = csv.writer(mf)
    meta_writer.writerow(["Logging Metadata"])
    meta_writer.writerow(["Start Time", start_time.isoformat(timespec='seconds')])
    meta_writer.writerow(["Zero Mode", "ON" if ZERO_MODE else "OFF"])
    meta_writer.writerow(["Sample Rate (input) Hz", SAMPLE_RATE])
    meta_writer.writerow(["Decimation Factor", decimator.decim])
    meta_writer.writerow(["Sample Rate (decimated) Hz", FS_DEC])
    meta_writer.writerow(["Frequencies", ";".join(str(f) for f in EXPANDED_FREQ_LIST)])
    meta_writer.writerow(["Gain", "AUTO" if GAIN is None else GAIN])
    meta_writer.writerow(["Bias-T", biast_status])
    meta_writer.writerow(["Input Block Size (samples)", BLOCK_SIZE])
    meta_writer.writerow(["Read Multiplier", READ_MULT])
    meta_writer.writerow(["Threshold Multiplier", THRESHOLD_MULT])
    meta_writer.writerow(["Minimum Width (ms)", MIN_WIDTH_MS])
    meta_writer.writerow(["Scan Time (s)", SCAN_TIME])
    meta_writer.writerow(["Data File", data_filename])

f = open(data_filename, mode='w', newline='')
writer = csv.writer(f)
writer.writerow([
    "Date", "Time (microseconds)", "Frequency (Hz)",
    "Amplitude (decimated)", "Peak (dB)", "Width (ms)",
    "SNR (dB)", "PAR (dB)", "Noise Floor (decimated)",
    "Time Since Last Peak (ms)", "Avg PRI (ms)", "Mode PRI (ms)",
    "Overloaded"
])

# Per-frequency PRI state
freq_state = {fr: {"last_peak_time": None, "pri_list": []} for fr in EXPANDED_FREQ_LIST}


# ---------------------------------------------------
# Main scanning loop
# ---------------------------------------------------
try:
    while True:
        for freq in EXPANDED_FREQ_LIST:
            sdr.center_freq = freq
            print(f"--- Scanning {freq/1e6:.6f} MHz ---")
            scan_start = time.time()

            while (time.time() - scan_start) < SCAN_TIME:
                # Read chunk; in zero-mode we reduce READ_MULT to ease CPU/memory
                samples = sdr.read_samples(BLOCK_SIZE * READ_MULT).astype(np.complex64, copy=False)

                # Overload detection on raw IQ
                ol_metrics = ol.update(samples)
                overloaded = ol_metrics["overloaded"]

                if OVERLOAD_DEBUG or overloaded:
                    msg = (f"[OVERLOAD={'YES' if overloaded else 'no '}] "
                           f"crest={ol_metrics['crest']:.2f} "
                           f"clip={ol_metrics['clip_ratio']:.2e} "
                           f"p99={ol_metrics['p99']:.3f} p999={ol_metrics['p999']:.3f} "
                           f"kurt={ol_metrics['kurt']:.2f}")
                    print(msg)

                if overloaded and AUTO_STEPDOWN:
                    maybe_stepdown_gain(sdr, verbose=True)

                # DC removal
                samples = samples - np.mean(samples)

                # Decimate
                dec = decimator.process(samples)

                # Envelope + smoothing + peak finding
                if ZERO_MODE:
                    # Smooth in power domain (cheap), convert back to amplitude for consistency
                    env2 = (dec.real * dec.real) + (dec.imag * dec.imag)
                    env2 = env2.astype(np.float32, copy=False)
                    env2 = ema.process(env2)
                    env = np.sqrt(env2, dtype=np.float32)  # amplitude envelope

                    # Keep noise_floor consistent (median of amplitude)
                    noise_floor = float(np.median(env)) if env.size else 0.0
                    threshold = noise_floor * THRESHOLD_MULT

                    # Lightweight peak detection (one peak per above-threshold region)
                    merged_peaks = peak_tracker.find_peaks(env, threshold)

                else:
                    env = np.abs(dec)
                    if ENV_SMOOTH_WIN > 1:
                        env = smooth_envelope(env)

                    noise_floor = float(np.median(env)) if env.size else 0.0
                    threshold = noise_floor * THRESHOLD_MULT

                    raw_peaks, _ = find_peaks(env, height=threshold)

                    # Merge close peaks (50 ms guard)
                    merged_peaks = []
                    if len(raw_peaks) > 0:
                        current = raw_peaks[0]
                        merge_samps = int(0.05 * FS_DEC)
                        for p in raw_peaks[1:]:
                            if (p - current) < merge_samps:
                                if env[p] > env[current]:
                                    current = p
                            else:
                                merged_peaks.append(current)
                                current = p
                        merged_peaks.append(current)

                # Process peaks (identical math in both modes for consistency)
                for p in merged_peaks:
                    now = datetime.datetime.now()
                    timestamp = now.time().isoformat(timespec='microseconds')
                    amp = float(env[p])

                    peak_db = 20.0 * np.log10(amp) if amp > 0 else -999.0

                    width_ms = estimate_width(env, p, FS_DEC) * 1e3
                    if width_ms < MIN_WIDTH_MS:
                        continue

                    snr = 20.0 * np.log10(amp / noise_floor) if noise_floor > 0 else 0.0
                    mean_env = float(np.mean(env)) if env.size > 0 else 0.0
                    par = 20.0 * np.log10(amp / mean_env) if mean_env > 0 else 0.0

                    state = freq_state[freq]
                    if state["last_peak_time"] is None:
                        delta_ms = 0.0
                    else:
                        delta = now - state["last_peak_time"]
                        delta_ms = delta.total_seconds() * 1e3
                        state["pri_list"].append(delta_ms)
                        if len(state["pri_list"]) > PRI_WINDOW:
                            state["pri_list"].pop(0)

                    state["last_peak_time"] = now

                    avg_pri = float(np.mean(state["pri_list"])) if state["pri_list"] else 0.0
                    try:
                        mode_pri = statistics.mode(state["pri_list"]) if state["pri_list"] else 0.0
                    except statistics.StatisticsError:
                        mode_pri = avg_pri

                    writer.writerow([
                        now.date().isoformat(), timestamp, f"{freq:.0f}",
                        f"{amp:.3f}", f"{peak_db:.2f}", f"{width_ms:.2f}",
                        f"{snr:.2f}", f"{par:.2f}", f"{noise_floor:.3f}",
                        f"{delta_ms:.2f}", f"{avg_pri:.2f}", f"{mode_pri:.2f}",
                        "TRUE" if overloaded else "FALSE"
                    ])

                    print(f"Pulse @ {timestamp}, freq={freq/1e6:.6f} MHz, amp={amp:.3f}, "
                          f"peak={peak_db:.2f} dB, width={width_ms:.2f} ms, "
                          f"SNR={snr:.2f} dB, PAR={par:.2f} dB, NF={noise_floor:.3f}, "
                          f"Δt={delta_ms:.2f} ms, Avg PRI={avg_pri:.2f}, Mode PRI={mode_pri:.2f}, "
                          f"Overloaded={'YES' if overloaded else 'no'}")

except KeyboardInterrupt:
    stop_time = datetime.datetime.now()
    with open(meta_filename, mode='a', newline='') as mf:
        meta_writer = csv.writer(mf)
        meta_writer.writerow(["Stop Time", stop_time.isoformat(timespec='seconds')])
    print("\nStopping continuous logging...")
    f.close()
    sdr.close()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PulseCounter SDR Logger — Decimation + Overload Detection (Configurable Decimation)
-----------------------------------------------------------------------------------
This script scans one or more RF frequencies using an RTL‑SDR receiver and detects
short RF pulses. It logs pulse characteristics (amplitude, peak dB, width, SNR,
PAR, PRI) to a CSV file and supports both manual offset scanning and automatic
offset discovery (±10 kHz in 1 kHz steps). It also supports enabling/disabling the
bias‑T (antenna power) unconditionally when requested by the user.

Enhancements:
- FIR anti-alias + aggressive decimation (configurable) to maximize sensitivity.
- OverloadMonitor detects ADC clipping and RF compression from IQ amplitude stats.
- Optional tuner gain step-down on overload (--overload-stepdown).

Added in this integrated version:
1) Sample-based timestamps for pulses:
   - pulse timestamp = TS_WALL0 + (global_decimated_sample_index / FS_DEC)
   - fixes PRI drift caused by processing delays / scheduling jitter

2) Per-frequency performance monitor:
   - measures read/proc/total times vs the real-time duration represented by the samples
   - prints every --perf-every blocks per frequency when --perf is enabled
   - warns when behind real-time
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
from scipy.signal import kaiserord, firwin, upfirdn, find_peaks

# -----------------------------
# Default configuration values
# -----------------------------
DEFAULT_CENTER_FREQ = 163.557e6  # Wildlife telemetry frequency
DEFAULT_SAMPLE_RATE = 2.4e6      # High-rate oversampling
DEFAULT_GAIN = None
DEFAULT_BLOCK_SIZE = 262144
DEFAULT_THRESHOLD_MULT = 5.0
DEFAULT_MIN_WIDTH_MS = 1.0
DEFAULT_SCAN_TIME = 30.0
PRI_WINDOW = 10

# Decimation defaults (can be overridden via CLI)
DEFAULT_TARGET_DECIMATED_RATE = 30_000.0
PB_FRAC = 0.50
TRANS_FRAC = 0.15
RIPPLE_DB = 60.0
ENV_SMOOTH_MS = 3.0

# -----------------------------
# Command-line argument parser
# -----------------------------
parser = argparse.ArgumentParser(
    description="Pulse counter SDR logger with scanning, bias‑T, configurable decimation, and overload detection"
)

parser.add_argument("-f", "--freq", type=float, nargs="+", default=[DEFAULT_CENTER_FREQ],
                    help="Base frequencies in Hz.")

parser.add_argument("--offsets", type=float, nargs="*", default=[0],
                    help="Manual frequency offsets in Hz.")

parser.add_argument("--autodiscover", action="store_true",
                    help="Auto-discover strongest offset (+/- 10 kHz, 1 kHz steps).")

parser.add_argument("--biast", action="store_true",
                    help="Enable bias‑T (antenna power). No hardware detection performed.")

parser.add_argument("-r", "--rate", type=float, default=DEFAULT_SAMPLE_RATE,
                    help="Input sample rate in Hz (e.g., 2.4e6)")

parser.add_argument("-g", "--gain", type=float, default=DEFAULT_GAIN,
                    help="Gain in dB (float). Omit for automatic gain (~38.6 dB).")

parser.add_argument("-b", "--block", type=int, default=DEFAULT_BLOCK_SIZE,
                    help="Block size at the input sample rate")

parser.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD_MULT,
                    help="Threshold multiplier on the decimated envelope (e.g., 5.0)")

parser.add_argument("-m", "--minwidth", type=float, default=DEFAULT_MIN_WIDTH_MS,
                    help="Minimum pulse width in ms")

parser.add_argument("-s", "--scantime", type=float, default=DEFAULT_SCAN_TIME,
                    help="Seconds to spend on each frequency")

# Decimation control
parser.add_argument("--target-fs-dec", type=float, default=DEFAULT_TARGET_DECIMATED_RATE,
                    help="Target decimated sample rate in Hz (e.g., 20000 for ~20 kS/s). Ignored if --decim is set.")
parser.add_argument("--decim", type=int, default=None,
                    help="Explicit integer decimation factor (overrides --target-fs-dec).")

# Overload detection options
parser.add_argument("--overload-stepdown", action="store_true",
                    help="Automatically step the tuner gain down one notch on overload.")
parser.add_argument("--overload-debug", action="store_true",
                    help="Print overload metrics every block (verbose).")

# Performance monitor options
parser.add_argument("--perf", action="store_true",
                    help="Print periodic performance monitor lines per frequency.")
parser.add_argument("--perf-every", type=int, default=10,
                    help="Print perf line every N blocks per frequency (default: 10).")
parser.add_argument("--perf-warn", type=float, default=1.05,
                    help="Warn if total_time/block_duration exceeds this ratio (default: 1.05).")

args = parser.parse_args()

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

PERF = bool(args.perf)
PERF_EVERY = max(1, int(args.perf_every))
PERF_WARN = float(args.perf_warn)

# ---------------------------------------------------
# Bias‑T control helpers
# ---------------------------------------------------
def bias_t_on():
    subprocess.run(["rtl_biast", "-b", "1"])
    print("Bias‑T ENABLED (antenna power ON)")

def bias_t_off():
    subprocess.run(["rtl_biast", "-b", "0"])
    print("Bias‑T DISABLED (antenna power OFF)")

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

        samples = sdr.read_samples(block_size * 2)
        env = np.abs(samples)
        score = float(np.max(env)) if env.size else 0.0

        if score > best_score:
            best_score = score
            best_offset = off

    print(f"  Best offset: {best_offset/1000:.1f} kHz → using {(base_freq + best_offset)/1e6:.6f} MHz")
    return base_freq + best_offset

# ---------------------------------------------------
# FIR decimator (polyphase) with overlap-save
# ---------------------------------------------------
class FIRDecimator:
    def __init__(self, fs_in, target_fs=DEFAULT_TARGET_DECIMATED_RATE, pb_frac=PB_FRAC, trans_frac=TRANS_FRAC,
                 ripple_db=RIPPLE_DB, explicit_decim=None):
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
# Overload detection (IQ amplitude statistics)
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
                print(f"[OVERLOAD] Reducing tuner gain: {current:.1f} dB → {new_gain:.1f} dB")
            return new_gain
        except Exception:
            pass
    return None

# -----------------------------
# Initialize SDR
# -----------------------------
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE

# Bias‑T control
if args.biast:
    print("Bias‑T requested → enabling")
    bias_t_on()
    biast_status = "ON"
else:
    print("Bias‑T not requested → disabling")
    bias_t_off()
    biast_status = "OFF"

# Gain handling
if GAIN is None:
    print("Auto gain requested → using default gain of 38.6 dB")
    sdr.gain = 38.6
else:
    sdr.gain = float(GAIN)

# Build FIR decimator with CLI control
decimator = FIRDecimator(
    fs_in=SAMPLE_RATE,
    target_fs=TARGET_FS_DEC_ARG,
    pb_frac=PB_FRAC,
    trans_frac=TRANS_FRAC,
    ripple_db=RIPPLE_DB,
    explicit_decim=DECIM_ARG
)
FS_DEC = decimator.fs_dec
print(f"[Decimator] Fs_in={SAMPLE_RATE/1e6:.3f} MS/s, decim={decimator.decim} → Fs_dec={FS_DEC/1e3:.1f} kS/s")

if FS_DEC < 8000:
    print(f"[WARN] Very low decimated rate ({FS_DEC:.0f} Hz). Ensure your tag is stable and threshold is tuned.")

# Envelope smoothing window (in decimated samples)
ENV_SMOOTH_WIN = max(1, int((ENV_SMOOTH_MS / 1000.0) * FS_DEC))
ENV_SMOOTH_KERNEL = np.ones(ENV_SMOOTH_WIN, dtype=float) / float(ENV_SMOOTH_WIN)

def smooth_envelope(env_dec: np.ndarray) -> np.ndarray:
    if ENV_SMOOTH_WIN <= 1:
        return env_dec
    return np.convolve(env_dec, ENV_SMOOTH_KERNEL, mode='same')

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

# Sample-based timestamp patch: fixed wall-clock origin + global decimated sample counter
TS_WALL0 = time.time()
DEC_SAMP_COUNTER = 0

# Performance monitor state (per frequency)
perf_state = {fr: {"count": 0, "ema_ratio": None} for fr in EXPANDED_FREQ_LIST}

# Metadata file
with open(meta_filename, mode='w', newline='') as mf:
    meta_writer = csv.writer(mf)
    meta_writer.writerow(["Logging Metadata"])
    meta_writer.writerow(["Start Time", start_time.isoformat(timespec='seconds')])
    meta_writer.writerow(["Timestamp Mode", "sample-based (TS_WALL0 + sample_index/FS_DEC)"])
    meta_writer.writerow(["Perf Monitor", "ON" if PERF else "OFF"])
    meta_writer.writerow(["Perf Every (blocks)", PERF_EVERY])
    meta_writer.writerow(["Perf Warn Ratio", PERF_WARN])
    meta_writer.writerow(["Sample Rate (input) Hz", SAMPLE_RATE])
    meta_writer.writerow(["Decimation Factor", decimator.decim])
    meta_writer.writerow(["Sample Rate (decimated) Hz", FS_DEC])
    meta_writer.writerow(["Frequencies", ";".join(str(f) for f in EXPANDED_FREQ_LIST)])
    meta_writer.writerow(["Gain", "AUTO" if GAIN is None else GAIN])
    meta_writer.writerow(["Bias‑T", biast_status])
    meta_writer.writerow(["Input Block Size (samples)", BLOCK_SIZE])
    meta_writer.writerow(["Threshold Multiplier", THRESHOLD_MULT])
    meta_writer.writerow(["Minimum Width (ms)", MIN_WIDTH_MS])
    meta_writer.writerow(["Scan Time (s)", SCAN_TIME])
    meta_writer.writerow(["Data File", data_filename])

# Data CSV
f = open(data_filename, mode='w', newline='')
writer = csv.writer(f)
writer.writerow([
    "Date", "Time (microseconds)", "Frequency (Hz)",
    "Amplitude (decimated)", "Peak (dB)", "Width (ms)",
    "SNR (dB)", "PAR (dB)", "Noise Floor (decimated)",
    "Time Since Last Peak (ms)", "Avg PRI (ms)", "Mode PRI (ms)",
    "Overloaded"
])

# PRI state per frequency
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

            # Reset perf counters for this scan window (per frequency)
            perf_state[freq]["count"] = 0
            perf_state[freq]["ema_ratio"] = None

            while (time.time() - scan_start) < SCAN_TIME:
                # ---- Perf timing start ----
                t_loop0 = time.perf_counter()
                t_read0 = t_loop0

                # Read chunk at input Fs for better decimation efficiency
                samples = sdr.read_samples(BLOCK_SIZE * 10).astype(np.complex64, copy=False)

                t_read1 = time.perf_counter()

                # ------ Overload detection on raw IQ ------
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

                # DC removal (mitigate tuner DC offset)
                samples = samples - np.mean(samples)

                # ======== FIR anti-alias + decimation (complex IQ) ========
                dec = decimator.process(samples)  # complex64 at FS_DEC

                # Envelope at low rate; smooth ~3 ms to reduce spikes
                env = np.abs(dec)
                if ENV_SMOOTH_WIN > 1:
                    env = smooth_envelope(env)

                # Sample-based timestamp patch: establish sample index range for this block
                block_dec_start = DEC_SAMP_COUNTER
                DEC_SAMP_COUNTER += int(env.size)

                # Robust noise estimate & threshold on decimated envelope
                noise_floor = float(np.median(env)) if env.size else 0.0
                threshold = noise_floor * THRESHOLD_MULT

                # Peak picking at decimated rate
                raw_peaks, _ = find_peaks(env, height=threshold)

                # Merge close peaks (within 50 ms)
                merged_peaks = []
                if len(raw_peaks) > 0:
                    current = raw_peaks[0]
                    merge_samps = int(0.05 * FS_DEC)  # 50 ms guard
                    for p in raw_peaks[1:]:
                        if (p - current) < merge_samps:
                            if env[p] > env[current]:
                                current = p
                        else:
                            merged_peaks.append(current)
                            current = p
                    merged_peaks.append(current)

                for p in merged_peaks:
                    # Sample-based timestamp for this peak
                    t_peak = TS_WALL0 + (block_dec_start + int(p)) / FS_DEC
                    now = datetime.datetime.fromtimestamp(t_peak)
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

                # ---- Perf timing end ----
                t_loop1 = time.perf_counter()

                # Per-frequency perf monitor line (prints every PERF_EVERY blocks)
                if PERF:
                    block_dur_s = samples.size / float(SAMPLE_RATE) if SAMPLE_RATE > 0 else 0.0
                    read_s = t_read1 - t_read0
                    total_s = t_loop1 - t_loop0
                    proc_s = total_s - read_s

                    ratio = (total_s / block_dur_s) if block_dur_s > 0 else 0.0

                    ps = perf_state[freq]
                    ps["count"] += 1
                    if ps["ema_ratio"] is None:
                        ps["ema_ratio"] = ratio
                    else:
                        ps["ema_ratio"] = 0.9 * ps["ema_ratio"] + 0.1 * ratio

                    if (ps["count"] % PERF_EVERY) == 0:
                        warn = " **BEHIND**" if ratio > PERF_WARN else ""
                        print(f"[PERF {freq/1e6:.6f} MHz] read={read_s:.3f}s proc={proc_s:.3f}s total={total_s:.3f}s "
                              f"block={block_dur_s:.3f}s ratio={ratio:.2f} ema={ps['ema_ratio']:.2f}{warn}")

except KeyboardInterrupt:
    stop_time = datetime.datetime.now()
    with open(meta_filename, mode='a', newline='') as mf:
        meta_writer = csv.writer(mf)
        meta_writer.writerow(["Stop Time", stop_time.isoformat(timespec='seconds')])
    print("\nStopping continuous logging...")
    f.close()
    sdr.close()

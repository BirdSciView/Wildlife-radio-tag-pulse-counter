#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PulseCounter SDR Logger — Fully Integrated Peak Detection Improvements + Zero Mode
----------------------------------------------------------------------------------
Zero Mode (--zero-mode):
  - Disables overload statistics entirely (no overload monitor update, no percentiles/kurtosis).
  - Keeps all peak detection improvements: decimation, normalization, smoothing, matched filter,
    adaptive noise tracking, hysteresis detection, refractory, width constraints, PRI logging.

Other features:
- Scan 1+ frequencies (manual offsets or auto offset discovery).
- Bias‑T control via rtl_biast.
- FIR anti-alias + aggressive decimation (configurable).
- Optional overload detection + optional gain step-down (disabled in zero-mode).
- Logs pulse metrics to CSV.
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
from scipy.signal import kaiserord, firwin, upfirdn

# Optional (Savitzky–Golay smoothing)
try:
    from scipy.signal import savgol_filter
    HAS_SAVGOL = True
except Exception:
    HAS_SAVGOL = False

# -----------------------------
# Defaults
# -----------------------------
DEFAULT_CENTER_FREQ = 163.557e6
DEFAULT_SAMPLE_RATE = 2.4e6
DEFAULT_GAIN = None
DEFAULT_BLOCK_SIZE = 262144
DEFAULT_SCAN_TIME = 30.0
PRI_WINDOW = 10

# Decimation defaults
DEFAULT_TARGET_DECIMATED_RATE = 20_000.0
DEFAULT_PB_FRAC = 0.50
DEFAULT_TRANS_FRAC = 0.15
RIPPLE_DB = 60.0

# Pulse/detection defaults
DEFAULT_PULSE_MS = 25.0
DEFAULT_MIN_WIDTH_MS = 10.0
DEFAULT_MAX_WIDTH_MS = 60.0
DEFAULT_THR_HI = 7.0
DEFAULT_THR_LO = 5.0
DEFAULT_NOISE_ALPHA = 0.98
DEFAULT_REFRACTORY_MS = 150.0

# Envelope shaping
DEFAULT_ENV_SMOOTH_MS = 3.0
DEFAULT_NORM_MODE = "rms"     # none|mean|rms
DEFAULT_NORM_WIN_MS = 200.0

# Overload detection defaults
DEFAULT_OVERLOAD_CLIP_THR = 0.995
DEFAULT_OVERLOAD_CREST_MIN = 2.2
DEFAULT_OVERLOAD_CREST_RELAX = 2.6
DEFAULT_OVERLOAD_P99_MIN = 0.85
DEFAULT_OVERLOAD_P999_MIN = 0.985
DEFAULT_OVERLOAD_KURT_MIN = -0.5
DEFAULT_OVERLOAD_KURT_RELAX = -0.2
DEFAULT_OVERLOAD_CLIP_RATIO_MIN = 1e-4
DEFAULT_OVERLOAD_CLIP_RATIO_RELAX = 5e-6

# -----------------------------
# CLI
# -----------------------------
parser = argparse.ArgumentParser(
    description="Wildlife telemetry pulse logger with decimation + matched filter + hysteresis + optional overload detection"
)

# Frequency scanning
parser.add_argument("-f", "--freq", type=float, nargs="+", default=[DEFAULT_CENTER_FREQ],
                    help="Base frequencies in Hz.")
parser.add_argument("--offsets", type=float, nargs="*", default=[0],
                    help="Manual frequency offsets in Hz.")
parser.add_argument("--autodiscover", action="store_true",
                    help="Auto-discover strongest offset (+/- 10 kHz, 1 kHz steps).")

# SDR config
parser.add_argument("--biast", action="store_true",
                    help="Enable bias‑T (antenna power).")
parser.add_argument("-r", "--rate", type=float, default=DEFAULT_SAMPLE_RATE,
                    help="Input sample rate in Hz.")
parser.add_argument("-g", "--gain", type=float, default=DEFAULT_GAIN,
                    help="Tuner gain in dB. Omit to use default (38.6 dB).")
parser.add_argument("--disable-agc", action="store_true", default=True,
                    help="Disable RTL/tuner AGC (recommended). Default: enabled.")
parser.add_argument("-b", "--block", type=int, default=DEFAULT_BLOCK_SIZE,
                    help="Block size at input rate.")
parser.add_argument("-s", "--scantime", type=float, default=DEFAULT_SCAN_TIME,
                    help="Seconds per frequency.")

# Decimation control
parser.add_argument("--target-fs-dec", type=float, default=DEFAULT_TARGET_DECIMATED_RATE,
                    help="Target decimated sample rate (Hz). Ignored if --decim is set.")
parser.add_argument("--decim", type=int, default=None,
                    help="Explicit integer decimation factor (overrides --target-fs-dec).")
parser.add_argument("--pb-frac", type=float, default=DEFAULT_PB_FRAC,
                    help="Passband fraction of decimated Nyquist (0..1).")
parser.add_argument("--trans-frac", type=float, default=DEFAULT_TRANS_FRAC,
                    help="Transition width fraction of decimated Nyquist (0..1).")

# Detection parameters
parser.add_argument("--pulse-ms", type=float, default=DEFAULT_PULSE_MS,
                    help="Expected pulse duration in ms for matched filter.")
parser.add_argument("--minwidth", type=float, default=DEFAULT_MIN_WIDTH_MS,
                    help="Minimum pulse width in ms.")
parser.add_argument("--maxwidth", type=float, default=DEFAULT_MAX_WIDTH_MS,
                    help="Maximum pulse width in ms.")
parser.add_argument("--thr-hi", type=float, default=DEFAULT_THR_HI,
                    help="High threshold multiplier (enter pulse).")
parser.add_argument("--thr-lo", type=float, default=DEFAULT_THR_LO,
                    help="Low threshold multiplier (exit pulse).")
parser.add_argument("--noise-alpha", type=float, default=DEFAULT_NOISE_ALPHA,
                    help="IIR smoothing factor for noise floor tracking (0..1).")
parser.add_argument("--refractory-ms", type=float, default=DEFAULT_REFRACTORY_MS,
                    help="Minimum time between detections (ms).")

# Envelope shaping / normalization
parser.add_argument("--env-smooth-ms", type=float, default=DEFAULT_ENV_SMOOTH_MS,
                    help="Moving-average smoothing on envelope (ms). 0 disables.")
parser.add_argument("--use-savgol", action="store_true",
                    help="Use Savitzky–Golay smoothing on envelope (requires SciPy).")
parser.add_argument("--savgol-win-ms", type=float, default=9.0,
                    help="SavGol window length (ms). Will be made odd and >= 5.")
parser.add_argument("--savgol-poly", type=int, default=2,
                    help="SavGol polynomial order (typically 2 or 3).")
parser.add_argument("--norm-mode", type=str, default=DEFAULT_NORM_MODE,
                    choices=["none", "mean", "rms"],
                    help="Local normalization mode: none|mean|rms.")
parser.add_argument("--norm-win-ms", type=float, default=DEFAULT_NORM_WIN_MS,
                    help="Window for local normalization (ms).")

# Overload options
parser.add_argument("--overload-stepdown", action="store_true",
                    help="Step tuner gain down one notch when overload is detected.")
parser.add_argument("--overload-debug", action="store_true",
                    help="Print overload metrics every block (verbose).")

# NEW: Zero Mode
parser.add_argument("--zero-mode", action="store_true",
                    help="Pi Zero-friendly mode: disables overload statistics entirely (no overload monitor, no stepdown).")

args = parser.parse_args()

# -----------------------------
# Parsed values
# -----------------------------
FREQ_LIST = args.freq
OFFSETS = args.offsets
SAMPLE_RATE = float(args.rate)
GAIN = args.gain
BLOCK_SIZE = int(args.block)
SCAN_TIME = float(args.scantime)

TARGET_FS_DEC = float(args.target_fs_dec)
DECIM_ARG = args.decim
PB_FRAC = float(args.pb_frac)
TRANS_FRAC = float(args.trans_frac)

PULSE_MS = float(args.pulse_ms)
MIN_WIDTH_MS = float(args.minwidth)
MAX_WIDTH_MS = float(args.maxwidth)
THR_HI_MULT = float(args.thr_hi)
THR_LO_MULT = float(args.thr_lo)
NOISE_ALPHA = float(args.noise_alpha)
REFRACTORY_MS = float(args.refractory_ms)

ENV_SMOOTH_MS = float(args.env_smooth_ms)
USE_SAVGOL = bool(args.use_savgol) and HAS_SAVGOL
SAVGOL_WIN_MS = float(args.savgol_win_ms)
SAVGOL_POLY = int(args.savgol_poly)

NORM_MODE = args.norm_mode
NORM_WIN_MS = float(args.norm_win_ms)

ZERO_MODE = bool(args.zero_mode)
OVERLOAD_DEBUG = bool(args.overload_debug) and (not ZERO_MODE)
AUTO_STEPDOWN = bool(args.overload_stepdown) and (not ZERO_MODE)

# ---------------------------------------------------
# Bias‑T control
# ---------------------------------------------------
def bias_t_on():
    subprocess.run(["rtl_biast", "-b", "1"])
    print("Bias‑T ENABLED (antenna power ON)")

def bias_t_off():
    subprocess.run(["rtl_biast", "-b", "0"])
    print("Bias‑T DISABLED (antenna power OFF)")

# ---------------------------------------------------
# Width at half maximum
# ---------------------------------------------------
def estimate_width_halfmax(env, peak_idx, sr):
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
# Offset discovery
# ---------------------------------------------------
def discover_best_offset(base_freq, sdr, block_size):
    OFFSET_RANGE = 10000
    OFFSET_STEP = 1000
    best_offset = 0
    best_score = 0.0
    print(f"  Auto-discovering offset for {base_freq/1e6:.6f} MHz...")
    for off in range(-OFFSET_RANGE, OFFSET_RANGE + 1, OFFSET_STEP):
        sdr.center_freq = base_freq + off
        samples = sdr.read_samples(block_size * 2)
        score = float(np.max(np.abs(samples))) if len(samples) else 0.0
        if score > best_score:
            best_score = score
            best_offset = off
    print(f"  Best offset: {best_offset/1000:.1f} kHz → using {(base_freq + best_offset)/1e6:.6f} MHz")
    return base_freq + best_offset

# ---------------------------------------------------
# Moving mean/RMS
# ---------------------------------------------------
def moving_mean_same(x, win):
    n = len(x)
    if win <= 1 or n == 0:
        return x.astype(float, copy=False) if x.dtype != float else x
    win = int(win)
    if win > n:
        m = float(np.mean(x)) if n else 0.0
        return np.full(n, m, dtype=float)
    c = np.cumsum(np.insert(x.astype(float, copy=False), 0, 0.0))
    valid = (c[win:] - c[:-win]) / float(win)
    pad_left = win // 2
    pad_right = n - (len(valid) + pad_left)
    return np.pad(valid, (pad_left, pad_right), mode="edge")

def moving_rms_same(x, win):
    return np.sqrt(moving_mean_same(x * x, win) + 1e-12)

# ---------------------------------------------------
# Matched filter (rectangular moving sum)
# ---------------------------------------------------
def matched_filter_rect_same(x, win):
    n = len(x)
    if win <= 1 or n == 0:
        return x.astype(float, copy=False) if x.dtype != float else x
    win = int(win)
    if win > n:
        return np.full(n, float(np.sum(x)), dtype=float)
    c = np.cumsum(np.insert(x.astype(float, copy=False), 0, 0.0))
    valid = (c[win:] - c[:-win])
    pad_left = win // 2
    pad_right = n - (len(valid) + pad_left)
    return np.pad(valid, (pad_left, pad_right), mode="edge")

# ---------------------------------------------------
# FIR decimator
# ---------------------------------------------------
class FIRDecimator:
    def __init__(self, fs_in, target_fs, pb_frac, trans_frac, ripple_db, explicit_decim=None):
        self.fs_in = float(fs_in)
        if explicit_decim is not None and int(explicit_decim) >= 1:
            self.decim = int(explicit_decim)
            self.fs_dec = self.fs_in / self.decim
        else:
            self.decim = max(1, int(round(self.fs_in / float(target_fs))))
            self.fs_dec = self.fs_in / self.decim

        nyq_in = self.fs_in / 2.0
        nyq_dec = self.fs_dec / 2.0

        fp = float(pb_frac) * nyq_dec
        tw = float(trans_frac) * nyq_dec
        width_norm = max(min(tw / nyq_in, 0.999), 1e-6)

        N, beta = kaiserord(ripple_db, width_norm)
        if N % 2 == 0:
            N += 1
        cutoff_norm = fp / nyq_in
        self.taps = firwin(N, cutoff=cutoff_norm, window=("kaiser", beta))
        self._in_tail = np.zeros(len(self.taps) - 1, dtype=np.complex64)

    def reset(self):
        self._in_tail = np.zeros(len(self.taps) - 1, dtype=np.complex64)

    def process(self, x):
        x_in = np.concatenate((self._in_tail, x.astype(np.complex64, copy=False)))
        y = upfirdn(self.taps, x_in, up=1, down=self.decim)
        trim = (len(self.taps) - 1) // self.decim
        if trim > 0 and y.size > trim:
            y = y[trim:]
        self._in_tail = x_in[-(len(self.taps) - 1):].copy()
        return y

# ---------------------------------------------------
# Overload monitor (created only if not ZERO_MODE)
# ---------------------------------------------------
class OverloadMonitor:
    def __init__(self,
                 clip_thr=DEFAULT_OVERLOAD_CLIP_THR,
                 crest_min=DEFAULT_OVERLOAD_CREST_MIN,
                 crest_relax=DEFAULT_OVERLOAD_CREST_RELAX,
                 p99_min=DEFAULT_OVERLOAD_P99_MIN,
                 p999_min=DEFAULT_OVERLOAD_P999_MIN,
                 kurt_min=DEFAULT_OVERLOAD_KURT_MIN,
                 kurt_relax=DEFAULT_OVERLOAD_KURT_RELAX,
                 clip_ratio_min=DEFAULT_OVERLOAD_CLIP_RATIO_MIN,
                 clip_ratio_relax=DEFAULT_OVERLOAD_CLIP_RATIO_RELAX,
                 rms_hist_len=50,
                 min_abs_rms_gate=0.25,
                 warmup_blocks=10):
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
        self.min_abs_rms_gate = float(min_abs_rms_gate)
        self.warmup_blocks = int(warmup_blocks)
        self._updates = 0

    def update(self, iq):
        I = iq.real
        Q = iq.imag
        mag = np.abs(iq)

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

        self._updates += 1
        warmup = self._updates < self.warmup_blocks

        if not self.overloaded:
            conds = 0
            conds += int(clip_ratio > self.clip_ratio_min)
            crest_rule = (crest < self.crest_min) and (rms > max(1.5 * rms_floor, self.min_abs_rms_gate)) and (not warmup)
            conds += int(crest_rule)
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
            "crest": crest, "clip_ratio": clip_ratio,
            "p99": p99, "p999": p999, "kurt": kurt,
            "overloaded": self.overloaded
        }

def maybe_stepdown_gain(sdr, verbose=True):
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
        sdr.gain = new_gain
        if verbose:
            print(f"[OVERLOAD] Reducing tuner gain: {current:.1f} dB → {new_gain:.1f} dB")
        return new_gain
    return None

# ---------------------------------------------------
# Hysteresis pulse detection + refractory
# ---------------------------------------------------
def detect_pulses_hysteresis(metric, hi_thr, lo_thr, fs_dec, min_width_ms, max_width_ms, refractory_ms):
    n = len(metric)
    if n == 0:
        return []

    min_samps = int((min_width_ms / 1000.0) * fs_dec)
    max_samps = int((max_width_ms / 1000.0) * fs_dec)
    ref_samps = int((refractory_ms / 1000.0) * fs_dec)

    pulses = []
    inside = False
    start = 0

    for i, v in enumerate(metric):
        if not inside:
            if v > hi_thr:
                inside = True
                start = i
        else:
            if v < lo_thr:
                end = i
                length = end - start
                if min_samps <= length <= max_samps:
                    peak = start + int(np.argmax(metric[start:end]))
                    pulses.append((start, end, peak))
                inside = False

    if inside:
        end = n
        length = end - start
        if min_samps <= length <= max_samps:
            peak = start + int(np.argmax(metric[start:end]))
            pulses.append((start, end, peak))

    # Refractory: keep strongest in each refractory window
    if not pulses or ref_samps <= 1:
        return pulses

    filtered = []
    pending = None
    for s, e, p in pulses:
        if pending is None:
            pending = (s, e, p)
            continue
        if (p - pending[2]) < ref_samps:
            if metric[p] > metric[pending[2]]:
                pending = (s, e, p)
        else:
            filtered.append(pending)
            pending = (s, e, p)
    if pending is not None:
        filtered.append(pending)
    return filtered

# ---------------------------------------------------
# SDR setup
# ---------------------------------------------------
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE

if args.biast:
    print("Bias‑T requested → enabling")
    bias_t_on()
    biast_status = "ON"
else:
    print("Bias‑T not requested → disabling")
    bias_t_off()
    biast_status = "OFF"

if args.disable_agc:
    try:
        sdr.rtl_agc = False
    except Exception:
        pass
    try:
        sdr.set_agc_mode(0)
    except Exception:
        pass
    try:
        sdr.set_manual_gain_enabled(True)
    except Exception:
        pass

if GAIN is None:
    print("Gain not provided → using default gain of 38.6 dB")
    sdr.gain = 38.6
else:
    sdr.gain = float(GAIN)

# Decimator
decimator = FIRDecimator(SAMPLE_RATE, TARGET_FS_DEC, PB_FRAC, TRANS_FRAC, RIPPLE_DB, DECIM_ARG)
FS_DEC = decimator.fs_dec
print(f"[Decimator] Fs_in={SAMPLE_RATE/1e6:.3f} MS/s, decim={decimator.decim} → Fs_dec={FS_DEC/1e3:.1f} kS/s")

# Derived counts
pulse_len_samps = max(1, int((PULSE_MS / 1000.0) * FS_DEC))
env_smooth_samps = max(1, int((ENV_SMOOTH_MS / 1000.0) * FS_DEC)) if ENV_SMOOTH_MS > 0 else 1
norm_win_samps = max(1, int((NORM_WIN_MS / 1000.0) * FS_DEC)) if NORM_MODE != "none" else 1

# SavGol params
def _savgol_params(fs_dec, win_ms, poly):
    win = int(round((win_ms / 1000.0) * fs_dec))
    win = max(win, 5)
    if win % 2 == 0:
        win += 1
    poly = max(1, int(poly))
    if poly >= win:
        poly = win - 2
    return win, poly

savgol_win, savgol_poly = _savgol_params(FS_DEC, SAVGOL_WIN_MS, SAVGOL_POLY)

# Overload monitor only if not ZERO_MODE
ol = None if ZERO_MODE else OverloadMonitor()
if ZERO_MODE:
    print("[Zero Mode] Overload statistics DISABLED (no overload monitor, no stepdown).")

# Expanded frequency list
EXPANDED_FREQ_LIST = []
if args.autodiscover:
    print("=== Automatic Offset Discovery Enabled ===")
    for base in FREQ_LIST:
        EXPANDED_FREQ_LIST.append(discover_best_offset(base, sdr, BLOCK_SIZE))
else:
    for base in FREQ_LIST:
        for off in OFFSETS:
            EXPANDED_FREQ_LIST.append(base + off)

print("Expanded scan frequencies (MHz):", [round(f/1e6, 6) for f in EXPANDED_FREQ_LIST])

# CSV files
current_date = datetime.date.today()
data_filename = f"pulsecounter-data-{current_date.isoformat()}.csv"
meta_filename = f"pulsecounter-meta-{current_date.isoformat()}.csv"
start_time = datetime.datetime.now()

with open(meta_filename, mode="w", newline="") as mf:
    mw = csv.writer(mf)
    mw.writerow(["Logging Metadata"])
    mw.writerow(["Start Time", start_time.isoformat(timespec="seconds")])
    mw.writerow(["Zero Mode", str(ZERO_MODE)])
    mw.writerow(["Sample Rate (input) Hz", SAMPLE_RATE])
    mw.writerow(["Decimation Factor", decimator.decim])
    mw.writerow(["Sample Rate (decimated) Hz", FS_DEC])
    mw.writerow(["Frequencies", ";".join(str(f) for f in EXPANDED_FREQ_LIST)])
    mw.writerow(["Gain", "AUTO(38.6)" if GAIN is None else GAIN])
    mw.writerow(["Bias‑T", biast_status])
    mw.writerow(["Expected Pulse (ms)", PULSE_MS])
    mw.writerow(["Threshold Hi Mult", THR_HI_MULT])
    mw.writerow(["Threshold Lo Mult", THR_LO_MULT])
    mw.writerow(["Noise Alpha", NOISE_ALPHA])
    mw.writerow(["Refractory (ms)", REFRACTORY_MS])
    mw.writerow(["Env Smooth (ms)", ENV_SMOOTH_MS])
    mw.writerow(["Norm Mode", NORM_MODE])
    mw.writerow(["Norm Win (ms)", NORM_WIN_MS])
    mw.writerow(["Use SavGol", USE_SAVGOL])
    mw.writerow(["Scan Time (s)", SCAN_TIME])
    mw.writerow(["Data File", data_filename])

f = open(data_filename, mode="w", newline="")
writer = csv.writer(f)
writer.writerow([
    "Date", "Time (microseconds)", "Frequency (Hz)",
    "AmpEnv", "Peak(dB)", "Width(ms)",
    "SNR(dB)", "PAR(dB)", "NoiseFloorMetric",
    "TimeSinceLastPeak(ms)", "AvgPRI(ms)", "ModePRI(ms)",
    "MetricPeak", "Overloaded"
])

# State
freq_state = {fr: {"last_peak_time": None, "pri_list": [], "noise_floor": None} for fr in EXPANDED_FREQ_LIST}

# ---------------------------------------------------
# Main loop
# ---------------------------------------------------
try:
    while True:
        for freq in EXPANDED_FREQ_LIST:
            sdr.center_freq = freq
            decimator.reset()
            print(f"--- Scanning {freq/1e6:.6f} MHz ---")
            scan_start = time.time()

            while (time.time() - scan_start) < SCAN_TIME:
                samples = sdr.read_samples(BLOCK_SIZE * 2).astype(np.complex64, copy=False)

                # Overload block (disabled in zero mode)
                if ZERO_MODE:
                    overloaded = False
                else:
                    olm = ol.update(samples)
                    overloaded = bool(olm["overloaded"])
                    if OVERLOAD_DEBUG or overloaded:
                        print(f"[OVERLOAD={'YES' if overloaded else 'no '}] "
                              f"crest={olm['crest']:.2f} clip={olm['clip_ratio']:.2e} "
                              f"p99={olm['p99']:.3f} p999={olm['p999']:.3f} kurt={olm['kurt']:.2f}")
                    if overloaded and AUTO_STEPDOWN:
                        maybe_stepdown_gain(sdr, verbose=True)

                # DC remove
                samples = samples - np.mean(samples)

                # Decimate
                dec = decimator.process(samples)
                env = np.abs(dec).astype(float, copy=False)

                # Local normalization
                if NORM_MODE != "none" and norm_win_samps > 1:
                    if NORM_MODE == "mean":
                        denom = moving_mean_same(env, norm_win_samps) + 1e-12
                    else:
                        denom = moving_rms_same(env, norm_win_samps) + 1e-12
                    envn = env / denom
                else:
                    envn = env

                # Smooth
                if ENV_SMOOTH_MS > 0 and env_smooth_samps > 1:
                    envn = moving_mean_same(envn, env_smooth_samps)
                if USE_SAVGOL and len(envn) >= savgol_win:
                    envn = savgol_filter(envn, savgol_win, savgol_poly, mode="interp")

                # Matched filter metric
                metric = matched_filter_rect_same(envn, pulse_len_samps)

                # Adaptive noise floor
                st = freq_state[freq]
                block_med = float(np.median(metric)) if len(metric) else 0.0
                if st["noise_floor"] is None:
                    st["noise_floor"] = block_med
                else:
                    st["noise_floor"] = (NOISE_ALPHA * st["noise_floor"]) + ((1.0 - NOISE_ALPHA) * block_med)

                noise_floor = float(st["noise_floor"])
                hi_thr = noise_floor * THR_HI_MULT
                lo_thr = noise_floor * THR_LO_MULT

                pulses = detect_pulses_hysteresis(metric, hi_thr, lo_thr, FS_DEC, MIN_WIDTH_MS, MAX_WIDTH_MS, REFRACTORY_MS)

                for (s_idx, e_idx, p_idx) in pulses:
                    now = datetime.datetime.now()
                    timestamp = now.time().isoformat(timespec="microseconds")

                    amp = float(envn[p_idx]) if 0 <= p_idx < len(envn) else 0.0
                    metric_peak = float(metric[p_idx]) if 0 <= p_idx < len(metric) else 0.0
                    peak_db = 20.0 * np.log10(amp) if amp > 0 else -999.0

                    width_ms = estimate_width_halfmax(envn, p_idx, FS_DEC) * 1e3
                    if width_ms < MIN_WIDTH_MS or width_ms > MAX_WIDTH_MS:
                        continue

                    env_baseline = (noise_floor / max(1, pulse_len_samps))
                    snr = 20.0 * np.log10(amp / env_baseline) if env_baseline > 0 else 0.0
                    mean_env = float(np.mean(envn)) if len(envn) else 0.0
                    par = 20.0 * np.log10(amp / mean_env) if mean_env > 0 else 0.0

                    # PRI
                    if st["last_peak_time"] is None:
                        delta_ms = 0.0
                    else:
                        delta_ms = (now - st["last_peak_time"]).total_seconds() * 1e3
                        st["pri_list"].append(delta_ms)
                        if len(st["pri_list"]) > PRI_WINDOW:
                            st["pri_list"].pop(0)
                    st["last_peak_time"] = now

                    avg_pri = float(np.mean(st["pri_list"])) if st["pri_list"] else 0.0
                    try:
                        mode_pri = statistics.mode(st["pri_list"]) if st["pri_list"] else 0.0
                    except statistics.StatisticsError:
                        mode_pri = avg_pri

                    writer.writerow([
                        now.date().isoformat(), timestamp, f"{freq:.0f}",
                        f"{amp:.5f}", f"{peak_db:.2f}", f"{width_ms:.2f}",
                        f"{snr:.2f}", f"{par:.2f}", f"{noise_floor:.5f}",
                        f"{delta_ms:.2f}", f"{avg_pri:.2f}", f"{mode_pri:.2f}",
                        f"{metric_peak:.5f}", "TRUE" if overloaded else "FALSE"
                    ])

                    print(f"Pulse @ {timestamp}, freq={freq/1e6:.6f} MHz, "
                          f"amp={amp:.5f}, metric={metric_peak:.5f}, peak={peak_db:.2f} dB, "
                          f"width={width_ms:.2f} ms, SNR≈{snr:.2f} dB, "
                          f"NF(metric)={noise_floor:.5f}, Δt={delta_ms:.2f} ms, "
                          f"AvgPRI={avg_pri:.2f}, ModePRI={mode_pri:.2f}, "
                          f"Overloaded={'YES' if overloaded else 'no'}")

except KeyboardInterrupt:
    stop_time = datetime.datetime.now()
    with open(meta_filename, mode="a", newline="") as mf:
        mw = csv.writer(mf)
        mw.writerow(["Stop Time", stop_time.isoformat(timespec="seconds")])
    print("\nStopping continuous logging...")
    try:
        f.close()
    except Exception:
        pass

try:
    sdr.close()
except Exception:
    pass
``

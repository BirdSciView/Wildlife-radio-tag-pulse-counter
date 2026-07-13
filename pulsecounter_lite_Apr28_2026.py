#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PulseCounter SDR Logger — Integrated with Range Improvements
-------------------------------------------------------------
Improvements over original:
  1. LO frequency offset to avoid DC spike (replaces mean subtraction)
  2. Low-percentile noise floor estimator (replaces median — less biased)
  3. Async streaming via read_samples_async to avoid dropped pulses
  4. Configurable pulse merge guard (default 15 ms, was 50 ms)
  5. Narrow digital bandpass after decimation to reduce wideband noise
  6. PRI-coherent integration for weak signals once PRI is established

Original features retained:
  - FIR anti-alias + aggressive decimation (configurable)
  - OverloadMonitor with optional gain step-down
  - Bias-T control
  - Auto-discover offset scanning
  - CSV data + metadata logging
  - PAR / SNR / pulse width estimation
"""

import numpy as np
import datetime
import csv
import argparse
import time
import statistics
import subprocess
import threading
from collections import deque
from rtlsdr import RtlSdr
from scipy.signal import kaiserord, firwin, upfirdn, find_peaks, butter, sosfiltfilt

# -----------------------------------------------------------------------
# Default configuration values
# -----------------------------------------------------------------------
DEFAULT_CENTER_FREQ          = 163.557e6   # Wildlife telemetry frequency
DEFAULT_SAMPLE_RATE          = 2.4e6       # High-rate oversampling
DEFAULT_GAIN                 = None
DEFAULT_BLOCK_SIZE           = 262144
DEFAULT_THRESHOLD_MULT       = 5.0
DEFAULT_MIN_WIDTH_MS         = 1.0
DEFAULT_SCAN_TIME            = 30.0
PRI_WINDOW                   = 10

# Decimation defaults
DEFAULT_TARGET_DECIMATED_RATE = 30_000.0
PB_FRAC                      = 0.50
TRANS_FRAC                   = 0.15
RIPPLE_DB                    = 60.0
ENV_SMOOTH_MS                = 3.0

# LO offset to push the DC spike away from the signal of interest
DEFAULT_LO_OFFSET = 100e3   # Hz — tune SDR 100 kHz above target, mix back down

# Noise floor percentile (low = less biased when pulses are present)
DEFAULT_NOISE_PERCENTILE = 10   # 10th percentile

# Bandpass filter around the tag signal (applied after decimation)
DEFAULT_BP_BW_HZ = 4000.0        # ±2 kHz around baseband centre
DEFAULT_BP_ORDER = 4

# Coherent integration: number of PRI-aligned blocks to stack
DEFAULT_COHERENT_N = 4

# -----------------------------------------------------------------------
# Command-line argument parser
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=(
        "Pulse counter SDR logger with scanning, bias-T, configurable decimation, "
        "overload detection, LO offset, async streaming, and coherent integration."
    )
)

parser.add_argument("-f", "--freq", type=float, nargs="+", default=[DEFAULT_CENTER_FREQ],
                    help="Base frequencies in Hz.")
parser.add_argument("--offsets", type=float, nargs="*", default=[0],
                    help="Manual frequency offsets in Hz.")
parser.add_argument("--autodiscover", action="store_true",
                    help="Auto-discover strongest offset (±10 kHz, 1 kHz steps).")
parser.add_argument("--biast", action="store_true",
                    help="Enable bias-T (antenna power). No hardware detection performed.")
parser.add_argument("-r", "--rate", type=float, default=DEFAULT_SAMPLE_RATE,
                    help="Input sample rate in Hz (e.g. 2.4e6).")
parser.add_argument("-g", "--gain", type=float, default=DEFAULT_GAIN,
                    help="Gain in dB. Omit for automatic gain (~38.6 dB).")
parser.add_argument("-b", "--block", type=int, default=DEFAULT_BLOCK_SIZE,
                    help="Block size at the input sample rate.")
parser.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD_MULT,
                    help="Threshold multiplier on the decimated envelope (e.g. 5.0).")
parser.add_argument("-m", "--minwidth", type=float, default=DEFAULT_MIN_WIDTH_MS,
                    help="Minimum pulse width in ms.")
parser.add_argument("-s", "--scantime", type=float, default=DEFAULT_SCAN_TIME,
                    help="Seconds to spend on each frequency.")

# Decimation control
parser.add_argument("--target-fs-dec", type=float, default=DEFAULT_TARGET_DECIMATED_RATE,
                    help="Target decimated sample rate in Hz. Ignored if --decim is set.")
parser.add_argument("--decim", type=int, default=None,
                    help="Explicit integer decimation factor (overrides --target-fs-dec).")

# Improvement 1 — LO offset
parser.add_argument("--lo-offset", type=float, default=DEFAULT_LO_OFFSET,
                    help="LO frequency offset in Hz to shift DC spike away from signal (default 100 kHz).")

# Improvement 2 — noise floor percentile
parser.add_argument("--noise-percentile", type=float, default=DEFAULT_NOISE_PERCENTILE,
                    help="Percentile used for noise floor estimation (default 10). Lower = less biased when pulses are present.")

# Improvement 4 — merge guard
parser.add_argument("--merge-ms", type=float, default=15.0,
                    help="Pulse merge guard window in ms (default 15 ms; original was 50 ms).")

# Improvement 5 — bandpass filter
parser.add_argument("--bp-bw", type=float, default=DEFAULT_BP_BW_HZ,
                    help="Digital bandpass filter bandwidth in Hz applied after decimation (default 4000 Hz). Set 0 to disable.")
parser.add_argument("--bp-order", type=int, default=DEFAULT_BP_ORDER,
                    help="Butterworth bandpass filter order (default 4).")

# Improvement 6 — coherent integration
parser.add_argument("--coherent-n", type=int, default=DEFAULT_COHERENT_N,
                    help="Number of PRI-aligned blocks to stack for coherent integration (default 4, set 1 to disable).")

# Overload detection
parser.add_argument("--overload-stepdown", action="store_true",
                    help="Automatically step the tuner gain down one notch on overload.")
parser.add_argument("--overload-debug", action="store_true",
                    help="Print overload metrics every block (verbose).")

args = parser.parse_args()

# Assign parsed values
FREQ_LIST          = args.freq
OFFSETS            = args.offsets
SAMPLE_RATE        = args.rate
GAIN               = args.gain
BLOCK_SIZE         = args.block
THRESHOLD_MULT     = args.threshold
MIN_WIDTH_MS       = args.minwidth
SCAN_TIME          = args.scantime
AUTO_STEPDOWN      = args.overload_stepdown
OVERLOAD_DEBUG     = args.overload_debug
TARGET_FS_DEC_ARG  = args.target_fs_dec
DECIM_ARG          = args.decim
LO_OFFSET          = args.lo_offset
NOISE_PERCENTILE   = args.noise_percentile
MERGE_MS           = args.merge_ms
BP_BW              = args.bp_bw
BP_ORDER           = args.bp_order
COHERENT_N         = max(1, args.coherent_n)

# -----------------------------------------------------------------------
# Bias-T helpers
# -----------------------------------------------------------------------
def bias_t_on():
    subprocess.run(["rtl_biast", "-b", "1"])
    print("Bias-T ENABLED (antenna power ON)")

def bias_t_off():
    subprocess.run(["rtl_biast", "-b", "0"])
    print("Bias-T DISABLED (antenna power OFF)")

# -----------------------------------------------------------------------
# Pulse width at half-maximum
# -----------------------------------------------------------------------
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

# -----------------------------------------------------------------------
# Automatic offset discovery
# -----------------------------------------------------------------------
def discover_best_offset(base_freq, sdr, block_size, sample_rate, lo_offset):
    OFFSET_RANGE = 10000
    OFFSET_STEP  = 1000

    best_offset = 0
    best_score  = 0

    print(f"  Auto-discovering offset for {base_freq/1e6:.6f} MHz...")

    for off in range(-OFFSET_RANGE, OFFSET_RANGE + 1, OFFSET_STEP):
        test_freq = base_freq + off
        sdr.center_freq = test_freq + lo_offset   # apply LO offset during discovery too
        samples = sdr.read_samples(block_size * 2)
        env = np.abs(samples)
        score = float(np.max(env)) if env.size else 0.0
        if score > best_score:
            best_score = score
            best_offset = off

    print(f"  Best offset: {best_offset/1000:.1f} kHz → using {(base_freq + best_offset)/1e6:.6f} MHz")
    return base_freq + best_offset

# -----------------------------------------------------------------------
# FIR decimator (polyphase, overlap-save)
# -----------------------------------------------------------------------
class FIRDecimator:
    def __init__(self, fs_in, target_fs=DEFAULT_TARGET_DECIMATED_RATE,
                 pb_frac=PB_FRAC, trans_frac=TRANS_FRAC, ripple_db=RIPPLE_DB,
                 explicit_decim=None):
        self.fs_in = float(fs_in)

        if explicit_decim is not None and explicit_decim >= 1:
            self.decim  = int(explicit_decim)
            self.fs_dec = self.fs_in / self.decim
        else:
            self.decim  = max(1, int(round(self.fs_in / float(target_fs))))
            self.fs_dec = self.fs_in / self.decim

        nyq_in  = self.fs_in / 2.0
        nyq_dec = self.fs_dec / 2.0

        fp         = pb_frac * nyq_dec
        tw         = trans_frac * nyq_dec
        width_norm = max(min(tw / nyq_in, 0.999), 1e-6)

        N, beta = kaiserord(ripple_db, width_norm)
        if N % 2 == 0:
            N += 1
        cutoff_norm = fp / nyq_in
        self.taps   = firwin(N, cutoff=cutoff_norm, window=('kaiser', beta))

        self._in_tail = np.zeros(len(self.taps) - 1, dtype=np.complex64)

    def process(self, x: np.ndarray) -> np.ndarray:
        x_in = np.concatenate((self._in_tail, x.astype(np.complex64, copy=False)))
        y    = upfirdn(self.taps, x_in, up=1, down=self.decim)
        trim = (len(self.taps) - 1) // self.decim
        if trim > 0 and y.size > trim:
            y = y[trim:]
        self._in_tail = x_in[-(len(self.taps) - 1):].copy()
        return y

# -----------------------------------------------------------------------
# Improvement 5 — narrow digital bandpass filter (applied post-decimation)
# -----------------------------------------------------------------------
def build_bandpass_sos(center_offset_hz: float, bw_hz: float, fs: float, order: int):
    """
    Build a Butterworth bandpass SOS filter centred at `center_offset_hz` Hz
    within the decimated baseband.  For baseband (DC-centred) signals after
    LO mixing, center_offset_hz should be 0.
    """
    nyq  = fs / 2.0
    low  = max((center_offset_hz - bw_hz / 2.0) / nyq, 1e-4)
    high = min((center_offset_hz + bw_hz / 2.0) / nyq, 0.9999)
    return butter(order, [low, high], btype='band', output='sos')


# -----------------------------------------------------------------------
# Overload monitor
# -----------------------------------------------------------------------
class OverloadMonitor:
    def __init__(self,
                 clip_thr=0.98, crest_min=2.2, crest_relax=2.6,
                 p99_min=0.85, p999_min=0.985,
                 kurt_min=-0.5, kurt_relax=-0.2,
                 clip_ratio_min=1e-4, clip_ratio_relax=5e-6,
                 rms_hist_len=50):
        self.clip_thr        = float(clip_thr)
        self.crest_min       = float(crest_min)
        self.crest_relax     = float(crest_relax)
        self.p99_min         = float(p99_min)
        self.p999_min        = float(p999_min)
        self.kurt_min        = float(kurt_min)
        self.kurt_relax      = float(kurt_relax)
        self.clip_ratio_min  = float(clip_ratio_min)
        self.clip_ratio_relax= float(clip_ratio_relax)
        self.rms_hist        = deque(maxlen=int(rms_hist_len))
        self.overloaded      = False

    def update(self, iq_block: np.ndarray):
        I   = iq_block.real
        Q   = iq_block.imag
        mag = np.abs(iq_block)

        rms   = float(np.sqrt(np.mean(mag * mag)) + 1e-20)
        peak  = float(np.max(mag) + 1e-20)
        crest = float(peak / rms)

        clip_hits  = np.logical_or(np.abs(I) >= self.clip_thr, np.abs(Q) >= self.clip_thr)
        clip_ratio = float(np.mean(clip_hits))

        p99  = float(np.percentile(mag, 99.0))
        p999 = float(np.percentile(mag, 99.9))

        mu    = float(np.mean(mag))
        sigma = float(np.std(mag) + 1e-20)
        kurt  = float(np.mean(((mag - mu) / sigma) ** 4) - 3.0)

        self.rms_hist.append(rms)
        rms_floor = float(np.median(self.rms_hist)) if self.rms_hist else rms

        if not self.overloaded:
            conds  = 0
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
            "rms": rms, "peak": peak, "crest": crest,
            "clip_ratio": clip_ratio, "p99": p99, "p999": p999,
            "kurt": kurt, "rms_floor": rms_floor,
            "overloaded": self.overloaded,
        }


def maybe_stepdown_gain(sdr: RtlSdr, verbose=True):
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

# -----------------------------------------------------------------------
# Initialize SDR
# -----------------------------------------------------------------------
sdr = RtlSdr()
sdr.sample_rate = SAMPLE_RATE

if args.biast:
    print("Bias-T requested → enabling")
    bias_t_on()
    biast_status = "ON"
else:
    print("Bias-T not requested → disabling")
    bias_t_off()
    biast_status = "OFF"

if GAIN is None:
    print("Auto gain requested → using default gain of 38.6 dB")
    sdr.gain = 38.6
else:
    sdr.gain = float(GAIN)

# Build FIR decimator
decimator = FIRDecimator(
    fs_in=SAMPLE_RATE,
    target_fs=TARGET_FS_DEC_ARG,
    pb_frac=PB_FRAC,
    trans_frac=TRANS_FRAC,
    ripple_db=RIPPLE_DB,
    explicit_decim=DECIM_ARG,
)
FS_DEC = decimator.fs_dec
print(f"[Decimator] Fs_in={SAMPLE_RATE/1e6:.3f} MS/s, decim={decimator.decim} → Fs_dec={FS_DEC/1e3:.1f} kS/s")

if FS_DEC < 8000:
    print(f"[WARN] Very low decimated rate ({FS_DEC:.0f} Hz). Ensure tag is stable and threshold is tuned.")

# Build bandpass SOS filter (Improvement 5)
bp_sos = None
if BP_BW > 0:
    try:
        bp_sos = build_bandpass_sos(0.0, BP_BW, FS_DEC, BP_ORDER)
        print(f"[Bandpass] ±{BP_BW/2:.0f} Hz around baseband, order {BP_ORDER}")
    except Exception as e:
        print(f"[WARN] Could not build bandpass filter: {e}")

# Envelope smoothing
ENV_SMOOTH_WIN    = max(1, int((ENV_SMOOTH_MS / 1000.0) * FS_DEC))
ENV_SMOOTH_KERNEL = np.ones(ENV_SMOOTH_WIN, dtype=float) / float(ENV_SMOOTH_WIN)

def smooth_envelope(env_dec: np.ndarray) -> np.ndarray:
    if ENV_SMOOTH_WIN <= 1:
        return env_dec
    return np.convolve(env_dec, ENV_SMOOTH_KERNEL, mode='same')

# Overload monitor
ol = OverloadMonitor()

# -----------------------------------------------------------------------
# Build final frequency list
# -----------------------------------------------------------------------
EXPANDED_FREQ_LIST = []

if args.autodiscover:
    print("=== Automatic Offset Discovery Enabled ===")
    for base in FREQ_LIST:
        real_freq = discover_best_offset(base, sdr, BLOCK_SIZE, SAMPLE_RATE, LO_OFFSET)
        EXPANDED_FREQ_LIST.append(real_freq)
else:
    for base in FREQ_LIST:
        for off in OFFSETS:
            EXPANDED_FREQ_LIST.append(base + off)

print("Expanded scan frequencies (MHz):", [round(f / 1e6, 6) for f in EXPANDED_FREQ_LIST])
print(f"[LO offset] ±{LO_OFFSET/1e3:.1f} kHz (DC spike shifted away from signal)")
print(f"[Noise floor] {NOISE_PERCENTILE}th percentile estimator")
print(f"[Merge guard] {MERGE_MS:.1f} ms")
print(f"[Coherent integration] N={COHERENT_N} PRI-aligned blocks")

# -----------------------------------------------------------------------
# Prepare CSV output files
# -----------------------------------------------------------------------
current_date  = datetime.date.today()
data_filename = f"pulsecounter-data-{current_date.isoformat()}.csv"
meta_filename = f"pulsecounter-meta-{current_date.isoformat()}.csv"
start_time    = datetime.datetime.now()

with open(meta_filename, mode='w', newline='') as mf:
    mw = csv.writer(mf)
    mw.writerow(["Logging Metadata"])
    mw.writerow(["Start Time",                     start_time.isoformat(timespec='seconds')])
    mw.writerow(["Sample Rate (input) Hz",          SAMPLE_RATE])
    mw.writerow(["Decimation Factor",               decimator.decim])
    mw.writerow(["Sample Rate (decimated) Hz",      FS_DEC])
    mw.writerow(["Frequencies",                     ";".join(str(f) for f in EXPANDED_FREQ_LIST)])
    mw.writerow(["Gain",                            "AUTO" if GAIN is None else GAIN])
    mw.writerow(["Bias-T",                          biast_status])
    mw.writerow(["Input Block Size (samples)",      BLOCK_SIZE])
    mw.writerow(["Threshold Multiplier",            THRESHOLD_MULT])
    mw.writerow(["Minimum Width (ms)",              MIN_WIDTH_MS])
    mw.writerow(["Scan Time (s)",                   SCAN_TIME])
    mw.writerow(["LO Offset (Hz)",                  LO_OFFSET])
    mw.writerow(["Noise Floor Percentile",          NOISE_PERCENTILE])
    mw.writerow(["Bandpass BW (Hz)",                BP_BW if bp_sos is not None else "disabled"])
    mw.writerow(["Coherent Integration N",          COHERENT_N])
    mw.writerow(["Pulse Merge Guard (ms)",          MERGE_MS])
    mw.writerow(["Data File",                       data_filename])

csv_file   = open(data_filename, mode='w', newline='')
writer     = csv.writer(csv_file)
writer.writerow([
    "Date", "Time (microseconds)", "Frequency (Hz)",
    "Amplitude (decimated)", "Peak (dB)", "Width (ms)",
    "SNR (dB)", "PAR (dB)", "Noise Floor (decimated)",
    "Time Since Last Peak (ms)", "Avg PRI (ms)", "Mode PRI (ms)",
    "Coherent Stacks", "Overloaded",
])

# -----------------------------------------------------------------------
# Per-frequency state (PRI tracking + coherent buffer)
# -----------------------------------------------------------------------
freq_state = {
    f: {
        "last_peak_time": None,
        "pri_list":       [],
        "coh_buffer":     [],   # rolling list of env slices for coherent stacking
    }
    for f in EXPANDED_FREQ_LIST
}

# -----------------------------------------------------------------------
# Improvement 3 — async streaming helpers
# -----------------------------------------------------------------------
_raw_queue = deque(maxlen=40)   # holds raw complex64 blocks from callback
_queue_lock = threading.Lock()
_stream_active = threading.Event()
_stream_active.set()


def _stream_callback(samples, context):
    """Called by rtlsdr in a background thread for each block."""
    if not _stream_active.is_set():
        raise RuntimeError("stop")   # signal the async loop to stop
    with _queue_lock:
        _raw_queue.append(samples.astype(np.complex64, copy=True))


def start_async_stream(sdr_dev):
    """Launch async streaming in a daemon thread."""
    def _run():
        try:
            sdr_dev.read_samples_async(_stream_callback, num_samples=BLOCK_SIZE)
        except Exception:
            pass   # RuntimeError("stop") from callback or KeyboardInterrupt

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def stop_async_stream(sdr_dev):
    _stream_active.clear()
    try:
        sdr_dev.cancel_read_async()
    except Exception:
        pass


# -----------------------------------------------------------------------
# Improvement 1 — LO offset mixing helper
# -----------------------------------------------------------------------
def _lo_mix(samples: np.ndarray, lo_hz: float, fs: float) -> np.ndarray:
    """
    Multiply by complex exponential to shift signal down by lo_hz.
    This cancels the LO offset applied when setting center_freq,
    placing the signal of interest back at baseband (DC).
    """
    n  = np.arange(len(samples), dtype=np.float32)
    phasor = np.exp(-2j * np.pi * (lo_hz / fs) * n).astype(np.complex64)
    return samples * phasor


# -----------------------------------------------------------------------
# Improvement 6 — PRI-coherent integration helper
# -----------------------------------------------------------------------
def _coherent_integrate(env: np.ndarray, pri_ms: float, fs_dec: float, n_stacks: int) -> np.ndarray:
    """
    Stack `n_stacks` consecutive PRI-length windows and average their
    magnitude envelopes.  Returns a shortened envelope with improved SNR.
    Improvement is approximately √n_stacks in amplitude (~3 dB for n=4).
    """
    pri_samps = int(round((pri_ms / 1000.0) * fs_dec))
    if pri_samps < 4 or len(env) < pri_samps * n_stacks:
        return env  # not enough data — return as-is

    slices = np.stack(
        [env[i * pri_samps: (i + 1) * pri_samps] for i in range(n_stacks)],
        axis=0
    )
    return np.mean(slices, axis=0)


# -----------------------------------------------------------------------
# Block processing function (shared between sync fallback and async path)
# -----------------------------------------------------------------------
def process_block(samples: np.ndarray, freq: float, overloaded: bool):
    """Process one raw IQ block for the given tuned frequency."""
    state = freq_state[freq]

    # --- Improvement 1: LO mix-down (replaces mean subtraction) ---------
    samples = _lo_mix(samples, LO_OFFSET, SAMPLE_RATE)

    # --- FIR decimation --------------------------------------------------
    dec = decimator.process(samples)   # complex64 at FS_DEC

    # --- Improvement 5: narrow digital bandpass --------------------------
    if bp_sos is not None and len(dec) > BP_ORDER * 6:
        try:
            dec = sosfiltfilt(bp_sos, dec).astype(np.complex64)
        except Exception:
            pass   # skip filter if block too short

    # --- Envelope + smoothing --------------------------------------------
    env = np.abs(dec)
    if ENV_SMOOTH_WIN > 1:
        env = smooth_envelope(env)

    # --- Improvement 6: coherent integration (if PRI known) --------------
    avg_pri = float(np.mean(state["pri_list"])) if state["pri_list"] else 0.0
    coherent_stacks = 1

    if COHERENT_N > 1 and avg_pri > 0:
        env_coh = _coherent_integrate(env, avg_pri, FS_DEC, COHERENT_N)
        if len(env_coh) >= 4:
            env = env_coh
            coherent_stacks = COHERENT_N

    # --- Improvement 2: low-percentile noise floor -----------------------
    noise_floor = float(np.percentile(env, NOISE_PERCENTILE)) if env.size else 0.0
    threshold   = noise_floor * THRESHOLD_MULT

    # --- Peak picking ----------------------------------------------------
    raw_peaks, _ = find_peaks(env, height=threshold)

    # --- Improvement 4: configurable merge guard -------------------------
    merge_samps = int((MERGE_MS / 1000.0) * FS_DEC)
    merged_peaks = []
    if len(raw_peaks) > 0:
        current = raw_peaks[0]
        for p in raw_peaks[1:]:
            if (p - current) < merge_samps:
                if env[p] > env[current]:
                    current = p
            else:
                merged_peaks.append(current)
                current = p
        merged_peaks.append(current)

    # --- Log each detected pulse -----------------------------------------
    for p in merged_peaks:
        now       = datetime.datetime.now()
        timestamp = now.time().isoformat(timespec='microseconds')
        amp       = float(env[p])

        peak_db  = 20.0 * np.log10(amp) if amp > 0 else -999.0
        width_ms = estimate_width(env, p, FS_DEC) * 1e3
        if width_ms < MIN_WIDTH_MS:
            continue

        snr      = 20.0 * np.log10(amp / noise_floor) if noise_floor > 0 else 0.0
        mean_env = float(np.mean(env)) if env.size > 0 else 0.0
        par      = 20.0 * np.log10(amp / mean_env) if mean_env > 0 else 0.0

        if state["last_peak_time"] is None:
            delta_ms = 0.0
        else:
            delta    = now - state["last_peak_time"]
            delta_ms = delta.total_seconds() * 1e3
            state["pri_list"].append(delta_ms)
            if len(state["pri_list"]) > PRI_WINDOW:
                state["pri_list"].pop(0)

        state["last_peak_time"] = now

        avg_pri_log = float(np.mean(state["pri_list"])) if state["pri_list"] else 0.0
        try:
            mode_pri = statistics.mode(state["pri_list"]) if state["pri_list"] else 0.0
        except statistics.StatisticsError:
            mode_pri = avg_pri_log

        writer.writerow([
            now.date().isoformat(), timestamp, f"{freq:.0f}",
            f"{amp:.3f}", f"{peak_db:.2f}", f"{width_ms:.2f}",
            f"{snr:.2f}", f"{par:.2f}", f"{noise_floor:.3f}",
            f"{delta_ms:.2f}", f"{avg_pri_log:.2f}", f"{mode_pri:.2f}",
            coherent_stacks,
            "TRUE" if overloaded else "FALSE",
        ])

        print(
            f"Pulse @ {timestamp}, freq={freq/1e6:.6f} MHz, amp={amp:.3f}, "
            f"peak={peak_db:.2f} dB, width={width_ms:.2f} ms, "
            f"SNR={snr:.2f} dB, PAR={par:.2f} dB, NF={noise_floor:.3f}, "
            f"Δt={delta_ms:.2f} ms, Avg PRI={avg_pri_log:.2f}, "
            f"Mode PRI={mode_pri:.2f}, CohN={coherent_stacks}, "
            f"Overloaded={'YES' if overloaded else 'no'}"
        )


# -----------------------------------------------------------------------
# Main scanning loop (async streaming per frequency)
# -----------------------------------------------------------------------
try:
    while True:
        for freq in EXPANDED_FREQ_LIST:
            # Improvement 1: tune SDR to freq + LO_OFFSET so DC spike
            # lands at +LO_OFFSET Hz, not at the signal frequency.
            sdr.center_freq = freq + LO_OFFSET
            print(f"--- Scanning {freq/1e6:.6f} MHz (LO={( freq + LO_OFFSET)/1e6:.6f} MHz) ---")

            # Clear queue, start async stream for this frequency
            with _queue_lock:
                _raw_queue.clear()
            _stream_active.set()
            stream_thread = start_async_stream(sdr)

            scan_start = time.time()
            try:
                while (time.time() - scan_start) < SCAN_TIME:
                    # Drain whatever blocks have arrived
                    blocks_this_iter = 0
                    while True:
                        with _queue_lock:
                            if not _raw_queue:
                                break
                            samples = _raw_queue.popleft()
                        blocks_this_iter += 1

                        # Overload detection on raw IQ
                        ol_metrics = ol.update(samples)
                        overloaded = ol_metrics["overloaded"]

                        if OVERLOAD_DEBUG or overloaded:
                            print(
                                f"[OVERLOAD={'YES' if overloaded else 'no '}] "
                                f"crest={ol_metrics['crest']:.2f} "
                                f"clip={ol_metrics['clip_ratio']:.2e} "
                                f"p99={ol_metrics['p99']:.3f} "
                                f"p999={ol_metrics['p999']:.3f} "
                                f"kurt={ol_metrics['kurt']:.2f}"
                            )

                        if overloaded and AUTO_STEPDOWN:
                            maybe_stepdown_gain(sdr, verbose=True)

                        process_block(samples, freq, overloaded)

                    if blocks_this_iter == 0:
                        time.sleep(0.005)   # brief yield while waiting for data

            finally:
                stop_async_stream(sdr)
                stream_thread.join(timeout=2.0)

except KeyboardInterrupt:
    stop_time = datetime.datetime.now()
    with open(meta_filename, mode='a', newline='') as mf:
        mw = csv.writer(mf)
        mw.writerow(["Stop Time", stop_time.isoformat(timespec='seconds')])
    print("\nStopping continuous logging...")
    csv_file.close()

sdr.close()

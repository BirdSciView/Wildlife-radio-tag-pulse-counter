#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PulseCounter SDR Logger — v8
-----------------------------
Intended for wildlife radio tags where pulse-per-minute (PPM) rate is used
to estimate temperature.  Pulse detection reliability is the absolute priority.

Change from v7 — single bug fix, one structural cleanup:

  BUG FIX: last_accepted_time was updated BEFORE the width check, meaning a
  partial pulse fragment (6-7 ms wide, correctly rejected by --minwidth) still
  anchored the lockout gate clock.  This caused the following sequence:

    1. Real pulse straddles block boundary → partial fragment detected
    2. last_accepted_time updated → width check fires → pulse correctly rejected
    3. FIR tail of the fragment leaks ~78-95 ms later → incorrectly suppressed
       by lockout gate (78-95 ms is inside the 500 ms window)
    4. Real next pulse arrives ~1483 ms later → appears as ~5933 ms gap (4x PRI)
       in the log because last_accepted_time was left at the fragment timestamp

  Fix: last_accepted_time is now updated only after ALL checks pass (lockout,
  width, minwidth).  A candidate pulse that fails any check no longer affects
  the lockout gate or PRI state at all.

  CLEANUP: last_accepted_time and last_peak_time were two separate variables
  tracking the same moment (time of last fully accepted pulse) updated in
  different places.  Consolidated into a single variable: last_pulse_time.
  This eliminates the inconsistency that caused the bug above.

Everything else is identical to v7:
  - Mean subtraction for DC removal
  - Median noise floor estimator
  - Within-block merge guard (--merge-ms, default 15 ms)
  - Cross-block lockout gate (--lockout-ms, default 500 ms)
  - Minimum pulse width filter (--minwidth, default 20 ms)
  - PRI outlier rejection (delta > 2x running median excluded)
  - Per-pulse timestamps derived from sample position within block
  - Scale-independent overload monitor
  - Block read multiplier (--read-mult, default 2)
  - Per-frequency session summary on shutdown
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

# -----------------------------------------------------------------------
# Default configuration
# -----------------------------------------------------------------------
DEFAULT_CENTER_FREQ           = 163.557e6
DEFAULT_SAMPLE_RATE           = 2.4e6
DEFAULT_GAIN                  = None
DEFAULT_BLOCK_SIZE            = 262144
DEFAULT_THRESHOLD_MULT        = 5.0
DEFAULT_MIN_WIDTH_MS          = 20.0
DEFAULT_SCAN_TIME             = 30.0
PRI_WINDOW                    = 10

DEFAULT_TARGET_DECIMATED_RATE = 30_000.0
PB_FRAC                       = 0.50
TRANS_FRAC                    = 0.15
RIPPLE_DB                     = 60.0
ENV_SMOOTH_MS                 = 3.0

# -----------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=(
        "Pulse counter SDR logger — v8, tuned for PPM-based temperature estimation."
    )
)

parser.add_argument("-f", "--freq", type=float, nargs="+",
                    default=[DEFAULT_CENTER_FREQ],
                    help="Base frequencies in Hz.")
parser.add_argument("--offsets", type=float, nargs="*", default=[0],
                    help="Manual frequency offsets in Hz.")
parser.add_argument("--autodiscover", action="store_true",
                    help="Auto-discover strongest offset (±10 kHz, 1 kHz steps).")
parser.add_argument("--biast", action="store_true",
                    help="Enable bias-T (antenna power).")
parser.add_argument("-r", "--rate", type=float, default=DEFAULT_SAMPLE_RATE,
                    help="Input sample rate in Hz (e.g. 2.4e6).")
parser.add_argument("-g", "--gain", type=float, default=DEFAULT_GAIN,
                    help="Gain in dB. Omit for automatic gain (~38.6 dB).")
parser.add_argument("-b", "--block", type=int, default=DEFAULT_BLOCK_SIZE,
                    help="Block size at the input sample rate.")
parser.add_argument("--read-mult", type=int, default=2,
                    help=(
                        "Block read multiplier (default 2). "
                        "Samples read per call = BLOCK_SIZE * read-mult. "
                        "Reduce to 1 if missed-pulse gaps persist."
                    ))
parser.add_argument("-t", "--threshold", type=float,
                    default=DEFAULT_THRESHOLD_MULT,
                    help="Threshold multiplier on the decimated envelope (default 5.0).")
parser.add_argument("-m", "--minwidth", type=float, default=DEFAULT_MIN_WIDTH_MS,
                    help=(
                        "Minimum accepted pulse width in ms (default 20.0 ms). "
                        "Rejects partial pulses at block boundaries. "
                        "Set to ~75%% of your tag's expected pulse width."
                    ))
parser.add_argument("-s", "--scantime", type=float, default=DEFAULT_SCAN_TIME,
                    help="Seconds to spend on each frequency.")
parser.add_argument("--target-fs-dec", type=float,
                    default=DEFAULT_TARGET_DECIMATED_RATE,
                    help="Target decimated sample rate in Hz (default 30000). "
                         "Ignored if --decim is set.")
parser.add_argument("--decim", type=int, default=None,
                    help="Explicit integer decimation factor (overrides --target-fs-dec).")
parser.add_argument("--merge-ms", type=float, default=15.0,
                    help="Within-block pulse merge guard in ms (default 15 ms).")
parser.add_argument("--lockout-ms", type=float, default=500.0,
                    help=(
                        "Cross-block duplicate suppression window in ms (default 500 ms). "
                        "Only applied after a pulse passes all other checks. "
                        "Rule of thumb: keep below PRI / 3."
                    ))
parser.add_argument("--overload-stepdown", action="store_true",
                    help="Automatically step tuner gain down one notch on overload.")
parser.add_argument("--overload-debug", action="store_true",
                    help="Print overload metrics every block (verbose).")

args = parser.parse_args()

FREQ_LIST         = args.freq
OFFSETS           = args.offsets
SAMPLE_RATE       = args.rate
GAIN              = args.gain
BLOCK_SIZE        = args.block
READ_MULT         = max(1, args.read_mult)
THRESHOLD_MULT    = args.threshold
MIN_WIDTH_MS      = args.minwidth
SCAN_TIME         = args.scantime
AUTO_STEPDOWN     = args.overload_stepdown
OVERLOAD_DEBUG    = args.overload_debug
TARGET_FS_DEC_ARG = args.target_fs_dec
DECIM_ARG         = args.decim
MERGE_MS          = args.merge_ms
LOCKOUT_MS        = args.lockout_ms

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
def discover_best_offset(base_freq, sdr, block_size, sample_rate):
    OFFSET_RANGE = 10000
    OFFSET_STEP  = 1000
    best_offset  = 0
    best_score   = 0

    print(f"  Auto-discovering offset for {base_freq/1e6:.6f} MHz...")

    for off in range(-OFFSET_RANGE, OFFSET_RANGE + 1, OFFSET_STEP):
        sdr.center_freq = base_freq + off
        samples = sdr.read_samples(block_size * 2)
        score = float(np.max(np.abs(samples))) if len(samples) else 0.0
        if score > best_score:
            best_score = score
            best_offset = off

    print(f"  Best offset: {best_offset/1000:.1f} kHz → "
          f"using {(base_freq + best_offset)/1e6:.6f} MHz")
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

        nyq_in  = self.fs_in  / 2.0
        nyq_dec = self.fs_dec / 2.0

        fp         = pb_frac    * nyq_dec
        tw         = trans_frac * nyq_dec
        width_norm = max(min(tw / nyq_in, 0.999), 1e-6)

        N, beta = kaiserord(ripple_db, width_norm)
        if N % 2 == 0:
            N += 1
        self.taps           = firwin(N, cutoff=fp / nyq_in, window=('kaiser', beta))
        self._in_tail       = np.zeros(len(self.taps) - 1, dtype=np.complex64)
        self.group_delay_ms = (len(self.taps) / 2.0) / self.fs_dec * 1e3

    def process(self, x: np.ndarray) -> np.ndarray:
        x_in = np.concatenate((self._in_tail, x.astype(np.complex64, copy=False)))
        y    = upfirdn(self.taps, x_in, up=1, down=self.decim)
        trim = (len(self.taps) - 1) // self.decim
        if trim > 0 and y.size > trim:
            y = y[trim:]
        self._in_tail = x_in[-(len(self.taps) - 1):].copy()
        return y

# -----------------------------------------------------------------------
# Overload monitor — scale-independent
# -----------------------------------------------------------------------
class OverloadMonitor:
    def __init__(self,
                 clip_frac=0.98, crest_min=2.2, crest_relax=2.6,
                 p99_frac=0.85, p999_frac=0.985,
                 kurt_min=-0.5, kurt_relax=-0.2,
                 clip_ratio_min=1e-4, clip_ratio_relax=5e-6,
                 rms_hist_len=50):
        self.clip_frac        = float(clip_frac)
        self.crest_min        = float(crest_min)
        self.crest_relax      = float(crest_relax)
        self.p99_frac         = float(p99_frac)
        self.p999_frac        = float(p999_frac)
        self.kurt_min         = float(kurt_min)
        self.kurt_relax       = float(kurt_relax)
        self.clip_ratio_min   = float(clip_ratio_min)
        self.clip_ratio_relax = float(clip_ratio_relax)
        self.rms_hist         = deque(maxlen=int(rms_hist_len))
        self.peak_hist        = deque(maxlen=int(rms_hist_len))
        self.overloaded       = False

    def update(self, iq_block: np.ndarray):
        I   = iq_block.real
        Q   = iq_block.imag
        mag = np.abs(iq_block)

        rms   = float(np.sqrt(np.mean(mag * mag)) + 1e-20)
        peak  = float(np.max(mag) + 1e-20)
        crest = float(peak / rms)

        self.peak_hist.append(peak)
        full_scale = float(np.max(self.peak_hist))

        clip_thr = self.clip_frac  * full_scale
        p99_min  = self.p99_frac   * full_scale
        p999_min = self.p999_frac  * full_scale

        clip_hits  = np.logical_or(np.abs(I) >= clip_thr,
                                   np.abs(Q) >= clip_thr)
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
            conds += int((p999 > p999_min) and (p99 > p99_min))
            conds += int(kurt < self.kurt_min)
            if conds >= 2:
                self.overloaded = True
        else:
            if (clip_ratio < self.clip_ratio_relax and
                    crest > self.crest_relax and
                    p999 < (0.97 * full_scale) and
                    kurt > self.kurt_relax):
                self.overloaded = False

        return {
            "rms": rms, "peak": peak, "crest": crest,
            "full_scale": full_scale,
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
        possible = [0.0, 9.9, 14.4, 19.7, 22.9, 25.4, 28.0,
                    32.8, 37.2, 38.6, 42.1, 49.6]
        lower = [g for g in possible if g < current]
        if lower:
            new_gain = max(lower)

    if new_gain is not None and new_gain < current:
        try:
            sdr.gain = new_gain
            if verbose:
                print(f"[OVERLOAD] Reducing tuner gain: "
                      f"{current:.1f} dB → {new_gain:.1f} dB")
            return new_gain
        except Exception:
            pass
    return None

# -----------------------------------------------------------------------
# PRI outlier rejection
# -----------------------------------------------------------------------
def append_pri(pri_list: list, delta_ms: float) -> bool:
    """
    Append delta_ms to pri_list if it passes the outlier gate.
    Returns True if appended, False if rejected as an outlier.
    Values more than 2x the current running median are treated as
    missed-pulse gaps and excluded from the rolling average.
    Requires at least 3 samples before the filter activates.
    """
    if len(pri_list) >= 3:
        median_pri = float(np.median(pri_list))
        if delta_ms > median_pri * 2.0:
            return False
    pri_list.append(delta_ms)
    if len(pri_list) > PRI_WINDOW:
        pri_list.pop(0)
    return True

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
READ_SAMPLES      = BLOCK_SIZE * READ_MULT
BLOCK_DURATION_MS = READ_SAMPLES / SAMPLE_RATE * 1e3

print(f"[Decimator]   Fs_in={SAMPLE_RATE/1e6:.3f} MS/s, "
      f"decim={decimator.decim} → Fs_dec={FS_DEC/1e3:.1f} kS/s, "
      f"group delay ≈ {decimator.group_delay_ms:.1f} ms")
print(f"[Block read]  {READ_SAMPLES:,} samples "
      f"({BLOCK_DURATION_MS:.1f} ms per read, --read-mult {READ_MULT})")
print(f"[Lockout]     {LOCKOUT_MS:.0f} ms  (applied only after all checks pass)")
print(f"[Min width]   {MIN_WIDTH_MS:.1f} ms")
print(f"[Merge guard] {MERGE_MS:.1f} ms within-block")
print(f"[PRI filter]  outliers > 2x running median excluded")

if FS_DEC < 8000:
    print(f"[WARN] Very low decimated rate ({FS_DEC:.0f} Hz).")
if LOCKOUT_MS > 0 and LOCKOUT_MS <= decimator.group_delay_ms * 2:
    print(f"[WARN] --lockout-ms {LOCKOUT_MS:.0f} ms may be too small "
          f"(FIR ghosts appear at ~{decimator.group_delay_ms*2:.0f} ms).")

# Envelope smoothing
ENV_SMOOTH_WIN    = max(1, int((ENV_SMOOTH_MS / 1000.0) * FS_DEC))
ENV_SMOOTH_KERNEL = np.ones(ENV_SMOOTH_WIN, dtype=float) / float(ENV_SMOOTH_WIN)

def smooth_envelope(env_dec: np.ndarray) -> np.ndarray:
    if ENV_SMOOTH_WIN <= 1:
        return env_dec
    return np.convolve(env_dec, ENV_SMOOTH_KERNEL, mode='same')

ol = OverloadMonitor()

# -----------------------------------------------------------------------
# Build final frequency list
# -----------------------------------------------------------------------
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

print("Expanded scan frequencies (MHz):",
      [round(f / 1e6, 6) for f in EXPANDED_FREQ_LIST])

# -----------------------------------------------------------------------
# CSV output
# -----------------------------------------------------------------------
current_date  = datetime.date.today()
data_filename = f"pulsecounter-data-{current_date.isoformat()}.csv"
meta_filename = f"pulsecounter-meta-{current_date.isoformat()}.csv"
start_time    = datetime.datetime.now()

with open(meta_filename, mode='w', newline='') as mf:
    mw = csv.writer(mf)
    mw.writerow(["Logging Metadata"])
    mw.writerow(["Start Time",                   start_time.isoformat(timespec='seconds')])
    mw.writerow(["Sample Rate (input) Hz",        SAMPLE_RATE])
    mw.writerow(["Decimation Factor",             decimator.decim])
    mw.writerow(["Sample Rate (decimated) Hz",    FS_DEC])
    mw.writerow(["FIR Group Delay (ms)",          f"{decimator.group_delay_ms:.1f}"])
    mw.writerow(["Frequencies",                   ";".join(str(f) for f in EXPANDED_FREQ_LIST)])
    mw.writerow(["Gain",                          "AUTO" if GAIN is None else GAIN])
    mw.writerow(["Bias-T",                        biast_status])
    mw.writerow(["Input Block Size (samples)",    BLOCK_SIZE])
    mw.writerow(["Read Multiplier",               READ_MULT])
    mw.writerow(["Read Size (samples)",           READ_SAMPLES])
    mw.writerow(["Block Duration (ms)",           f"{BLOCK_DURATION_MS:.1f}"])
    mw.writerow(["Threshold Multiplier",          THRESHOLD_MULT])
    mw.writerow(["Minimum Width (ms)",            MIN_WIDTH_MS])
    mw.writerow(["Scan Time (s)",                 SCAN_TIME])
    mw.writerow(["DC Removal",                    "mean subtraction"])
    mw.writerow(["Noise Floor Estimator",         "median"])
    mw.writerow(["Within-block Merge Guard (ms)", MERGE_MS])
    mw.writerow(["Cross-block Lockout (ms)",      LOCKOUT_MS])
    mw.writerow(["PRI Outlier Rejection",         "delta > 2x running median excluded"])
    mw.writerow(["Data File",                     data_filename])

csv_file = open(data_filename, mode='w', newline='')
writer   = csv.writer(csv_file)
writer.writerow([
    "Date", "Time (microseconds)", "Frequency (Hz)",
    "Amplitude (decimated)", "Peak (dB)", "Width (ms)",
    "SNR (dB)", "PAR (dB)", "Noise Floor (decimated)",
    "Time Since Last Peak (ms)", "Avg PRI (ms)", "Mode PRI (ms)",
    "Overloaded",
])

# -----------------------------------------------------------------------
# Per-frequency state
# -----------------------------------------------------------------------
freq_state = {
    f: {
        # Single timestamp for both lockout gate and PRI timing.
        # Updated only after a pulse passes ALL checks (lockout + width).
        "last_pulse_time":   None,
        "pri_list":          [],
        "suppressed_count":  0,   # lockout gate rejections
        "partial_count":     0,   # width filter rejections
        "pri_outlier_count": 0,   # PRI outlier exclusions
    }
    for f in EXPANDED_FREQ_LIST
}

# -----------------------------------------------------------------------
# Main scanning loop
# -----------------------------------------------------------------------
try:
    while True:
        for freq in EXPANDED_FREQ_LIST:
            sdr.center_freq = freq
            print(f"--- Scanning {freq/1e6:.6f} MHz ---")
            scan_start = time.time()

            while (time.time() - scan_start) < SCAN_TIME:

                samples    = sdr.read_samples(READ_SAMPLES)
                block_time = datetime.datetime.now()
                samples    = samples.astype(np.complex64, copy=False)

                # Overload detection
                ol_metrics = ol.update(samples)
                overloaded = ol_metrics["overloaded"]

                if OVERLOAD_DEBUG or overloaded:
                    print(
                        f"[OVERLOAD={'YES' if overloaded else 'no '}] "
                        f"crest={ol_metrics['crest']:.2f} "
                        f"clip={ol_metrics['clip_ratio']:.2e} "
                        f"p99={ol_metrics['p99']:.3f} "
                        f"p999={ol_metrics['p999']:.3f} "
                        f"full_scale={ol_metrics['full_scale']:.3f} "
                        f"kurt={ol_metrics['kurt']:.2f}"
                    )

                if overloaded and AUTO_STEPDOWN:
                    maybe_stepdown_gain(sdr, verbose=True)

                # DC removal
                samples = samples - np.mean(samples)

                # FIR anti-alias + decimation
                dec = decimator.process(samples)

                # Envelope + smoothing
                env = np.abs(dec)
                if ENV_SMOOTH_WIN > 1:
                    env = smooth_envelope(env)

                n_dec = len(env)

                # Noise floor and threshold
                noise_floor = float(np.median(env)) if env.size else 0.0
                threshold   = noise_floor * THRESHOLD_MULT

                # Peak picking
                raw_peaks, _ = find_peaks(env, height=threshold)

                # Within-block merge guard
                merge_samps  = int((MERGE_MS / 1000.0) * FS_DEC)
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

                state = freq_state[freq]

                for p in merged_peaks:
                    # Per-pulse timestamp from sample position in block
                    samples_from_end = n_dec - int(p)
                    pulse_time = block_time - datetime.timedelta(
                        seconds=samples_from_end / FS_DEC)

                    # --------------------------------------------------
                    # CHECK 1: Cross-block lockout gate.
                    # Uses last_pulse_time, which is only set after a pulse
                    # passes ALL checks — so partial fragments can never
                    # poison the lockout clock.
                    # --------------------------------------------------
                    if LOCKOUT_MS > 0 and state["last_pulse_time"] is not None:
                        since_last_ms = (
                            pulse_time - state["last_pulse_time"]
                        ).total_seconds() * 1e3
                        if since_last_ms < LOCKOUT_MS:
                            state["suppressed_count"] += 1
                            print(
                                f"  [LOCKOUT] Ghost suppressed @ "
                                f"{pulse_time.time().isoformat(timespec='milliseconds')}, "
                                f"Δt={since_last_ms:.1f} ms < {LOCKOUT_MS:.0f} ms lockout "
                                f"(total suppressed: {state['suppressed_count']})"
                            )
                            continue

                    # --------------------------------------------------
                    # CHECK 2: Pulse width filter.
                    # Rejects partial pulses at block boundaries.
                    # last_pulse_time is NOT updated here on rejection —
                    # the lockout gate clock remains at the last good pulse.
                    # --------------------------------------------------
                    amp      = float(env[p])
                    peak_db  = 20.0 * np.log10(amp) if amp > 0 else -999.0
                    width_ms = estimate_width(env, p, FS_DEC) * 1e3

                    if width_ms < MIN_WIDTH_MS:
                        state["partial_count"] += 1
                        print(
                            f"  [PARTIAL] Narrow pulse rejected @ "
                            f"{pulse_time.time().isoformat(timespec='milliseconds')}, "
                            f"width={width_ms:.2f} ms < {MIN_WIDTH_MS:.1f} ms "
                            f"(total partial: {state['partial_count']})"
                        )
                        continue

                    # --------------------------------------------------
                    # All checks passed — update state and log the pulse.
                    # last_pulse_time is set here and nowhere else.
                    # --------------------------------------------------
                    snr      = (20.0 * np.log10(amp / noise_floor)
                                if noise_floor > 0 else 0.0)
                    mean_env = float(np.mean(env)) if env.size > 0 else 0.0
                    par      = (20.0 * np.log10(amp / mean_env)
                                if mean_env > 0 else 0.0)

                    if state["last_pulse_time"] is None:
                        delta_ms = 0.0
                    else:
                        delta    = pulse_time - state["last_pulse_time"]
                        delta_ms = delta.total_seconds() * 1e3
                        accepted = append_pri(state["pri_list"], delta_ms)
                        if not accepted:
                            state["pri_outlier_count"] += 1
                            print(
                                f"  [PRI-OUTLIER] Δt={delta_ms:.1f} ms excluded "
                                f"(>2x median; total excluded: "
                                f"{state['pri_outlier_count']})"
                            )

                    # Single update point for timing state
                    state["last_pulse_time"] = pulse_time

                    avg_pri = (float(np.mean(state["pri_list"]))
                               if state["pri_list"] else 0.0)
                    try:
                        mode_pri = (statistics.mode(state["pri_list"])
                                    if state["pri_list"] else 0.0)
                    except statistics.StatisticsError:
                        mode_pri = avg_pri

                    timestamp = pulse_time.time().isoformat(timespec='microseconds')

                    writer.writerow([
                        pulse_time.date().isoformat(), timestamp,
                        f"{freq:.0f}",
                        f"{amp:.3f}", f"{peak_db:.2f}", f"{width_ms:.2f}",
                        f"{snr:.2f}", f"{par:.2f}", f"{noise_floor:.3f}",
                        f"{delta_ms:.2f}", f"{avg_pri:.2f}",
                        f"{mode_pri:.2f}",
                        "TRUE" if overloaded else "FALSE",
                    ])

                    print(
                        f"Pulse @ {timestamp}, "
                        f"freq={freq/1e6:.6f} MHz, amp={amp:.3f}, "
                        f"peak={peak_db:.2f} dB, width={width_ms:.2f} ms, "
                        f"SNR={snr:.2f} dB, PAR={par:.2f} dB, "
                        f"NF={noise_floor:.3f}, Δt={delta_ms:.2f} ms, "
                        f"Avg PRI={avg_pri:.2f}, Mode PRI={mode_pri:.2f}, "
                        f"Overloaded={'YES' if overloaded else 'no'}"
                    )

# -----------------------------------------------------------------------
# Graceful shutdown
# -----------------------------------------------------------------------
except KeyboardInterrupt:
    stop_time = datetime.datetime.now()

    print("\n--- Session summary ---")
    for f, state in freq_state.items():
        print(f"  {f/1e6:.6f} MHz: "
              f"{state['suppressed_count']} ghost(s) suppressed, "
              f"{state['partial_count']} partial pulse(s) rejected, "
              f"{state['pri_outlier_count']} PRI outlier(s) excluded")

    with open(meta_filename, mode='a', newline='') as mf:
        mw = csv.writer(mf)
        mw.writerow(["Stop Time", stop_time.isoformat(timespec='seconds')])
        for f, state in freq_state.items():
            mw.writerow([f"Ghosts suppressed ({f/1e6:.6f} MHz)",
                         state['suppressed_count']])
            mw.writerow([f"Partial pulses rejected ({f/1e6:.6f} MHz)",
                         state['partial_count']])
            mw.writerow([f"PRI outliers excluded ({f/1e6:.6f} MHz)",
                         state['pri_outlier_count']])

    print("Stopping continuous logging...")
    csv_file.close()

sdr.close()

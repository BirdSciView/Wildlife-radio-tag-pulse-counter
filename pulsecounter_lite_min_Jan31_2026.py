#!/usr/bin/env python3
import numpy as np
import time
from rtlsdr import RtlSdr

# --------------------------
# USER SETTINGS
# --------------------------
CENTER_FREQ = 163.557e6     # Hz
SAMPLE_RATE = 1_024_000     # Hz (safe RTL-SDR rate)
GAIN = 38.6                 # dB (fixed gain)

BLOCK_SIZE = 16384          # samples per read (small-ish for Pi)
# Detection tuning:
K_MAD = 8.0                 # threshold = med + K_MAD * 1.4826 * MAD
MIN_WIDTH_MS = 1.0          # require pulse above threshold for >= this
MIN_PEAK_OVER_THR_DB = 3.0  # peak must exceed threshold by this many dB
REFRACTORY_MS = 200.0       # ignore new pulses for this long after one fires

# Optional: simple DC removal (helps reduce slow drifts)
DO_DC_REMOVE = True

# --------------------------
# INIT SDR
# --------------------------
sdr = RtlSdr()
sdr.sample_rate = int(SAMPLE_RATE)
sdr.center_freq = CENTER_FREQ
sdr.gain = GAIN

print("Starting PRI logger (robust threshold + min width)...")
print(f"Fs={SAMPLE_RATE} Hz, K_MAD={K_MAD}, min_width={MIN_WIDTH_MS} ms")

# --------------------------
# SAMPLE-BASED TIMEBASE
# --------------------------
TS0 = time.time()
sample_counter = 0

# --------------------------
# DETECTOR STATE
# --------------------------
last_pulse_t = None
refractory_samp = int((REFRACTORY_MS / 1000.0) * SAMPLE_RATE)
min_width_samp = max(1, int((MIN_WIDTH_MS / 1000.0) * SAMPLE_RATE))
cooldown = 0  # samples remaining in refractory

def robust_med_mad(x: np.ndarray):
    """Return (median, sigma_est) where sigma_est ~ std for Gaussian noise."""
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-12
    sigma = 1.4826 * mad
    return med, sigma

def db20(x):
    return 20.0 * np.log10(max(x, 1e-12))

try:
    while True:
        samples = sdr.read_samples(BLOCK_SIZE).astype(np.complex64, copy=False)

        if DO_DC_REMOVE:
            samples = samples - np.mean(samples)

        env = np.abs(samples)  # amplitude envelope

        # Robust threshold from median + MAD
        med, sigma = robust_med_mad(env)
        thr = med + (K_MAD * sigma)

        # Binary mask above threshold
        above = env > thr

        # If we’re in refractory/cooldown, suppress early part
        if cooldown > 0:
            # Skip detection in this block until cooldown expires
            skip = min(cooldown, above.size)
            above[:skip] = False
            cooldown -= skip

        # Find start/end indices of above-threshold runs
        # pad to catch edges
        padded = np.concatenate(([False], above, [False]))
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        starts = changes[0::2]
        ends = changes[1::2]  # end index in original env is end-1

        for st, en in zip(starts, ends):
            run_len = en - st
            if run_len < min_width_samp:
                continue  # too narrow => likely noise

            # Peak within the run
            seg = env[st:en]
            peak = float(np.max(seg))

            # Require peak sufficiently above threshold (in dB)
            if (db20(peak) - db20(thr)) < MIN_PEAK_OVER_THR_DB:
                continue

            # Use peak index for time stamping
            peak_idx = st + int(np.argmax(seg))
            global_index = sample_counter + peak_idx
            t_peak = TS0 + (global_index / SAMPLE_RATE)

            # PRI
            if last_pulse_t is None:
                print(f"Pulse @ {t_peak:.6f} (first)")
            else:
                pri_ms = (t_peak - last_pulse_t) * 1e3
                print(f"Pulse @ {t_peak:.6f}  PRI={pri_ms:.2f} ms  peak={peak:.4f} thr={thr:.4f}")

            last_pulse_t = t_peak

            # Enter refractory so we don't double-detect within the same pulse/echoes
            cooldown = refractory_samp

        sample_counter += len(samples)

except KeyboardInterrupt:
    print("Stopping.")
finally:
    sdr.close()

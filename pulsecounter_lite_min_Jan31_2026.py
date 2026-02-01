#!/usr/bin/env python3
import numpy as np
import time
from rtlsdr import RtlSdr

############################################
# USER SETTINGS
############################################

CENTER_FREQ = 163.557e6      # Hz – set to your wildlife tag freq
SAMPLE_RATE = 1.024e6        # Hz – safe & stable on all RTL-SDRs
BLOCK_SIZE  = 4096           # samples per read (small = low-latency)
THRESH_MULT = 5.0            # threshold = THRESH_MULT × noise floor
MIN_GAP_SEC = 0.3            # ignore pulses closer than this (anti-merge)

############################################
# INIT SDR
############################################
sdr = RtlSdr()
sdr.sample_rate = int(SAMPLE_RATE)
sdr.center_freq = CENTER_FREQ
sdr.gain        = 38.6        # or your fixed gain

print("Starting pulse logger...")
print("Sample-based timing ensures stable PRI measurement.")

############################################
# SAMPLE-BASED TIMESTAMP STATE
############################################
TS0 = time.time()              # wall-clock reference
sample_counter = 0             # global sample index (at SAMPLE_RATE)

############################################
# DETECTION STATE
############################################
last_pulse_time = None
last_pulse_timestamp = None

############################################
# MAIN LOOP
############################################
try:
    while True:
        # Read next chunk of samples
        samples = sdr.read_samples(BLOCK_SIZE)
        samples = samples.astype(np.complex64, copy=False)

        # Envelope = amplitude
        env = np.abs(samples)

        # Noise floor estimate (median)
        noise = float(np.median(env))
        threshold = noise * THRESH_MULT

        # Find threshold crossings
        above = env > threshold

        # Extract rising edges only
        edges = np.where(np.logical_and(above, np.logical_not(np.roll(above, 1))))[0]

        for e in edges:
            # Convert sample index to timestamp
            global_index = sample_counter + e
            t_peak = TS0 + global_index / SAMPLE_RATE

            # Reject pulses too close (prevents double-detection)
            if last_pulse_time is not None:
                if t_peak - last_pulse_time < MIN_GAP_SEC:
                    continue

            # Compute PRI
            if last_pulse_time is None:
                pri_ms = None
                print(f"Pulse detected @ {t_peak:.6f} (first pulse)")
            else:
                pri_ms = (t_peak - last_pulse_time) * 1e3
                print(f"Pulse @ {t_peak:.6f}  PRI = {pri_ms:.2f} ms")

            last_pulse_time = t_peak

        # Advance the global sample time
        sample_counter += len(samples)

except KeyboardInterrupt:
    print("Stopping.")
    sdr.close()

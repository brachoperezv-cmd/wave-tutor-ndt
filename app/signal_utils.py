# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.signal import find_peaks


def load_signal(filepath):
    """Loads a tab-separated signal file."""
    data = np.loadtxt(filepath, skiprows=1)
    return data[:, 0], data[:, 1]  # time_us, amplitude


def find_correlation_peaks(excit_filepath, received_filepath) -> list:
    """Performs cross-correlation to find the time delay peaks of received waves.

    Returns:
        A list of sorted, unique time delays (in microseconds) corresponding
        to the different arriving wave packets (including noise artifacts).
    """
    t_ex, excit = load_signal(excit_filepath)
    _t_rec, received = load_signal(received_filepath)

    # Time step in microseconds
    dt = t_ex[1] - t_ex[0]

    # Calculate cross-correlation (full mode)
    corr = np.correlate(received, excit, mode="full")

    # Normalize correlation to peak amplitude of 1.0
    max_corr = np.max(np.abs(corr))
    if max_corr > 0:
        corr = corr / max_corr

    abs_corr = np.abs(corr)

    # Detect peaks in correlation envelope
    # We require peaks to have at least 0.15 normalized amplitude
    # and be separated by at least 1.0 microseconds (pulse cycle resolution)
    min_dist_samples = int(1.0 / dt)
    peaks, _ = find_peaks(abs_corr, height=0.15, distance=min_dist_samples)

    peak_info = []
    for p in peaks:
        # Convert index of full correlation to sample shift
        delay_samples = p - len(excit) + 1
        delay_us = delay_samples * dt

        # Keep only causal/positive delays (with a tiny buffer for t=0 noise)
        if delay_us >= -0.2:
            amplitude = abs_corr[p]
            peak_info.append((delay_us, amplitude))

    # Sort by arrival time
    peak_info.sort(key=lambda x: x[0])

    # Group close peaks belonging to the same Tukey packet cycles (within 1.2 us)
    grouped_peaks = []
    for delay, amp in peak_info:
        if not grouped_peaks:
            grouped_peaks.append((delay, amp))
        else:
            last_delay, last_amp = grouped_peaks[-1]
            if delay - last_delay < 1.2:
                # Keep the peak with the stronger correlation amplitude
                if amp > last_amp:
                    grouped_peaks[-1] = (delay, amp)
            else:
                grouped_peaks.append((delay, amp))

    # Return list of rounded delays
    return [round(p[0], 3) for p in grouped_peaks]


def find_signal_peaks(filepath, height=0.08, distance_us=0.8) -> list:
    """Detects positive amplitude peaks in the signal file and returns their times in microseconds."""
    t, amp = load_signal(filepath)
    dt = t[1] - t[0]
    min_dist_samples = int(distance_us / dt)
    peaks, _ = find_peaks(amp, height=height, distance=min_dist_samples)
    return [float(round(t[p], 3)) for p in peaks]

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

import os
import re

import numpy as np

# Database of the 9 metals and their sound speed limits (in m/s)
METALS_DATABASE = {
    "lead": {"cL_min": 1900, "cL_max": 2200, "cS_min": 650, "cS_max": 750},
    "iron": {"cL_min": 4900, "cL_max": 6000, "cS_min": 2750, "cS_max": 3300},
    "aluminum": {"cL_min": 6200, "cL_max": 6500, "cS_min": 2950, "cS_max": 3150},
    "molybdenum": {"cL_min": 6150, "cL_max": 6300, "cS_min": 3250, "cS_max": 3450},
    "magnesium": {"cL_min": 5700, "cL_max": 5850, "cS_min": 2950, "cS_max": 3150},
    "tin": {"cL_min": 2500, "cL_max": 3350, "cS_min": 1600, "cS_max": 1750},
    "niobium": {"cL_min": 4800, "cL_max": 5000, "cS_min": 2050, "cS_max": 2150},
    "nickel": {"cL_min": 5600, "cL_max": 6100, "cS_min": 2950, "cS_max": 3050},
    "zinc": {"cL_min": 4100, "cL_max": 4250, "cS_min": 2350, "cS_max": 2500},
}


def tukey_window(N, alpha=0.4):
    n = np.arange(N)
    w = np.ones(N)
    if alpha <= 0:
        return w
    if alpha >= 1:
        return np.hanning(N)
    edge = int(np.floor(alpha * (N - 1) / 2))
    if edge > 0:
        w[:edge] = 0.5 * (1 + np.cos(np.pi * ((2 * n[:edge] / (alpha * (N - 1))) - 1)))
        nf = n[-edge:]
        w[-edge:] = 0.5 * (
            1 + np.cos(np.pi * ((2 * nf / (alpha * (N - 1))) - (2 / alpha) + 1))
        )
    return w


def make_tukey_burst(frequency, n_cycles, fs, alpha=0.4, phase=0):
    duration = n_cycles / frequency
    N = round(duration * fs)
    t_local = np.arange(N) / fs
    window = tukey_window(N, alpha)
    carrier = np.sin(2 * np.pi * frequency * t_local + phase)
    return carrier * window


def place_delayed_burst(t, burst, delay_s, scale=1.0, fs=60e6):
    y = np.zeros_like(t)
    delay_samples = round(delay_s * fs)
    end_idx = min(delay_samples + len(burst), len(t))
    if delay_samples < len(t):
        y[delay_samples:end_idx] = scale * burst[: end_idx - delay_samples]
    return y


def add_beginning_tukey_like_artifact(
    signal,
    t,
    fs,
    rng,
    probability=0.90,
    frequency=600e3,
    cycles_min=2.5,
    cycles_max=4.5,
    amp_min=0.20,
    amp_max=1.65,
    decay_strength_min=1.5,
    decay_strength_max=3.5,
):
    y = signal.copy()
    if rng.random() > probability:
        return y, None

    n_cycles_artifact = rng.uniform(cycles_min, cycles_max)
    artifact_duration = n_cycles_artifact / frequency
    N_artifact = round(artifact_duration * fs)
    artifact_amp = rng.uniform(amp_min, amp_max)
    artifact_sign = rng.choice([-1, 1])
    artifact_amp = artifact_sign * artifact_amp
    phase_offset = rng.uniform(0.7 * np.pi, 1.8 * np.pi)

    t_art = np.arange(N_artifact) / fs
    decay_strength = rng.uniform(decay_strength_min, decay_strength_max)
    envelope = np.exp(-decay_strength * t_art / artifact_duration)
    taper = tukey_window(N_artifact, alpha=0.7)

    artifact = (
        artifact_amp
        * envelope
        * taper
        * np.sin(2 * np.pi * frequency * t_art + phase_offset)
    )

    y[:N_artifact] = y[:N_artifact] + artifact

    artifact_info = {
        "artifact_cycles": n_cycles_artifact,
        "artifact_duration_us": artifact_duration * 1e6,
        "artifact_amp": artifact_amp,
        "phase_offset_rad": phase_offset,
        "decay_strength": decay_strength,
    }
    return y, artifact_info


def generate_signal_data(
    material_name: str,
    thickness_mm: float = 20.0,
    frequency_khz: float = 1000.0,
    output_folder: str = "outputs",
) -> dict:
    """Generates excitation, longitudinal, and shear wave signals.

    Calculates speeds based on the provided material boundaries, adds noise and
    artifacts, and writes files into the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    rng = np.random.default_rng()

    # Normalize material name
    mat_key = material_name.lower().strip()
    if mat_key not in METALS_DATABASE:
        raise ValueError(
            f"Material '{material_name}' not supported. Choose from: "
            f"{list(METALS_DATABASE.keys())}"
        )

    # Randomly pick sound speeds within boundaries
    db = METALS_DATABASE[mat_key]
    c_L = rng.uniform(db["cL_min"], db["cL_max"])
    c_S = rng.uniform(db["cS_min"], db["cS_max"])

    # Fixed / Configurable params
    fs = 60e6  # 60 MHz sampling
    f0 = frequency_khz * 1e3
    n_cycles = 5
    alpha = 0.4

    # Scales
    longitudinal_scale = 0.7
    shear_scale = 0.4
    shear_near_L_cycles = 7
    shear_near_L_frequency = f0 * 3.0
    # Randomize the magnitude of the longitudinal leakage wave on the shear trace
    shear_near_L_scale = rng.uniform(0.08, 0.30)
    shear_near_L_delay_offset_us = 0.3

    # Noise settings - randomize baseline noise magnitudes
    add_random_noise = True
    noise_level_excitation = 0.0
    noise_level_longitudinal = rng.uniform(0.03, 0.045)
    noise_level_shear = rng.uniform(0.03, 0.045)

    # Artifact settings - 7 cycles for initial bang with strong decay
    add_beginning_artifact = True
    beginning_artifact_probability = 0.90
    beginning_artifact_cycles_min = 6.8
    beginning_artifact_cycles_max = 7.2
    # Initial bang magnitude range: keep min same, set max to 3.5
    beginning_artifact_amp_min = 1.0
    beginning_artifact_amp_max = 3.50
    beginning_artifact_decay_min = 9.0
    beginning_artifact_decay_max = 12.0

    # Derive values
    thickness_m = thickness_mm / 1000
    burst_duration = n_cycles / f0
    longitudinal_delay = thickness_m / c_L
    shear_delay = thickness_m / c_S
    shear_near_L_delay = longitudinal_delay + shear_near_L_delay_offset_us * 1e-6

    # Length of simulation trace
    t_end = shear_delay + 2 * burst_duration
    t = np.arange(0, t_end, 1 / fs)
    time_us = t * 1e6

    # Create bursts
    excitation_burst = make_tukey_burst(f0, n_cycles, fs, alpha)
    longitudinal_burst = excitation_burst.copy()
    shear_burst = excitation_burst.copy()
    shear_near_L_burst = make_tukey_burst(
        shear_near_L_frequency, shear_near_L_cycles, fs, alpha, phase=np.pi / 5
    )

    # Place wave packets
    excitation = place_delayed_burst(t, excitation_burst, 0, 1.0, fs)
    received_longitudinal = place_delayed_burst(
        t, longitudinal_burst, longitudinal_delay, longitudinal_scale, fs
    )
    early_shear_near_L = place_delayed_burst(
        t, shear_near_L_burst, shear_near_L_delay, shear_near_L_scale, fs
    )
    late_shear_arrival = place_delayed_burst(
        t, shear_burst, shear_delay, shear_scale, fs
    )
    received_shear = early_shear_near_L + late_shear_arrival

    # Add random noise
    if add_random_noise:
        excitation += noise_level_excitation * rng.normal(size=len(excitation))
        received_longitudinal += noise_level_longitudinal * rng.normal(
            size=len(received_longitudinal)
        )
        received_shear += noise_level_shear * rng.normal(size=len(received_shear))

    # Add beginning startup artifacts
    artifact_info = {}
    if add_beginning_artifact:
        # Do not add beginning artifact to excitation signal to keep it completely clean!
        artifact_info["excitation"] = None
        # Add beginning artifacts to received signals only
        received_longitudinal, artifact_info["longitudinal"] = (
            add_beginning_tukey_like_artifact(
                signal=received_longitudinal,
                t=t,
                fs=fs,
                rng=rng,
                probability=beginning_artifact_probability,
                frequency=f0 * 4.375,
                cycles_min=beginning_artifact_cycles_min,
                cycles_max=beginning_artifact_cycles_max,
                amp_min=beginning_artifact_amp_min,
                amp_max=beginning_artifact_amp_max,
                decay_strength_min=beginning_artifact_decay_min,
                decay_strength_max=beginning_artifact_decay_max,
            )
        )
        received_shear, artifact_info["shear"] = add_beginning_tukey_like_artifact(
            signal=received_shear,
            t=t,
            fs=fs,
            rng=rng,
            probability=beginning_artifact_probability,
            frequency=f0 * 4.375,
            cycles_min=beginning_artifact_cycles_min,
            cycles_max=beginning_artifact_cycles_max,
            amp_min=beginning_artifact_amp_min,
            amp_max=beginning_artifact_amp_max,
            decay_strength_min=beginning_artifact_decay_min,
            decay_strength_max=beginning_artifact_decay_max,
        )

    material_file = re.sub(r"[^a-z0-9]+", "_", mat_key).strip("_")
    excit_file = os.path.join(output_folder, f"{material_file}_excit.txt")
    longi_file = os.path.join(output_folder, f"{material_file}_longi.txt")
    shear_file = os.path.join(output_folder, f"{material_file}_shear.txt")

    np.savetxt(
        excit_file,
        np.column_stack((time_us, excitation)),
        header="Time_us\tExcitation",
        delimiter="\t",
        comments="",
    )
    np.savetxt(
        longi_file,
        np.column_stack((time_us, received_longitudinal)),
        header="Time_us\tReceived_Longitudinal",
        delimiter="\t",
        comments="",
    )
    np.savetxt(
        shear_file,
        np.column_stack((time_us, received_shear)),
        header="Time_us\tReceived_Shear",
        delimiter="\t",
        comments="",
    )

    import json

    info = {
        "material": mat_key,
        "thickness_mm": thickness_mm,
        "frequency_khz": frequency_khz,
        "actual_cL_m_s": c_L,
        "actual_cS_m_s": c_S,
        "longitudinal_delay_us": longitudinal_delay * 1e6,
        "shear_delay_us": shear_delay * 1e6,
        "excit_file": excit_file,
        "longi_file": longi_file,
        "shear_file": shear_file,
        "artifact_info": artifact_info,
    }

    meta_file = os.path.join(output_folder, f"{material_file}_meta.json")
    with open(meta_file, "w") as f:
        json.dump(info, f, indent=2)

    return info

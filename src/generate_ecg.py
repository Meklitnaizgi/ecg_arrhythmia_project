"""
generate_ecg.py
----------------
Synthesizes a realistic ECG (electrocardiogram) signal.

BIOMEDICAL ENGINEERING BACKGROUND:
A heartbeat produces a characteristic electrical waveform called the PQRST complex:
    P wave  -> atrial depolarization (upper chambers contract)
    QRS complex -> ventricular depolarization (lower chambers contract - the big spike)
    T wave  -> ventricular repolarization (heart "resets" electrically)

We model each wave as a Gaussian (bell curve) bump, positioned at the right time offset
relative to the R-peak, with realistic amplitude and width. This is a simplified version
of the McSharry dynamical ECG model used in real biomedical signal processing research.

We also inject:
    - Realistic physiological noise (baseline wander from breathing, muscle noise, 60Hz
      powerline interference)
    - Arrhythmic beats: Premature Ventricular Contractions (PVCs), which are abnormal
      early beats originating in the ventricles. They look different (wider QRS, no
      preceding P wave, different T wave) and arrive earlier than expected.
"""

import numpy as np


def gaussian_wave(t, center, amplitude, width):
    """A single PQRST sub-wave, modeled as a Gaussian bump."""
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))


def make_normal_beat(t_local):
    """
    Build one NORMAL heartbeat waveform centered at t_local = 0 (the R-peak).
    Offsets are in seconds relative to the R-peak; amplitudes in millivolts (mV),
    consistent with real ECG scaling.
    """
    p_wave = gaussian_wave(t_local, center=-0.20, amplitude=0.15, width=0.025)
    q_wave = gaussian_wave(t_local, center=-0.05, amplitude=-0.15, width=0.010)
    r_wave = gaussian_wave(t_local, center=0.00, amplitude=1.20, width=0.010)
    s_wave = gaussian_wave(t_local, center=0.05, amplitude=-0.25, width=0.010)
    t_wave = gaussian_wave(t_local, center=0.30, amplitude=0.30, width=0.040)
    return p_wave + q_wave + r_wave + s_wave + t_wave


def make_pvc_beat(t_local):
    """
    Build one ARRHYTHMIC beat: a Premature Ventricular Contraction (PVC).
    Clinically, PVCs are:
      - WIDER QRS complex (ventricles depolarize slower via an abnormal pathway)
      - NO preceding P wave (doesn't originate from the atria's normal pacemaker)
      - Often an inverted/different T wave
      - Bigger amplitude spike
    """
    q_wave = gaussian_wave(t_local, center=-0.06, amplitude=-0.10, width=0.020)
    r_wave = gaussian_wave(t_local, center=0.00, amplitude=1.60, width=0.035)  # wider + taller
    s_wave = gaussian_wave(t_local, center=0.09, amplitude=-0.45, width=0.030)  # wider
    t_wave = gaussian_wave(t_local, center=0.35, amplitude=-0.25, width=0.060)  # inverted T
    return q_wave + r_wave + s_wave + t_wave  # note: no P wave


def generate_ecg_record(duration_sec=600, fs=250, heart_rate_bpm=72,
                         pvc_probability=0.10, noise_level=0.03, seed=42):
    """
    Generate a full synthetic ECG record.

    Parameters
    ----------
    duration_sec : total length of the recording, in seconds
    fs           : sampling frequency in Hz (250 Hz is a common clinical ECG sample rate)
    heart_rate_bpm : average heart rate in beats per minute
    pvc_probability : probability that any given beat is a PVC (arrhythmic)
    noise_level  : amount of realistic noise to inject
    seed         : random seed for reproducibility

    Returns
    -------
    t          : time array (seconds)
    signal     : the synthetic ECG voltage signal (mV)
    beat_times : true R-peak locations (seconds) -- our "ground truth"
    beat_labels: 'N' (normal) or 'V' (PVC/ventricular ectopic) for each beat -- ground truth
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    signal = np.zeros(n_samples)

    avg_rr_interval = 60.0 / heart_rate_bpm  # average time between beats, in seconds

    beat_times = []
    beat_labels = []

    current_time = 0.5  # start slightly after t=0
    while current_time < duration_sec - 1.0:
        # Real hearts don't beat like a metronome -- there's natural variability
        # (heart rate variability, HRV), so we add small random jitter to RR interval.
        rr = avg_rr_interval + rng.normal(0, 0.04)

        is_pvc = rng.random() < pvc_probability
        if is_pvc:
            # PVCs characteristically occur EARLY (shortened RR interval before them)
            rr *= 0.6

        current_time += rr
        if current_time >= duration_sec - 1.0:
            break

        beat_times.append(current_time)
        beat_labels.append('V' if is_pvc else 'N')

        # Stamp this beat's waveform onto the full signal
        window = 0.6  # seconds of signal influenced by one beat
        idx_start = max(0, int((current_time - window / 2) * fs))
        idx_end = min(n_samples, int((current_time + window / 2) * fs))
        t_local = t[idx_start:idx_end] - current_time

        beat_wave = make_pvc_beat(t_local) if is_pvc else make_normal_beat(t_local)
        signal[idx_start:idx_end] += beat_wave

    # --- Realistic noise sources ---
    # 1) Baseline wander: slow drift from breathing / body movement (~0.2-0.5 Hz)
    baseline_wander = 0.08 * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))
    # 2) Muscle noise / electrode contact noise: high-frequency random noise
    muscle_noise = rng.normal(0, noise_level, n_samples)
    # 3) Powerline interference: 60 Hz electrical hum (very common real-world artifact)
    powerline_noise = 0.02 * np.sin(2 * np.pi * 60 * t)

    signal = signal + baseline_wander + muscle_noise + powerline_noise

    return t, signal, np.array(beat_times), np.array(beat_labels)


if __name__ == "__main__":
    t, signal, beat_times, beat_labels = generate_ecg_record()
    print(f"Generated {len(t)} samples ({t[-1]:.1f} sec) at {len(beat_times)} beats.")
    n_pvc = np.sum(beat_labels == 'V')
    print(f"Normal beats: {len(beat_labels) - n_pvc}, PVC (arrhythmic) beats: {n_pvc}")
    np.savez("/home/claude/ecg_project/data/synthetic_ecg.npz",
             t=t, signal=signal, beat_times=beat_times, beat_labels=beat_labels)
    print("Saved to data/synthetic_ecg.npz")

"""
filter_ecg.py
--------------
Cleans the raw ECG signal using digital filters.

BIOMEDICAL ENGINEERING / SIGNAL PROCESSING BACKGROUND:
Real ECG signals are contaminated by three main noise sources:
  1. Baseline wander (~0.15-0.5 Hz)  -- caused by patient breathing / body movement
  2. Muscle noise / EMG artifact (broadband, high frequency) -- random muscle twitches
  3. Powerline interference (60 Hz in the US, 50 Hz in Europe) -- electrical hum picked
     up by the electrodes/wires like an antenna

The actual heartbeat's useful energy lives roughly between 0.5 Hz and 40 Hz.
So we apply:
  - A BANDPASS filter (0.5-40 Hz) to simultaneously remove baseline wander (too slow)
    and high-frequency muscle noise (too fast), keeping only the ECG-relevant band.
  - A NOTCH filter at 60 Hz to specifically cancel out powerline hum.

We use a Butterworth filter design (industry standard for physiological signals -
it's designed to have a maximally flat response in the passband, so it doesn't
distort the signal's shape/amplitude, which matters because we care about the
QRS complex's exact shape).
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(signal, fs, lowcut=0.5, highcut=40.0, order=4):
    """
    Apply a Butterworth bandpass filter.

    filtfilt (as opposed to lfilter) applies the filter forward AND backward,
    which cancels out phase distortion -- critical here because we need the
    R-peak's exact TIME location to stay accurate for heart-rate calculations.
    """
    nyquist = 0.5 * fs  # Nyquist frequency: max frequency representable at this sample rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def notch_filter(signal, fs, notch_freq=60.0, quality_factor=30.0):
    """Remove powerline interference at notch_freq Hz (60 Hz in North America)."""
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist
    b, a = iirnotch(freq, quality_factor)
    return filtfilt(b, a, signal)


def clean_ecg(signal, fs):
    """Full cleaning pipeline: bandpass filter, then notch filter."""
    filtered = bandpass_filter(signal, fs)
    filtered = notch_filter(filtered, fs)
    return filtered


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = np.load("/home/claude/ecg_project/data/synthetic_ecg.npz")
    t, raw_signal = data['t'], data['signal']
    fs = 250

    cleaned_signal = clean_ecg(raw_signal, fs)
    np.savez("/home/claude/ecg_project/data/cleaned_ecg.npz", t=t, signal=cleaned_signal)

    # Compare raw vs cleaned in a short window
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    mask = (t >= 5) & (t <= 12)
    axes[0].plot(t[mask], raw_signal[mask], color='gray')
    axes[0].set_title("BEFORE filtering (raw, noisy signal)")
    axes[0].set_ylabel("mV")

    axes[1].plot(t[mask], cleaned_signal[mask], color='#2980b9')
    axes[1].set_title("AFTER bandpass (0.5-40Hz) + 60Hz notch filtering")
    axes[1].set_ylabel("mV")
    axes[1].set_xlabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig("/home/claude/ecg_project/plots/02_filtering_comparison.png", dpi=120)
    print("Filtering complete. Saved cleaned_ecg.npz and comparison plot.")

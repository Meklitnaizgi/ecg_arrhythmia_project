"""
extract_features.py
--------------------
For each detected heartbeat, compute a small set of numeric features that a
classifier can use to distinguish Normal beats from PVCs (arrhythmic beats).

BIOMEDICAL ENGINEERING BACKGROUND -- why these specific features:

1. RR INTERVAL (time since previous beat):
   PVCs are, by definition, PREMATURE - they fire early, interrupting the heart's
   normal rhythm. So the RR interval immediately before a PVC is characteristically
   SHORTER than the patient's normal RR interval. This is one of the most reliable
   single indicators cardiologists use.

2. QRS WIDTH (duration of the main spike, measured via a threshold crossing):
   Normal beats conduct through the heart's fast, specialized conduction pathway
   (the His-Purkinje system), producing a NARROW QRS (~80-100ms).
   PVCs originate in ventricular muscle tissue itself and conduct cell-to-cell,
   which is much slower -> WIDER QRS (often >120ms). This is a textbook ECG
   diagnostic criterion for ventricular arrhythmias.

3. R-PEAK AMPLITUDE:
   PVCs often (not always) have a different peak amplitude than normal beats
   because the depolarization wavefront travels a different, abnormal path
   through the heart.
"""

import numpy as np


def compute_qrs_width(signal, fs, peak_idx, search_window=0.12):
    """
    Estimate QRS width by finding where the signal crosses a fraction of the
    peak's amplitude on either side of the R-peak (a standard half-max width method).
    """
    window_samples = int(search_window * fs)
    lo = max(0, peak_idx - window_samples)
    hi = min(len(signal), peak_idx + window_samples)

    peak_val = signal[peak_idx]
    half_max = peak_val * 0.5

    # Search backward from peak for where signal drops below half_max
    left_idx = peak_idx
    for i in range(peak_idx, lo, -1):
        if signal[i] < half_max:
            left_idx = i
            break

    # Search forward from peak for where signal drops below half_max
    right_idx = peak_idx
    for i in range(peak_idx, hi):
        if signal[i] < half_max:
            right_idx = i
            break

    width_samples = right_idx - left_idx
    width_seconds = width_samples / fs
    return width_seconds


def extract_beat_features(signal, fs, peak_indices):
    """
    Build a feature matrix: one row per beat, columns = [rr_interval, qrs_width, r_amplitude]

    Note: the very first beat has no preceding beat, so we can't compute an RR
    interval for it -- we drop it from the feature set (standard practice).
    """
    features = []
    valid_peak_indices = []

    for i in range(1, len(peak_indices)):
        curr_idx = peak_indices[i]
        prev_idx = peak_indices[i - 1]

        rr_interval = (curr_idx - prev_idx) / fs  # seconds
        qrs_width = compute_qrs_width(signal, fs, curr_idx)
        r_amplitude = signal[curr_idx]

        features.append([rr_interval, qrs_width, r_amplitude])
        valid_peak_indices.append(curr_idx)

    return np.array(features), np.array(valid_peak_indices)


if __name__ == "__main__":
    cleaned = np.load("/home/claude/ecg_project/data/cleaned_ecg.npz")
    detected = np.load("/home/claude/ecg_project/data/detected_beats.npz")
    ground_truth = np.load("/home/claude/ecg_project/data/synthetic_ecg.npz")

    signal = cleaned['signal']
    t = cleaned['t']
    fs = 250
    peak_indices = detected['detected_indices']

    features, valid_peak_indices = extract_beat_features(signal, fs, peak_indices)

    # Assign ground-truth labels to each detected beat (nearest true beat within tolerance)
    true_times = ground_truth['beat_times']
    true_labels = ground_truth['beat_labels']
    detected_times = t[valid_peak_indices]

    labels = []
    for dt in detected_times:
        idx = np.argmin(np.abs(true_times - dt))
        labels.append(true_labels[idx])
    labels = np.array(labels)

    print(f"Extracted features for {len(features)} beats.")
    print(f"Feature matrix shape: {features.shape}  (columns: RR interval, QRS width, R amplitude)")
    print(f"\nLabel distribution: Normal={np.sum(labels=='N')}, PVC={np.sum(labels=='V')}")

    print("\nMean feature values by class:")
    for cls in ['N', 'V']:
        cls_features = features[labels == cls]
        print(f"  {cls}: RR={cls_features[:,0].mean():.3f}s, "
              f"QRS width={cls_features[:,1].mean()*1000:.1f}ms, "
              f"R amplitude={cls_features[:,2].mean():.3f}mV")

    np.savez("/home/claude/ecg_project/data/features.npz",
             features=features, labels=labels, peak_indices=valid_peak_indices)
    print("\nSaved features.npz")

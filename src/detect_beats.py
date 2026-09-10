"""
detect_beats.py
----------------
Detects R-peaks (individual heartbeats) in a cleaned ECG signal.

BIOMEDICAL ENGINEERING BACKGROUND -- Pan-Tompkins Algorithm:
This is a simplified version of the Pan-Tompkins algorithm (1985), the classic and
still widely-used method for real-time QRS detection in cardiac monitors, Holter
monitors, and implantable devices. It works in stages:

  1. DIFFERENTIATE: take the derivative of the signal. The QRS complex has the
     steepest slope of any part of the heartbeat, so this stage emphasizes it and
     suppresses slower waves (P wave, T wave) which have gentler slopes.
  2. SQUARE: squaring makes everything positive and further emphasizes the
     (now large) QRS-related values, since squaring disproportionately boosts
     larger numbers.
  3. MOVING WINDOW INTEGRATION: smooth the squared signal over a short window
     (~150 ms, matched to typical QRS duration) to produce a clean "energy burst"
     at each heartbeat location.
  4. PEAK DETECTION with an adaptive threshold: find local maxima in that energy
     signal that exceed a threshold, with a "refractory period" (minimum spacing)
     since two real heartbeats can't physiologically occur closer than ~200-250ms
     apart (that would be over 240 bpm).
"""

import numpy as np
from scipy.signal import find_peaks


def pan_tompkins_detect(signal, fs):
    """
    Detect R-peak sample indices using a simplified Pan-Tompkins pipeline.

    Returns
    -------
    r_peak_indices : array of sample indices where R-peaks were detected
    """
    # Stage 1: Derivative (approximate using np.diff, emphasizes steep QRS slopes)
    derivative = np.diff(signal, prepend=signal[0])

    # Stage 2: Squaring (makes all values positive, emphasizes large derivative values)
    squared = derivative ** 2

    # Stage 3: Moving window integration (window ~ 150ms, matched to QRS width)
    window_size = int(0.15 * fs)
    window = np.ones(window_size) / window_size
    integrated = np.convolve(squared, window, mode='same')

    # Stage 4: Adaptive threshold peak detection
    # Threshold set relative to signal statistics (mean + fraction of max), a common
    # simplification of Pan-Tompkins' original dual-threshold adaptive scheme.
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)

    # Refractory period: real hearts can't beat faster than ~240 bpm -> min 250ms apart
    min_distance = int(0.25 * fs)

    peak_indices, _ = find_peaks(integrated, height=threshold, distance=min_distance)

    # Refinement: the peak in the INTEGRATED signal is delayed/smeared relative to
    # the true R-peak. Snap each detected peak to the actual local max of the
    # original signal within a small window around it, for time accuracy.
    refined_peaks = []
    search_radius = int(0.05 * fs)  # 50ms search window
    for p in peak_indices:
        lo = max(0, p - search_radius)
        hi = min(len(signal), p + search_radius)
        local_max_idx = lo + np.argmax(signal[lo:hi])
        refined_peaks.append(local_max_idx)

    return np.array(sorted(set(refined_peaks)))


def evaluate_detection(detected_times, true_times, tolerance=0.075):
    """
    Compare detected beat times to ground-truth beat times.
    A detection counts as correct (True Positive) if it falls within `tolerance`
    seconds of a true beat (75ms is a standard clinical tolerance window).

    Returns a dict with TP, FP, FN counts and sensitivity/precision -- the standard
    metrics used to evaluate medical detection algorithms.
    """
    true_times = np.array(true_times)
    matched_true = np.zeros(len(true_times), dtype=bool)
    tp = 0
    fp = 0

    for dt in detected_times:
        diffs = np.abs(true_times - dt)
        idx = np.argmin(diffs) if len(diffs) > 0 else None
        if idx is not None and diffs[idx] <= tolerance and not matched_true[idx]:
            matched_true[idx] = True
            tp += 1
        else:
            fp += 1

    fn = np.sum(~matched_true)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # a.k.a. recall: % of real beats we caught
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # % of our detections that were real

    return {"TP": tp, "FP": fp, "FN": fn, "sensitivity": sensitivity, "precision": precision}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cleaned = np.load("/home/claude/ecg_project/data/cleaned_ecg.npz")
    ground_truth = np.load("/home/claude/ecg_project/data/synthetic_ecg.npz")

    t, signal = cleaned['t'], cleaned['signal']
    true_beat_times = ground_truth['beat_times']
    fs = 250

    detected_indices = pan_tompkins_detect(signal, fs)
    detected_times = t[detected_indices]

    metrics = evaluate_detection(detected_times, true_beat_times)
    print("R-peak detection performance vs. ground truth:")
    print(f"  True Positives (correctly detected beats): {metrics['TP']}")
    print(f"  False Positives (phantom detections):      {metrics['FP']}")
    print(f"  False Negatives (missed real beats):        {metrics['FN']}")
    print(f"  Sensitivity (recall):  {metrics['sensitivity']*100:.1f}%")
    print(f"  Precision:             {metrics['precision']*100:.1f}%")

    np.savez("/home/claude/ecg_project/data/detected_beats.npz",
             detected_indices=detected_indices, detected_times=detected_times)

    # Plot detected peaks on the signal
    fig, ax = plt.subplots(figsize=(14, 4))
    mask = (t >= 5) & (t <= 15)
    ax.plot(t[mask], signal[mask], color='#2980b9', linewidth=1, label='Cleaned ECG')
    det_mask = (detected_times >= 5) & (detected_times <= 15)
    ax.scatter(detected_times[det_mask], signal[detected_indices][ [i for i,d in enumerate(detected_times) if 5<=d<=15] ],
               color='red', marker='x', s=80, label='Detected R-peak', zorder=5)
    ax.set_title(f"Automated R-Peak Detection (Sensitivity: {metrics['sensitivity']*100:.1f}%, Precision: {metrics['precision']*100:.1f}%)")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude (mV)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/home/claude/ecg_project/plots/03_beat_detection.png", dpi=120)
    print("Saved beat detection plot.")

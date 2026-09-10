"""
run_pipeline.py
----------------
Master script: runs the full ECG arrhythmia detection pipeline end-to-end.

    1. Generate synthetic ECG signal (or load real data if provided)
    2. Filter/clean the signal
    3. Detect R-peaks (heartbeats)
    4. Extract clinical features per beat
    5. Classify beats as Normal or PVC (arrhythmic)
    6. Report performance metrics

Run with:  python3 src/run_pipeline.py
"""

import numpy as np
from generate_ecg import generate_ecg_record
from filter_ecg import clean_ecg
from detect_beats import pan_tompkins_detect, evaluate_detection
from extract_features import extract_beat_features
from classify_beats import train_and_evaluate


def main():
    print("=" * 60)
    print("ECG ARRHYTHMIA DETECTION PIPELINE")
    print("=" * 60)

    fs = 250

    print("\n[1/5] Generating synthetic ECG signal...")
    t, raw_signal, true_beat_times, true_beat_labels = generate_ecg_record(
        duration_sec=600, fs=fs, heart_rate_bpm=72, pvc_probability=0.10
    )
    print(f"      -> {len(true_beat_times)} beats generated "
          f"({np.sum(true_beat_labels=='N')} Normal, {np.sum(true_beat_labels=='V')} PVC)")

    print("\n[2/5] Filtering signal (bandpass 0.5-40Hz + 60Hz notch)...")
    cleaned_signal = clean_ecg(raw_signal, fs)

    print("\n[3/5] Detecting R-peaks (Pan-Tompkins algorithm)...")
    detected_indices = pan_tompkins_detect(cleaned_signal, fs)
    detected_times = t[detected_indices]
    detection_metrics = evaluate_detection(detected_times, true_beat_times)
    print(f"      -> Sensitivity: {detection_metrics['sensitivity']*100:.1f}%  "
          f"Precision: {detection_metrics['precision']*100:.1f}%")

    print("\n[4/5] Extracting per-beat features (RR interval, QRS width, R amplitude)...")
    features, valid_peak_indices = extract_beat_features(cleaned_signal, fs, detected_indices)
    valid_times = t[valid_peak_indices]
    labels = []
    for vt in valid_times:
        idx = np.argmin(np.abs(true_beat_times - vt))
        labels.append(true_beat_labels[idx])
    labels = np.array(labels)
    print(f"      -> {len(features)} beats with extracted features")

    print("\n[5/5] Training classifier (Random Forest) and evaluating on held-out test set...")
    results = train_and_evaluate(features, labels)
    print(f"      -> Test accuracy: {results['accuracy']*100:.1f}%")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Recording length:         {t[-1]:.0f} seconds ({t[-1]/60:.1f} minutes)")
    print(f"Total beats:              {len(true_beat_times)}")
    print(f"R-peak detection sensitivity: {detection_metrics['sensitivity']*100:.1f}%")
    print(f"R-peak detection precision:   {detection_metrics['precision']*100:.1f}%")
    print(f"Classifier test accuracy:     {results['accuracy']*100:.1f}%")
    print("\nClassification report:")
    print(results['report'])


if __name__ == "__main__":
    main()

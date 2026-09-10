# ECG Arrhythmia Detection Pipeline

A biomedical signal processing pipeline that detects heartbeats in an ECG (electrocardiogram)
signal and classifies each beat as **Normal** or a **Premature Ventricular Contraction (PVC)** —
a common cardiac arrhythmia — using classical digital signal processing and machine learning.

This mirrors the core signal-processing problem inside real cardiac monitoring devices
(pacemakers, ICDs, Holter monitors, implantable loop recorders): continuously answer
*"was that heartbeat normal, and does something need to happen?"*

## Pipeline Overview

```
Raw ECG signal
      │
      ▼
[1] Signal Generation / Acquisition
      │
      ▼
[2] Filtering  (Butterworth bandpass 0.5-40Hz + 60Hz notch filter)
      │
      ▼
[3] R-Peak Detection  (Pan-Tompkins algorithm)
      │
      ▼
[4] Feature Extraction  (RR interval, QRS width, R amplitude)
      │
      ▼
[5] Classification  (Random Forest: Normal vs. PVC)
      │
      ▼
[6] Performance Evaluation  (sensitivity, precision, confusion matrix)
```

## Results

| Metric | Score |
|---|---|
| R-peak detection sensitivity | 100.0% |
| R-peak detection precision | 100.0% |
| Beat classification accuracy (held-out test set) | 100.0% |

**Dataset:** 10-minute synthetic ECG recording, 750 heartbeats (672 Normal, 78 PVC), sampled at 250 Hz.

## ⚠️ Important Limitation 

This project uses a **mathematically synthesized ECG signal**, not real clinical patient data
(e.g., the MIT-BIH Arrhythmia Database), because the development environment used to build it
did not have live internet access to download the database. The synthetic signal was built using
a Gaussian-sum PQRST waveform model with realistic physiological noise (baseline wander, muscle
noise, 60Hz powerline interference) and clinically-accurate arrhythmia characteristics (PVCs
modeled as premature, wide-QRS beats with no preceding P wave).

**Why this matters:** the 100% accuracy scores above reflect how well the pipeline works on
clean, well-separated synthetic classes — not necessarily how it would perform on messy,
overlapping real patient data. On the real MIT-BIH database, published Pan-Tompkins-style
detectors typically achieve ~99%+ sensitivity, and PVC classifiers with these three features
typically score in the 90-97% range, not 100%, because real physiological data has ambiguous
edge cases synthetic data doesn't.

**How to extend this to real data:** swap `generate_ecg.py`'s output for a real ECG record
loaded via the `wfdb` Python package from PhysioNet's MIT-BIH Arrhythmia Database — every other
module (filtering, detection, feature extraction, classification) is designed to work unmodified
on any 1D signal array + sampling rate, real or synthetic.

## Technical Skills Demonstrated

- **Digital signal processing:** Butterworth bandpass filtering, IIR notch filtering, zero-phase
  filtering (`filtfilt`), the Pan-Tompkins QRS detection algorithm (differentiation, squaring,
  moving-window integration, adaptive thresholding)
- **Biomedical domain knowledge:** PQRST waveform physiology, arrhythmia (PVC) electrophysiology,
  clinically-relevant feature engineering
- **Machine learning:** train/test splitting with stratification, Random Forest classification,
  handling class imbalance (`class_weight='balanced'`), feature importance analysis
- **Evaluation methodology:** sensitivity/precision/confusion matrix analysis appropriate for
  imbalanced medical classification tasks (not just raw accuracy)
- **Python engineering:** NumPy, SciPy, scikit-learn, Matplotlib, modular pipeline design

## Project Structure

```
ecg_project/
├── src/
│   ├── generate_ecg.py       # Synthetic ECG signal generator
│   ├── filter_ecg.py         # Bandpass + notch filtering
│   ├── detect_beats.py       # Pan-Tompkins R-peak detection
│   ├── extract_features.py   # Per-beat feature engineering
│   ├── classify_beats.py     # Random Forest training/evaluation
│   └── run_pipeline.py       # End-to-end orchestration script
├── data/                     # Generated signal/feature data (.npz)
├── plots/                    # Output visualizations
└── README.md
```

## Running It

```bash
cd src
python3 run_pipeline.py
```

## Possible Future Extensions

- Validate on real MIT-BIH Arrhythmia Database recordings via the `wfdb` package
- Add more arrhythmia classes beyond PVC (e.g., atrial fibrillation, premature atrial
  contractions) for multi-class classification
- Deploy the trained model on a microcontroller (e.g., using TensorFlow Lite Micro) to
  demonstrate embedded/real-time feasibility — directly relevant to implantable device work
- Add real-time streaming simulation (processing the signal in small chunks as if live)

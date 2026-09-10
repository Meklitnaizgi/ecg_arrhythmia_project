"""
classify_beats.py
------------------
Trains a machine learning classifier to distinguish Normal beats from PVCs
(arrhythmic beats) using the extracted features (RR interval, QRS width, R amplitude).

ML / BIOMEDICAL ENGINEERING BACKGROUND:

- We use a RANDOM FOREST classifier: an ensemble of many decision trees, each
  trained on a random subset of data/features, that vote on the final answer.
  It's a strong default choice for tabular, small-feature-count clinical data
  like this because it's robust to noisy features and doesn't need feature scaling.

- TRAIN/TEST SPLIT: We hold out 30% of the beats as a TEST set the model never
  sees during training. This is essential -- testing on the same data you trained
  on gives a falsely optimistic accuracy (the model can just "memorize" answers).
  This mirrors how real medical-device algorithms are validated: on unseen patient data.

- CLASS IMBALANCE: Only ~10% of our beats are PVCs. This is realistic (arrhythmias
  ARE rarer than normal beats in real patients) but means plain "accuracy" can be
  misleading -- a lazy model that always predicts "Normal" would still score ~90%
  accuracy while being clinically useless. That's why we report SENSITIVITY
  (did we catch the actual arrhythmias?) and a CONFUSION MATRIX, not just accuracy.
  In a medical context, missing a real arrhythmia (a False Negative) is usually far
  more dangerous than a false alarm (False Positive), so sensitivity for the PVC
  class is the metric that matters most clinically.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def train_and_evaluate(features, labels, test_size=0.3, seed=42):
    """
    Train a Random Forest on the training split and evaluate on the held-out test split.
    """
    # Convert string labels ('N'/'V') to binary (0/1) for sklearn
    y = (labels == 'V').astype(int)  # 1 = PVC (positive class), 0 = Normal

    X_train, X_test, y_train, y_test = train_test_split(
        features, y, test_size=test_size, random_state=seed, stratify=y
        # stratify=y ensures both train and test sets keep the same ~10% PVC ratio
    )

    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed,
                                  class_weight='balanced')
    # class_weight='balanced' tells the model to pay MORE attention to the minority
    # (PVC) class during training, since it's rarer -- otherwise the model could
    # get lazy and under-predict PVCs.
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Normal', 'PVC'])

    # Feature importance: which feature did the model rely on most?
    importances = dict(zip(['RR_interval', 'QRS_width', 'R_amplitude'], clf.feature_importances_))

    return {
        "model": clf,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "report": report,
        "feature_importances": importances,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    data = np.load("/home/claude/ecg_project/data/features.npz")
    features, labels = data['features'], data['labels']

    results = train_and_evaluate(features, labels)

    print(f"Test set accuracy: {results['accuracy']*100:.1f}%\n")
    print("Classification report:")
    print(results['report'])
    print("Feature importances (what the model relied on most):")
    for feat, imp in sorted(results['feature_importances'].items(), key=lambda x: -x[1]):
        print(f"  {feat}: {imp*100:.1f}%")

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'PVC'], yticklabels=['Normal', 'PVC'], ax=ax)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f"Confusion Matrix (Test Accuracy: {results['accuracy']*100:.1f}%)")
    plt.tight_layout()
    plt.savefig("/home/claude/ecg_project/plots/04_confusion_matrix.png", dpi=120)
    print("\nSaved confusion matrix plot.")

    # Feature scatter plot colored by class
    fig, ax = plt.subplots(figsize=(7, 5))
    is_pvc = labels == 'V'
    ax.scatter(features[~is_pvc, 0], features[~is_pvc, 1], alpha=0.5, label='Normal', color='#2980b9')
    ax.scatter(features[is_pvc, 0], features[is_pvc, 1], alpha=0.8, label='PVC', color='#e74c3c', marker='^', s=60)
    ax.set_xlabel("RR Interval (seconds)")
    ax.set_ylabel("QRS Width (seconds)")
    ax.set_title("Beat Features: Normal vs. PVC")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/home/claude/ecg_project/plots/05_feature_scatter.png", dpi=120)
    print("Saved feature scatter plot.")

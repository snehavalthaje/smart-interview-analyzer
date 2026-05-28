"""
SVM Emotion Classifier
=======================
Trains an SVM on the preprocessed RAVDESS features.
Saves model + scaler to models/ folder.

Usage:
    python src/training/train_svm.py
"""

import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT          = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROCESSED_DIR = os.path.join(ROOT, "data/processed")
MODELS_DIR    = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED = 42

def main():
    print("=" * 55)
    print("  Smart Interview Analyzer — SVM Training")
    print("=" * 55)

    # ── Load data ──────────────────────────────────────────────
    X = np.load(os.path.join(PROCESSED_DIR, "X.npy"))
    y = np.load(os.path.join(PROCESSED_DIR, "y.npy"))
    label_names = joblib.load(os.path.join(PROCESSED_DIR, "label_names.pkl"))

    print(f"  X shape      : {X.shape}")
    print(f"  y shape      : {y.shape}")
    print(f"  Classes      : {label_names}")

    # ── Split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED)

    print(f"  Train        : {len(X_train)}")
    print(f"  Test         : {len(X_test)}")

    # ── Scale ──────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Train SVM ──────────────────────────────────────────────
    print("\n  🚀 Training SVM with GridSearch...")
    print("     (takes 2-5 minutes)")

    param_grid = {
        "C":     [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01],
        "kernel":["rbf"],
    }

    grid = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_SEED),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_sc, y_train)

    best_svm = grid.best_estimator_
    print(f"\n  Best Params  : {grid.best_params_}")

    # ── Evaluate ───────────────────────────────────────────────
    y_pred    = best_svm.predict(X_test_sc)
    test_acc  = accuracy_score(y_test, y_pred)

    print(f"\n  ✅ Test Accuracy : {test_acc * 100:.2f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=label_names))

    # ── Confusion matrix ───────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f"SVM Confusion Matrix  (Accuracy: {test_acc*100:.2f}%)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(MODELS_DIR, "svm_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\n  Confusion matrix saved → {cm_path}")

    # ── Save model + scaler ────────────────────────────────────
    joblib.dump(best_svm, os.path.join(MODELS_DIR, "svm_model.pkl"))
    joblib.dump(scaler,   os.path.join(MODELS_DIR, "svm_scaler.pkl"))
    joblib.dump(label_names, os.path.join(MODELS_DIR, "label_names.pkl"))

    print("\n  Saved:")
    print("    models/svm_model.pkl")
    print("    models/svm_scaler.pkl")
    print("    models/label_names.pkl")
    print("    models/svm_confusion_matrix.png")
    print("\n  🎉 SVM READY!")


if __name__ == "__main__":
    main()

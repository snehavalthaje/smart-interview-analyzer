import os
import numpy as np
import joblib
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ─── Paths ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PROCESSED_DIR = os.path.join(ROOT, "data/processed")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Hyperparameters ──────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0005
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42


# ──────────────────────────────────────────────────────
# 🔥 IMPROVED MODEL
# ──────────────────────────────────────────────────────
def build_model(input_dim, num_classes):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(512, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(256, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation="relu",
                     kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax")
    ])
    return model


# ──────────────────────────────────────────────────────
# 📊 Plot
# ──────────────────────────────────────────────────────
def plot_history(history, path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    print(f"📊 Saved graph → {path}")


# ──────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 50)
    print("🔥 TRAINING MODEL (IMPROVED)")
    print("=" * 50)

    X_path = os.path.join(PROCESSED_DIR, "X.npy")
    y_path = os.path.join(PROCESSED_DIR, "y.npy")

    if not os.path.exists(X_path):
        print("❌ Run feature_extractor first")
        return

    # Load
    X = np.load(X_path)
    y = np.load(y_path)

    print(f"Original X shape: {X.shape}")

    # Labels
    label_path = os.path.join(PROCESSED_DIR, "label_names.pkl")
    if os.path.exists(label_path):
        label_names = joblib.load(label_path)
    else:
        label_names = ["neutral", "happy", "nervous", "stressed", "confident"]
        joblib.dump(label_names, label_path)

    num_classes = len(label_names)

    print(f"Classes: {label_names}")

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        stratify=y_train,
        random_state=RANDOM_SEED,
    )

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Class weights
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(cw))

    # Build model
    model = build_model(X.shape[1], num_classes)

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Callbacks (STRONGER)
    cb = [
        callbacks.EarlyStopping(patience=8, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6),
        callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "emotion_model.h5"),
            save_best_only=True,
            verbose=1
        ),
    ]

    print("\n🚀 Training started...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=cb,
        verbose=1
    )

    # Evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n✅ FINAL ACCURACY: {acc * 100:.2f}%")

    # Save labels
    joblib.dump(label_names, os.path.join(MODELS_DIR, "label_names.pkl"))

    # Plot
    plot_history(history, os.path.join(MODELS_DIR, "training.png"))

    print("\n🎉 MODEL READY → models/emotion_model.h5")


if __name__ == "__main__":
    main()
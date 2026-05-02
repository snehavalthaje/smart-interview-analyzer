import os
import sys
import numpy as np
import librosa
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RAVDESS_DIR = os.path.join(ROOT, "data/raw/RAVDESS_WAV")
PROCESSED_DIR = os.path.join(ROOT, "data/processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",
    "03": "happy",
    "04": "stressed",
    "05": "stressed",
    "06": "nervous",
    "07": "stressed",
    "08": "confident",
}

LABEL_MAP = {
    "neutral": 0,
    "happy": 1,
    "nervous": 2,
    "stressed": 3,
    "confident": 4,
}

SAMPLE_RATE = 22050
DURATION = 3.0


# ================= AUDIO =================
def load_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        target_len = int(SAMPLE_RATE * DURATION)

        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# ================= FEATURES (IMPROVED) =================
def extract_features(y):

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Delta (important for emotion)
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    chroma_mean = np.mean(chroma, axis=1)

    # Spectral features
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE))
    spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE))
    spec_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE))

    # Energy features
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))

    # Combine all
    features = np.hstack([
        mfcc_mean,
        mfcc_std,
        delta_mean,
        chroma_mean,
        spec_centroid,
        spec_bandwidth,
        spec_contrast,
        zcr,
        rms
    ])

    return features


# ================= DATASET =================
def parse_ravdess():
    records = []

    for root, _, files in os.walk(RAVDESS_DIR):
        for fname in files:
            if not fname.endswith(".wav"):
                continue

            name = fname.replace(".wav", "")
            parts = name.split("-")

            if len(parts) < 3:
                continue

            emo_code = parts[2]
            emotion = RAVDESS_EMOTION_MAP.get(emo_code)

            if emotion is None:
                continue

            records.append({
                "path": os.path.join(root, fname),
                "emotion": emotion
            })

    return records


# ================= MAIN =================
def main():
    print("🔥 STARTING FEATURE EXTRACTION...\n")

    records = parse_ravdess()
    print(f"✅ Files found: {len(records)}")

    if len(records) == 0:
        print("❌ No files detected → CHECK PATH")
        sys.exit()

    X, y = [], []

    for r in tqdm(records):
        audio = load_audio(r["path"])
        if audio is None:
            continue

        features = extract_features(audio)

        X.append(features)
        y.append(LABEL_MAP[r["emotion"]])

    X = np.array(X)
    y = np.array(y)

    print(f"\n📊 Feature shape: {X.shape}")

    np.save(os.path.join(PROCESSED_DIR, "X.npy"), X)
    np.save(os.path.join(PROCESSED_DIR, "y.npy"), y)

    print("\n🎉 DONE!")
    print("Saved in data/processed/")


if __name__ == "__main__":
    main()
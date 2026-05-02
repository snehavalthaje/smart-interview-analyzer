import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DATA_DIR = os.path.join(ROOT, "data/raw/RAVDESS_WAV")
OUTPUT_DIR = os.path.join(ROOT, "data/spectrograms")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 22050
DURATION = 3.0

def save_spectrogram(file_path, save_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        # Fix length
        target_len = int(SAMPLE_RATE * DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # Create mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Plot and save
        plt.figure(figsize=(3, 3))
        librosa.display.specshow(mel_db, sr=sr, cmap="magma")
        plt.axis("off")

        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"Error: {file_path} → {e}")


def main():
    print("🔥 Generating spectrograms...\n")

    count = 0

    for root, _, files in os.walk(DATA_DIR):
        for file in tqdm(files):
            if not file.endswith(".wav"):
                continue

            file_path = os.path.join(root, file)

            # label extraction
            name = file.replace(".wav", "")
            parts = name.split("-")

            if len(parts) < 3:
                continue

            emotion = parts[2]

            label_map = {
                "01": "neutral",
                "02": "neutral",
                "03": "happy",
                "04": "stressed",
                "05": "stressed",
                "06": "nervous",
                "07": "stressed",
                "08": "confident",
            }

            label = label_map.get(emotion)
            if label is None:
                continue

            save_dir = os.path.join(OUTPUT_DIR, label)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{count}.png")

            save_spectrogram(file_path, save_path)
            count += 1

    print(f"\n✅ DONE → {count} spectrograms created")


if __name__ == "__main__":
    main()
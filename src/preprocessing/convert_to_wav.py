import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

INPUT_DIR = os.path.join(ROOT, "data/raw/RAVDESS")
OUTPUT_DIR = os.path.join(ROOT, "data/raw/RAVDESS_WAV")

os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0

for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(".mp4"):
            in_path = os.path.join(root, file)

            actor_folder = os.path.basename(root)
            out_folder = os.path.join(OUTPUT_DIR, actor_folder)
            os.makedirs(out_folder, exist_ok=True)

            out_file = file.replace(".mp4", ".wav")
            out_path = os.path.join(out_folder, out_file)

            print(f"🎧 Converting: {file}")

            cmd = f'ffmpeg -i "{in_path}" -vn "{out_path}" -loglevel quiet'
            os.system(cmd)

            count += 1

print(f"\n✅ Converted {count} files")
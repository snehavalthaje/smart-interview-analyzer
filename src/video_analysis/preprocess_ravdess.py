import cv2
import os

RAVDESS_PATH = r"D:\SMART_INTERVIEW_ANALYZER\data\ravdess"
OUTPUT_PATH = r"D:\SMART_INTERVIEW_ANALYZER\processed_faces"

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

os.makedirs(OUTPUT_PATH, exist_ok=True)

for emotion in EMOTION_MAP.values():
    os.makedirs(os.path.join(OUTPUT_PATH, emotion), exist_ok=True)


def get_emotion_from_filename(filename):
    parts = filename.split("-")
    emotion_code = parts[2]
    return EMOTION_MAP.get(emotion_code)


def process_video(video_path, emotion, video_name):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        # take 1 frame after every 30 frames
        if frame_count % 30 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            continue

        # take largest face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))

        save_name = f"{video_name}_{saved_count}.jpg"
        save_path = os.path.join(OUTPUT_PATH, emotion, save_name)

        cv2.imwrite(save_path, face)
        saved_count += 1

    cap.release()
    return saved_count


total_saved = 0

for root, dirs, files in os.walk(RAVDESS_PATH):
    for file in files:
        if file.endswith(".mp4"):
            emotion = get_emotion_from_filename(file)

            if emotion is None:
                continue

            video_path = os.path.join(root, file)
            video_name = os.path.splitext(file)[0]

            saved = process_video(video_path, emotion, video_name)
            total_saved += saved

            print(f"{file} → {emotion} → {saved} faces saved")

print("\nDone!")
print("Total face images saved:", total_saved)
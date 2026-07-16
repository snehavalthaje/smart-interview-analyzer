import cv2
import os
import sys
import numpy as np
from tensorflow.keras.models import load_model

# ==========================================================
# Project Paths
# ==========================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    ROOT,
    "models",
    "video_emotion_model.keras"
)

# ==========================================================
# Check if model exists
# ==========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Video emotion model not found at: {MODEL_PATH}"
    )

# Load model only once
model = load_model(MODEL_PATH)

# ==========================================================
# Constants
# ==========================================================

CLASS_NAMES = {
    0: "angry",
    1: "calm",
    2: "disgust",
    3: "fearful",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprised"
}

POSITIVE_EMOTIONS = [
    "happy",
    "neutral",
    "surprised",
    "calm"
]

IMG_SIZE = 224
FRAME_SKIP = 30


# ==========================================================
# Video Prediction Function
# ==========================================================

def predict_video_emotion(video_path):

    # Load Haar Cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        return {
            "error": "Could not load Haar Cascade face detector."
        }

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "error": "Could not open video file."
        }

    total_frames = 0
    analyzed_frames = 0
    face_detected_frames = 0

    emotion_counts = {
        "angry": 0,
        "calm": 0,
        "disgust": 0,
        "fearful": 0,
        "happy": 0,
        "neutral": 0,
        "sad": 0,
        "surprised": 0
    }

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            total_frames += 1

            # Analyze every 30th frame
            if total_frames % FRAME_SKIP != 0:
                continue

            analyzed_frames += 1

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            if len(faces) == 0:
                continue

            face_detected_frames += 1

            # Select largest face
            x, y, w, h = max(
                faces,
                key=lambda rect: rect[2] * rect[3]
            )

            face = frame[y:y+h, x:x+w]

            # Convert BGR to RGB
            face = cv2.cvtColor(
                face,
                cv2.COLOR_BGR2RGB
            )

            face = cv2.resize(
                face,
                (IMG_SIZE, IMG_SIZE)
            )

            face = face.astype("float32") / 255.0

            face = np.expand_dims(
                face,
                axis=0
            )

            prediction = model.predict(
                face,
                verbose=0
            )

            predicted_index = np.argmax(prediction)

            predicted_emotion = CLASS_NAMES[predicted_index]

            emotion_counts[predicted_emotion] += 1

    finally:
        cap.release()

    # ======================================================
    # Error Handling
    # ======================================================

    if analyzed_frames == 0:
        return {
            "error": "No frames were analyzed."
        }

    if face_detected_frames == 0:
        return {
            "error": "No face detected in the video."
        }

    # ======================================================
    # Score Calculation
    # ======================================================

    dominant_emotion = max(
        emotion_counts,
        key=emotion_counts.get
    )

    positive_count = sum(
        emotion_counts[e]
        for e in POSITIVE_EMOTIONS
    )

    visibility_score = (
        face_detected_frames /
        analyzed_frames
    ) * 100

    engagement_score = (
        positive_count /
        face_detected_frames
    ) * 100

    consistency_score = (
        emotion_counts[dominant_emotion] /
        face_detected_frames
    ) * 100

    video_score = (
        0.45 * engagement_score +
        0.30 * consistency_score +
        0.25 * visibility_score
    )

    return {

        "total_frames": total_frames,

        "analyzed_frames": analyzed_frames,

        "face_detected_frames": face_detected_frames,

        "dominant_emotion": dominant_emotion,

        "emotion_counts": emotion_counts,

        "engagement_score": round(
            engagement_score,
            2
        ),

        "consistency_score": round(
            consistency_score,
            2
        ),

        "visibility_score": round(
            visibility_score,
            2
        ),

        "video_score": round(
            video_score,
            2
        )

    }


# ==========================================================
# Testing from Terminal
# ==========================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Usage:")

        print(
            r'python predict_video_emotion.py "path_to_video.mp4"'
        )

    else:

        video_path = sys.argv[1]

        result = predict_video_emotion(video_path)

        print("\n========== Video Analysis ==========\n")

        for key, value in result.items():
            print(f"{key}: {value}")
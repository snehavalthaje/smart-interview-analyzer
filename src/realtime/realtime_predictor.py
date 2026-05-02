import os
import numpy as np
import sounddevice as sd
import librosa
import joblib
import tensorflow as tf

# ✅ THESE MUST BE AT THE TOP (OUTSIDE THE CLASS)
SAMPLE_RATE = 22050
DURATION = 3.0
WINDOW_SEC = 3.0 

def record_audio(duration=WINDOW_SEC):
    """Standalone function for app.py to find"""
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    return recording.flatten()

def extract_features(y):
    # Your 137-feature logic
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta = librosa.feature.delta(mfcc)
    delta_mean = np.mean(delta, axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    chroma_mean = np.mean(chroma, axis=1)
    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE)
    contrast_mean = np.mean(contrast, axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    
    return np.hstack([mfcc_mean, mfcc_std, delta_mean, chroma_mean, contrast_mean, zcr, rms]).astype(np.float32)

class EmotionPredictor:
    def __init__(self):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.model = tf.keras.models.load_model(os.path.join(root, "models", "emotion_model.h5"))
        self.scaler = joblib.load(os.path.join(root, "models", "scaler.pkl"))
        self.labels = joblib.load(os.path.join(root, "models", "label_names.pkl"))

    def predict_from_array(self, audio_data):
        target_len = int(SAMPLE_RATE * DURATION)
        audio_data = np.pad(audio_data, (0, max(0, target_len - len(audio_data))))[:target_len]
        features = extract_features(audio_data)
        
        # Force 137 features
        expected = self.scaler.n_features_in_
        if len(features) > expected: features = features[:expected]
        elif len(features) < expected: features = np.pad(features, (0, expected - len(features)))
            
        scaled = self.scaler.transform(features.reshape(1, -1))
        preds = self.model.predict(scaled, verbose=0)[0]
        
        emotions = {0: "neutral", 1: "happy", 2: "nervous", 3: "stressed", 4: "confident"}
        icons = {"neutral": "😐", "happy": "😊", "nervous": "😰", "stressed": "😤", "confident": "💪"}
        scores = {"neutral": 70, "happy": 85, "nervous": 45, "stressed": 30, "confident": 95}
        
        label = emotions[np.argmax(preds)]
        return {"emotion": label, "confidence": float(np.max(preds)), "score": scores[label], "icon": icons[label]}
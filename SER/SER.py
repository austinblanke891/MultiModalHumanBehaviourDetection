import os
import json
import tempfile
import torch
import librosa
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

# === Paths ===
BASE = r"C:\Users\AustinBlanke\OneDrive - Blanke Advisors\Desktop\Final Model\Recordings"
OUTPUT_DIR = os.path.join(BASE, "SER_Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = r"C:\Users\AustinBlanke\OneDrive - Blanke Advisors\Desktop\Final Model\SER\wav2vec2-large-superb-er"

# === Load emotion model offline ===
extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH, local_files_only=True)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()

id2label = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

# === Feature extraction ===
def extract_features(audio_seg, sr):
    volume = float(np.sqrt(np.mean(audio_seg ** 2)))
    try:
        f0 = librosa.yin(audio_seg, fmin=50, fmax=500, sr=sr)
        pitch = float(np.nanmean(f0))
    except:
        pitch = None
    centroid = librosa.feature.spectral_centroid(y=audio_seg, sr=sr)
    tone = float(np.mean(centroid))
    return volume, pitch, tone

# === Facial detection for left/right ===
def detect_participants(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return ["Participant A", "Participant B"]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) < 2:
        return ["Participant A"]

    faces_sorted = sorted(faces, key=lambda x: x[0])  # sort by x position
    return ["Participant A", "Participant B"]

# === Process all MP4s ===
files = [f for f in os.listdir(BASE) if f.lower().endswith(".mp4")]
if not files:
    print("No MP4 files found.")
    exit()

for file in files:
    video_path = os.path.join(BASE, file)
    print(f"\n📌 Processing: {video_path}")

    participants = detect_participants(video_path)

    # === Extract audio ===
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_audio_path = tmp.name

    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', verbose=False, logger=None)

    # === Load audio ===
    y, sr = librosa.load(temp_audio_path, sr=16000, mono=True)

    # === Split into segments (2s chunks) ===
    duration = librosa.get_duration(y=y, sr=sr)
    segment_length = 2.0
    segments = [(i, min(i+segment_length, duration)) for i in np.arange(0, duration, segment_length)]

    results = []
    for idx, (start, end) in enumerate(segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_seg = y[start_sample:end_sample]
        if len(audio_seg) == 0:
            continue

        volume, pitch, tone = extract_features(audio_seg, sr)

        inputs = extractor(audio_seg, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            pred_class = torch.argmax(logits, dim=-1).item()
            emotion_label = id2label[pred_class]

        speaker = participants[idx % len(participants)]

        results.append({
            "participant": speaker,
            "start": round(start, 2),
            "end": round(end, 2),
            "emotion": emotion_label,
            "volume": volume,
            "pitch": pitch,
            "tone": tone
        })

    # === Save JSON per video ===
    out_json = os.path.join(OUTPUT_DIR, f"{os.path.splitext(file)[0]}_SER.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved SER JSON: {out_json}")

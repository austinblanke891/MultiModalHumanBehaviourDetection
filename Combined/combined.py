import os
import json
import time
import torch
import numpy as np
from datetime import timedelta
from pathlib import Path
from moviepy.editor import VideoFileClip
from transformers import AutoTokenizer, AutoModelForCausalLM, Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import librosa
import cv2
import torchvision.transforms as T
from collections import Counter
import re

# --- PATHS ---
BASE = r"C:\Users\AustinBlanke\OneDrive - Blanke Advisors\Desktop\Final_Model"
RECORDINGS_DIR = r"C:\Users\AustinBlanke\OneDrive - Blanke Advisors\Desktop\final recordings"
OUTPUT_DIR = os.path.join(BASE, "Final_Analysis")
SER_MODEL_DIR = os.path.join(BASE, "SER", "wav2vec2-large-superb-er")
GAZE_MODEL_PATH = os.path.join(BASE, "GazeTracking", "trained_gaze_field.pth")
LLM_PATH = r"C:\LLM\qwen15_7b_chat"
SEGMENT_LENGTH = 2.0
FRAME_INTERVAL = 8
torch.set_num_threads(os.cpu_count())
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cpu")
print(f"All models will run on: {device.type.upper()}")

print(f"Loading Qwen1.5-7B-Chat model and tokenizer from {LLM_PATH} (this may take a minute)...")
try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LLM_PATH, local_files_only=True, trust_remote_code=True).to(device)
    model.eval()
    print("Qwen1.5-7B-Chat loaded!")
except Exception as e:
    print(f"Failed to load Qwen1.5-7B-Chat: {e}")
    exit(1)

# --- SER ---
id2label = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad", 4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}
extractor = AutoFeatureExtractor.from_pretrained(SER_MODEL_DIR, local_files_only=True)
ser_model = Wav2Vec2ForSequenceClassification.from_pretrained(SER_MODEL_DIR, local_files_only=True).to(device)
ser_model.eval()
def extract_features(audio_seg, sr):
    volume = float(np.sqrt(np.mean(audio_seg ** 2)))
    try:
        f0 = librosa.yin(audio_seg, fmin=50, fmax=500, sr=sr)
        pitch = float(np.nanmean(f0))
    except Exception:
        pitch = None
    centroid = librosa.feature.spectral_centroid(y=audio_seg, sr=sr)
    tone = float(np.mean(centroid))
    return volume, pitch, tone

def run_ser(video_path):
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_audio_path = tmp.name
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', verbose=False, logger=None)
    y, sr = librosa.load(temp_audio_path, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    segments = [(i, min(i+SEGMENT_LENGTH, duration)) for i in np.arange(0, duration, SEGMENT_LENGTH)]
    results = []
    for idx, (start, end) in enumerate(segments):
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_seg = y[start_sample:end_sample]
        if len(audio_seg) == 0:
            continue
        volume, pitch, tone = extract_features(audio_seg, sr)
        inputs = extractor(audio_seg, sampling_rate=sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = ser_model(**inputs).logits
            pred_class = torch.argmax(logits, dim=-1).item()
            emotion_label = id2label[pred_class]
        results.append({
            "start": round(start, 2),
            "end": round(end, 2),
            "emotion": emotion_label,
            "volume": volume,
            "pitch": pitch,
            "tone": tone
        })
    return results

# --- Gaze Tracking Model ---
from gaze_field_net import GazeFieldNet

def load_gaze_model():
    model = GazeFieldNet().to(device)
    state_dict = torch.load(GAZE_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def preprocess(img, crop=None, image_size=128, head_size=64):
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    head_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((head_size, head_size)),
        T.ToTensor()
    ])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if crop is not None:
        head_crop = img_rgb[crop[1]:crop[3], crop[0]:crop[2]]
    else:
        head_crop = img_rgb
    img_tensor = transform(img_rgb)
    head_tensor = head_transform(head_crop)
    return img_tensor.unsqueeze(0), head_tensor.unsqueeze(0)

def get_gaze_stats(video_path, gaze_model, frame_interval=FRAME_INTERVAL):
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    table = 0
    partner = 0
    straight = 0
    total = 0

    for t in range(0, duration, frame_interval):
        frame = clip.get_frame(t)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # For simplicity, let's say left=partner, right=table, center=straight ahead
        h, w, _ = frame_bgr.shape
        regions = [
            (0, 0, w // 2, h),        # left: partner
            (w // 2, 0, w, h)         # right: table
        ]
        stats = []
        for i, (x1, y1, x2, y2) in enumerate(regions):
            img_tensor, head_tensor = preprocess(frame_bgr, (x1, y1, x2, y2))
            head_cx = (x1 + x2) / 2
            head_cy = (y1 + y2) / 2
            head_pos = torch.tensor([[head_cx / w, head_cy / h]], dtype=torch.float32)
            img_tensor = img_tensor.to(device)
            head_tensor = head_tensor.to(device)
            head_pos = head_pos.to(device)
            with torch.no_grad():
                field = gaze_model(img_tensor, head_pos, head_tensor).cpu().numpy()[0, 0]
                max_idx = np.unravel_index(np.argmax(field), field.shape)
                gx = max_idx[1]
                gx_norm = gx / field.shape[1]
                if gx_norm < 0.33:
                    stats.append("partner")
                elif gx_norm > 0.66:
                    stats.append("table")
                else:
                    stats.append("straight")
        label = Counter(stats).most_common(1)[0][0]
        if label == "table":
            table += 1
        elif label == "partner":
            partner += 1
        else:
            straight += 1
        total += 1

    table_p = int(100 * table / total) if total else 0
    partner_p = int(100 * partner / total) if total else 0
    straight_p = int(100 * straight / total) if total else 0
    gaze_str = f"{table_p}% at LEGO table, {partner_p}% at partner's face, {straight_p}% straight ahead"
    return gaze_str

def build_summary(ser_json, gaze_str):
    emotions = [s['emotion'] for s in ser_json]
    emotion = Counter(emotions).most_common(1)[0][0] if emotions else "Unknown"
    pitch = round(float(np.mean([s['pitch'] for s in ser_json if s.get('pitch') is not None])), 2)
    tone = round(float(np.mean([s['tone'] for s in ser_json if s.get('tone') is not None])), 2)
    volume = round(float(np.mean([s['volume'] for s in ser_json if s.get('volume') is not None])), 2)
    return f"Emotion: {emotion}, Avg. Pitch: {pitch}, Avg. Tone: {tone}, Avg. Volume: {volume}, Gaze: {gaze_str}. This pattern suggests the participant's focus and level of interaction."

def build_rbq_prompt(video_name, summary):
    return (
        f"[Video: {video_name}]\n"
        "Based on the participant's emotion, voice, and gaze (see summary below), rate Participant A in these four traits on a scale of 1-5. Use the full range, even for subtle differences between traits.\n"
        "1. Friendly (warm/kind)\n"
        "2. Talkative (spoke frequently)\n"
        "3. Engaged (attentive)\n"
        "4. Expressive (emotions/gestures)\n"
        "Format example (showing all 4 traits, and not all the same number):\n"
        "1. Friendly: 4\n"
        "2. Talkative: 2\n"
        "3. Engaged: 5\n"
        "4. Expressive: 3\n"
        "Summary: " + summary + "\n"
        "After the four ratings, write exactly two concise sentences explaining your choices. Be attentive to small differences. Keep the explanation brief."
    )

def parse_ratings(text):
    """Returns a dict of found trait numbers, e.g. {1:score, ...}"""
    ratings = {}
    for line in text.splitlines():
        m = re.match(r"\s*(\d)[\.\)]\s*[\w\s]+:\s*([1-5])", line, re.I)
        if m:
            ratings[int(m.group(1))] = m.group(2)
    return ratings

def run_llm_qwen(prompt, system_msg, max_tokens=160, retries=5):
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    output_text = ""
    for attempt in range(retries):
        try:
            input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            idx = output_text.lower().find(prompt.lower().strip())
            if idx >= 0:
                output_text = output_text[idx+len(prompt):].strip()
            ratings = parse_ratings(output_text)
            if all(n in ratings for n in range(1, 5)) and '.' in output_text:
                return output_text.strip()
            print(f"  [!] Attempt {attempt+1}: Missing ratings, retrying...")
        except Exception as err:
            print(f"  [!] Exception: {err}. Retrying...")

    ratings = parse_ratings(output_text)
    lines = []
    for n, trait in enumerate(["Friendly", "Talkative", "Engaged", "Expressive"], 1):
        if n in ratings:
            lines.append(f"{n}. {trait}: {ratings[n]}")
        else:
            lines.append(f"{n}. {trait}: N/A")
    return "\n".join(lines)

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

def main():
    gaze_model = load_gaze_model()
    video_files = sorted([f for f in os.listdir(RECORDINGS_DIR) if f.lower().endswith(".mp4")], key=natural_sort_key)
    n = len(video_files)
    if n == 0:
        print("No videos found!")
        return

    times = []
    for i, video_file in enumerate(video_files, 1):
        video_base = os.path.splitext(video_file)[0]
        analysis_path = os.path.join(OUTPUT_DIR, f"{video_base}_analysis.txt")
        print(f"\n--- Processing {video_base} ({i}/{n}) ---")
        t0 = time.time()
        ser_json = run_ser(os.path.join(RECORDINGS_DIR, video_file))
        gaze_str = get_gaze_stats(os.path.join(RECORDINGS_DIR, video_file), gaze_model)
        summary = build_summary(ser_json, gaze_str)
        prompt = build_rbq_prompt(video_base, summary)
        system_msg = (
            "You are a behavioral analyst assistant. Only provide the four numbered trait ratings as instructed. "
            "Use the full 1-5 range and avoid giving all traits the same score unless justified. After the ratings, write exactly two short sentences summarizing your rationale."
        )
        print("\n--- LLM Prompt Start ---\n", prompt, "--- LLM Prompt End ---\n")
        llm_output = run_llm_qwen(prompt, system_msg, max_tokens=160)
        if not llm_output.strip().endswith('.'):
            print("[Warning] LLM output may be cut off—try increasing max_tokens.")
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(llm_output)
        elapsed = time.time() - t0
        times.append(elapsed)
        avg_time = sum(times) / len(times)
        eta = (n - i) * avg_time
        print(f"Saved: {analysis_path} ({elapsed:.1f}s, ETA ~{timedelta(seconds=int(eta))} left)")

if __name__ == "__main__":
    main()

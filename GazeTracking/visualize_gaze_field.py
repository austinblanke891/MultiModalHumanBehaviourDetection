import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from gaze_field_net import GazeFieldNet

# === Paths ===
BASE = r"C:\Users\AustinBlanke\OneDrive - Blanke Advisors\Desktop\Final Model"
FRAMES_DIR = os.path.join(BASE, "frames")
OUTPUT_DIR = os.path.join(BASE, "GazeTracking")
MODEL_PATH = os.path.join(OUTPUT_DIR, "trained_gaze_field.pth")
VIS_DIR = os.path.join(OUTPUT_DIR, "visualized_frames")
os.makedirs(VIS_DIR, exist_ok=True)

device = torch.device("cpu")
print("Using device:", device)

# === Load model ===
model = GazeFieldNet().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=False)  # Ignore missing/extra keys
model.eval()

# === Preprocessing ===
image_size = 128  # Match training
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((image_size, image_size)),
    T.ToTensor()
])
head_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((64, 64)),
    T.ToTensor()
])

# === Helper: overlay heatmap ===
def overlay_gaze_field(image, field, head_cx, head_cy, alpha=0.4):
    H, W, _ = image.shape
    field_resized = cv2.resize(field, (W, H))
    heatmap = cv2.applyColorMap((field_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    cv2.circle(blended, (int(head_cx), int(head_cy)), 6, (0, 255, 0), -1)
    return blended

# === Process all frames ===
frame_files = [f for f in os.listdir(FRAMES_DIR) if f.lower().endswith((".jpg", ".png"))]
frame_files.sort()

for fname in frame_files:
    img_path = os.path.join(FRAMES_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        continue

    vis_img = img.copy()
    h, w, _ = img.shape
    half_w = w // 2

    regions = [
        (0, 0, half_w, h),    # Left (Participant A)
        (half_w, 0, w, h)     # Right (Participant B)
    ]

    for (x1, y1, x2, y2) in regions:
        head_crop = img[y1:y2, x1:x2]
        head_cx = (x1 + x2) / 2
        head_cy = (y1 + y2) / 2

        img_tensor = transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        head_tensor = head_transform(cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        head_pos = torch.tensor([[head_cx / w, head_cy / h]], dtype=torch.float32).to(device)

        with torch.no_grad():
            field = model(img_tensor, head_pos, head_tensor).cpu().numpy()[0, 0]

        vis_img = overlay_gaze_field(vis_img, field, head_cx, head_cy)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    out_path = os.path.join(VIS_DIR, fname)
    cv2.imwrite(out_path, vis_img)
    print(f"Saved visualization: {out_path}")

print("Visualization complete.")

import os
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm
import time
from gaze_field_net import GazeFieldNet

# === Paths ===
BASE = r"C:\Users\AustinBlanke\OneDrive - Blanke Advisors\Desktop\Final Model"
FRAMES_DIR = os.path.join(BASE, "frames")
ANNOTATION_FILE = os.path.join(BASE, "annotations.json")
OUTPUT_DIR = os.path.join(BASE, "GazeTracking")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Force CPU ===
device = torch.device("cpu")
torch.set_num_threads(8)
print("Training on CPU")

# === Hyperparameters ===
image_size = 128
field_size = 64
batch_size = 4
accum_steps = 8
num_epochs = 50
learning_rate = 5e-5
gamma = 12

# === Generate gaze cone field ===
def generate_gaze_cone_field(head_pos, gaze_point, H, W, gamma=15):
    B = head_pos.shape[0]
    xs = torch.linspace(0, 1, W)
    ys = torch.linspace(0, 1, H)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    grid_x = grid_x.unsqueeze(0).repeat(B, 1, 1)
    grid_y = grid_y.unsqueeze(0).repeat(B, 1, 1)

    hx = head_pos[:, 0].view(-1, 1, 1)
    hy = head_pos[:, 1].view(-1, 1, 1)
    gx = gaze_point[:, 0].view(-1, 1, 1)
    gy = gaze_point[:, 1].view(-1, 1, 1)

    v1x = gx - hx
    v1y = gy - hy
    v1_norm = torch.sqrt(v1x ** 2 + v1y ** 2) + 1e-6
    v1x /= v1_norm
    v1y /= v1_norm

    v2x = grid_x - hx
    v2y = grid_y - hy
    v2_norm = torch.sqrt(v2x ** 2 + v2y ** 2) + 1e-6
    v2x /= v2_norm
    v2y /= v2_norm

    dot = v1x * v2x + v1y * v2y
    cone = torch.clamp(dot, 0, 1) ** gamma
    return cone.unsqueeze(1)

# === Dataset ===
class GazeDataset(Dataset):
    def __init__(self, frames_dir, annotation_file, transform=None):
        self.frames_dir = frames_dir
        self.transform = transform
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)
        self.samples = []
        for img_name, ann in self.annotations.items():
            for i in [1, 2]:
                bbox = ann[f"bbox{i}"]
                gaze = ann[f"gaze{i}"]
                self.samples.append({"img": img_name, "bbox": bbox, "gaze": gaze})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.frames_dir, sample["img"])
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0  # Normalize

        h, w, _ = img.shape
        bbox = sample["bbox"]
        gaze = sample["gaze"]

        head_x1 = bbox["x1"] / w
        head_y1 = bbox["y1"] / h
        head_x2 = bbox["x2"] / w
        head_y2 = bbox["y2"] / h
        gaze_x = gaze["x"] / w
        gaze_y = gaze["y"] / h

        head_cx = (head_x1 + head_x2) / 2.0
        head_cy = (head_y1 + head_y2) / 2.0

        head_crop = img_rgb[int(bbox["y1"]):int(bbox["y2"]), int(bbox["x1"]):int(bbox["x2"])]
        if head_crop.size == 0:
            head_crop = np.zeros((32, 32, 3), dtype=np.float32)

        img_tensor = self.transform(img_rgb)
        head_tensor = T.Resize((64, 64))(T.ToTensor()(head_crop))

        return {
            "image": img_tensor,
            "head_pos": torch.tensor([head_cx, head_cy], dtype=torch.float32),
            "gaze_point": torch.tensor([gaze_x, gaze_y], dtype=torch.float32),
            "head_crop": head_tensor
        }

# === Data Loader ===
transform = T.Compose([
    T.ToPILImage(),
    T.Resize((image_size, image_size)),
    T.ToTensor()
])

dataset = GazeDataset(FRAMES_DIR, ANNOTATION_FILE, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# === Model and Optimizer ===
model = GazeFieldNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# === Training Loop with ETA ===
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        image = batch["image"].to(device)
        head_pos = batch["head_pos"].to(device)
        gaze_point = batch["gaze_point"].to(device)
        head_crop = batch["head_crop"].to(device)

        target_field = generate_gaze_cone_field(head_pos, gaze_point, H=field_size, W=field_size, gamma=gamma)

        pred_field = model(image, head_pos, head_crop)
        loss = F.mse_loss(pred_field, target_field) / accum_steps

        loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * accum_steps

    elapsed = time.time() - start_time
    eta = elapsed * (num_epochs - epoch - 1)
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss / len(loader):.4f} | ETA: ~{eta/60:.1f} min")

# === Save Model ===
model_path = os.path.join(OUTPUT_DIR, "trained_gaze_field.pth")
torch.save(model.state_dict(), model_path)
print(f"Training complete. Model saved to {model_path}")

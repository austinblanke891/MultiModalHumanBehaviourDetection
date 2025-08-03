import cv2
import os
import json
import re

# Define paths
BASE_FOLDER = r"C:\Users\AustinBlanke\OneDrive - Blanke Advisors\Desktop\Final Model"
FRAMES_FOLDER = os.path.join(BASE_FOLDER, "frames")
ANNOTATION_FILE = os.path.join(BASE_FOLDER, "annotations.json")

# Load existing annotations
annotations = {}
if os.path.exists(ANNOTATION_FILE):
    with open(ANNOTATION_FILE, "r") as f:
        annotations = json.load(f)

# Globals
drawing = False
ix, iy = -1, -1
cx, cy = -1, -1
step = 0
current_annotation = {}

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, cx, cy, step, current_annotation

    if event == cv2.EVENT_LBUTTONDOWN:
        if step in [0, 2]:  # Start drawing box
            drawing = True
            ix, iy = x, y
            cx, cy = x, y

        elif step in [1, 3]:  # Gaze click
            key = f"gaze{(step + 1) // 2}"
            current_annotation[key] = {"x": x, "y": y}
            print(f"{key} set at ({x}, {y})")
            step += 1  # Now we increment after gaze is set

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cx, cy = x, y  # Update for live preview

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        key = f"bbox{(step + 1)}"
        current_annotation[key] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        print(f"{key} set from ({x1}, {y1}) to ({x2}, {y2})")
        step += 1

def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

def draw_annotations(img, ann, live_box=False):
    """Draws saved bboxes/gaze + optional live box"""
    display = img.copy()
    # Draw saved annotations
    for i in [1, 2]:
        bbox_key = f"bbox{i}"
        gaze_key = f"gaze{i}"
        if bbox_key in ann:
            b = ann[bbox_key]
            cv2.rectangle(display, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), 2)
        if bbox_key in ann and gaze_key in ann:
            b = ann[bbox_key]
            g = ann[gaze_key]
            cx_mid = (b["x1"] + b["x2"]) // 2
            cy_mid = (b["y1"] + b["y2"]) // 2
            cv2.circle(display, (g["x"], g["y"]), 5, (0, 0, 255), -1)
            cv2.line(display, (cx_mid, cy_mid), (g["x"], g["y"]), (255, 0, 0), 2)

    # Draw live dragging box
    if live_box:
        cv2.rectangle(display, (ix, iy), (cx, cy), (255, 255, 0), 1)

    return display

def annotate_images():
    global step, current_annotation, drawing

    files = sorted(
        [f for f in os.listdir(FRAMES_FOLDER) if f.lower().endswith((".jpg", ".png"))],
        key=extract_number
    )

    cv2.namedWindow("Annotator")
    cv2.setMouseCallback("Annotator", mouse_callback)

    for fname in files:
        if fname in annotations:
            print(f"Skipping {fname}, already annotated.")
            continue

        path = os.path.join(FRAMES_FOLDER, fname)
        image = cv2.imread(path)
        current_annotation = {}
        step = 0

        while True:
            live_box_active = drawing and step in [0, 2]
            display = draw_annotations(image, current_annotation, live_box=live_box_active)

            instructions = ["Drag BBOX 1", "Click GAZE 1", "Drag BBOX 2", "Click GAZE 2"]
            cv2.putText(display, f"Step {step+1}/4: {instructions[step]}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Annotator", display)
            key = cv2.waitKey(20) & 0xFF

            if key == 27:  # ESC
                print("Exiting...")
                save_annotations()
                cv2.destroyAllWindows()
                return

            if step == 4:
                annotations[fname] = current_annotation.copy()
                save_annotations()
                print(f"Saved {fname}")
                break

    cv2.destroyAllWindows()

def save_annotations():
    with open(ANNOTATION_FILE, "w") as f:
        json.dump(annotations, f, indent=2)

if __name__ == "__main__":
    annotate_images()

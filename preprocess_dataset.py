import os
import cv2
import numpy as np
import json
import random
from glob import glob
from tqdm import tqdm

# ==============================
# === CONFIGURATION ============
# ==============================
BW_VIDEO_DIR = 'data/bw_videos/'
COLOR_REF_DIR = 'data/color_refs/'  # Optional
OUTPUT_DIR = 'processed/'
RESIZE_DIM = (256, 256)  # (width, height)
SPLIT_RATIO = [0.7, 0.15, 0.15]  # Train/Val/Test

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(42)

# ==============================
# === FRAME EXTRACTION =========
# ==============================
grayscale_frames = []
reference_frames = []
metadata = {}

bw_videos = glob(os.path.join(BW_VIDEO_DIR, '*.mp4')) + \
            glob(os.path.join(BW_VIDEO_DIR, '*.avi'))

print(f"Found {len(bw_videos)} black-and-white video(s)")

for vid_path in tqdm(bw_videos, desc="Processing B&W videos"):
    cap = cv2.VideoCapture(vid_path)
    video_name = os.path.basename(vid_path)
    frame_ids = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize and convert to grayscale
        resized = cv2.resize(frame, RESIZE_DIM)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = gray[..., np.newaxis]  # Shape: (H, W, 1)
        grayscale_frames.append(gray)

        # Use BGR resized frame as "reference" (for now)
        reference_frames.append(resized)  # Shape: (H, W, 3)

        frame_ids.append(len(grayscale_frames) - 1)

    cap.release()
    metadata[video_name] = frame_ids

print("Total frames extracted:", len(grayscale_frames))

# ==============================
# === CONVERT TO NP ARRAYS =====
# ==============================
grayscale_frames = np.array(grayscale_frames, dtype=np.uint8)
reference_frames = np.array(reference_frames, dtype=np.uint8)

# ==============================
# === SPLIT DATASET ============
# ==============================
total = len(grayscale_frames)
indices = list(range(total))
random.shuffle(indices)

train_end = int(SPLIT_RATIO[0] * total)
val_end = train_end + int(SPLIT_RATIO[1] * total)

train_idx = indices[:train_end]
val_idx = indices[train_end:val_end]
test_idx = indices[val_end:]

def save_npz(split_name, idxs):
    gs = grayscale_frames[idxs]
    ref = reference_frames[idxs]
    path = os.path.join(OUTPUT_DIR, f"{split_name}.npz")
    np.savez_compressed(path, grayscale=gs, reference=ref)
    print(f"Saved {split_name}.npz — {len(idxs)} samples")

save_npz("train", train_idx)
save_npz("val", val_idx)
save_npz("test", test_idx)

# ==============================
# === SAVE METADATA ============
# ==============================
with open(os.path.join(OUTPUT_DIR, 'metadata_index.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("Saved metadata_index.json")
print("✅ Data preprocessing complete.")

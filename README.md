project/
├── preprocess_dataset.py
├── README.md
├── data/
│   ├── raw_videos/            # Grayscale videos
│   ├── reference_images/      # Optional color references
│   └── processed/             # Output directory (created automatically)


## 🎯 Purpose

Prepare a clean and structured dataset of grayscale video frames and corresponding reference color images (optional) for training deep learning models in video colorization.

---

## 📥 Input

- ✅ Raw black-and-white video clips (`.mp4`, `.avi`, etc.)
- ✅ Optional: Reference color video clips or images (e.g., `.jpg`, `.png`)

---

## 📤 Output

- ✅ `.npz` files:
  - `train.npz`
  - `val.npz`
  - `test.npz`
- ✅ Each file contains:
  - `grayscale`: shape `(N, H, W, 1)` – grayscale frames
  - `reference`: shape `(N, H, W, 3)` – paired reference color images
- ✅ Optional: `metadata.json` file mapping filenames to frame sequences

---

## ⚙️ Features

- Extract grayscale frames from video
- Pair grayscale frames with reference color images (optional)
- Resize all frames to a fixed resolution (default: `256x256`)
- Split dataset into train, validation, and test subsets
- Save everything in compressed `.npz` format for efficient training

---

## 🛠️ Setup Instructions

### 1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate


pip install opencv-python numpy tqdm


Example Command

python preprocess_dataset.py \
  --video_dir ./data/raw_videos \
  --ref_dir ./data/reference_images \
  --output_dir ./data/processed \
  --image_size 128



references

📌 Notes
The dataset is automatically shuffled before splitting.

Ratio split is 70% train, 15% val, 15% test (you can modify in code).

Works with Python 3.8+.

If the number of reference images is smaller than grayscale frames, references will be reused or randomly assigned.




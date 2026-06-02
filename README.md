# Satellite Ship Detection & Classification

End-to-end pipeline for detecting and classifying ships in satellite imagery, combining classical computer vision with deep learning.

## What it does

**Part A — Detection & Cropping** (`satellite_ship_pipeline.ipynb`, cells A.1–A.9)

Uses OpenCV-based multi-channel segmentation to find ships in a raw satellite image without any trained model:

1. Loads a satellite image from the `images/` folder.
2. Runs auto-detection using HSV brightness thresholding + saturation channel, morphological cleanup, area/aspect-ratio filtering, and NMS.
3. A v2 pipeline adds CLAHE+Otsu, Canny edge detection, LAB-based cloud exclusion, and asymmetric morphology for better precision.
4. Exposes interactive sliders (`ipywidgets`) to tune detection parameters live.
5. Crops each detected ship and saves them to `crops/` as PNG files, resized to 80×80 px for downstream classification.

**Part B — Training & Classification** (`satellite_ship_pipeline.ipynb`, cells B.1–B.22)

Trains a binary classifier (ship / no-ship) on the [ShipNet](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) labeled dataset:

1. Downloads the dataset from Kaggle (4000 labeled 80×80 tiles).
2. Trains a **baseline CNN** from scratch (3 conv blocks + BN + Dropout).
3. Trains **MobileNetV2** via transfer learning from ImageNet in two phases: head-only first, then partial fine-tuning of the last 5 feature blocks.
4. Evaluates both models: confusion matrix, classification report, AUC-ROC, Grad-CAM heatmaps.
5. Applies the trained classifier to the crops produced by Part A.

**YOLO11 fine-tuning** (`ship_detection_local.ipynb`, `ship_detection_local_v2.ipynb`, `ship_detection_local_v3.ipynb`)

Fine-tunes `yolo11m.pt` on a custom annotated dataset for bounding-box ship detection. Unlike Part A (which uses only OpenCV), this approach learns directly from labeled examples and produces confidence scores per detection.

The pipeline in all three notebooks is identical:

1. Download the ShipNet image chips from Kaggle (80×80 px tiles, `1__*.png` = ship).
2. Upload images to Roboflow and annotate bounding boxes (or use the Auto Label tool with Grounding DINO / SAM).
3. Download the annotated dataset in YOLOv11 format.
4. Fine-tune `yolo11m.pt` locally with satellite-specific augmentations.
5. Evaluate on the held-out test split (mAP@50, precision, recall).
6. Run inference with a confidence threshold sweep to find the best operating point.

**Why three notebooks?**

They represent three successive training experiments, each tuning a different set of hyperparameters:

| Notebook | Run dir | `epochs` | `imgsz` | `cls` | `mosaic` | `degrees` | Time | mAP@50 (test) |
|---|---|---|---|---|---|---|---|---|
| `ship_detection_local.ipynb` | `ship_detection_v13` | 30 | 640 | 0.5 | 1.0 | 30° | 0.621 h | 0.463 |
| `ship_detection_local_v2.ipynb` | `ship_detection_v14` | 100 | 640 | 1.5 | 1.0 | 30° | 1.821 h | 0.616 |
| `ship_detection_local_v3.ipynb` | `ship_detection_v32` | 120 | 960 | 0.5 | 0.5 | 10° | 3.909 h | 0.611 |

All runs use `yolo11m.pt` pretrained on COCO, AdamW with cosine LR decay, `flipud=0.5`, `fliplr=0.5`, `patience=50`, and AutoBatch. Best weights for each run are saved under `runs/detect/<run_dir>/weights/best.pt`.

`_local_v3` is the current reference notebook. v1 and v2 are kept for comparison across experiments.

## Repository structure

```
satellite_ship_pipeline.ipynb   # Main notebook (Parts A + B — OpenCV + CNN/MobileNetV2)
ship_detection_local.ipynb      # YOLO11 training — v1 (30 epochs, imgsz=640)
ship_detection_local_v2.ipynb   # YOLO11 training — v2 (100 epochs, imgsz=640)
ship_detection_local_v3.ipynb   # YOLO11 training — v3 (120 epochs, imgsz=960) ← current best
conclusions.md                  # Final project report (TP Final Computer Vision)
images/                         # Input satellite images
crops/                          # Auto-cropped ship candidates (generated)
models/                         # Saved weights and ONNX exports (see table below)
runs/                           # Ultralytics training outputs (generated)
  └── detect/
      ├── ship_detection_v13/   # v1 best weights
      ├── ship_detection_v14/   # v2 best weights
      └── ship_detection_v32/   # v3 best weights ← current best
datasets/                       # ShipNet chips + Roboflow annotated dataset
pyproject.toml                  # Dependencies (managed with uv)
```

## Saved models

All files live in `models/`. The table maps each file to the notebook cell that produces it.

| File | Size | What it is | Notebook | Cell |
|---|---|---|---|---|
| `best_cnn.pth` | 6.8 MB | Best baseline CNN weights saved by early stopping | `satellite_ship_pipeline.ipynb` | **B.9** — Training loop |
| `best_mobilenet_phase1.pth` | 9.0 MB | Best MobileNetV2 weights after head-only training | `satellite_ship_pipeline.ipynb` | **B.13** — Phase 1: train head only |
| `best_mobilenet_phase2.pth` | 9.0 MB | Best MobileNetV2 weights after partial fine-tuning | `satellite_ship_pipeline.ipynb` | **B.14** — Phase 2: fine-tuning |
| `mobilenet_ship.pth` | 9.5 MB | Final MobileNetV2 `state_dict` (use with `build_mobilenet()`) | `satellite_ship_pipeline.ipynb` | **B.19** — Export model |
| `mobilenet_ship_full.pt` | 9.1 MB | Full MobileNetV2 object (`torch.save(model, ...)`) | `satellite_ship_pipeline.ipynb` | **B.19** — Export model |
| `mobilenet_ship.onnx` | 9.2 MB | MobileNetV2 exported to ONNX (opset 12, dynamic batch) | `satellite_ship_pipeline.ipynb` | **B.19** — Export model |
| `yolo11m.pt` | 38.8 MB | Ultralytics YOLO11m pretrained base model (cached to avoid re-download) | `ship_detection_local.ipynb` · `ship_detection_local_v2.ipynb` | config cell |
| `yolo11n.pt` | 5.4 MB | Ultralytics YOLO11n pretrained base (downloaded automatically by Ultralytics AMP checks) | `ship_detection_local_v2.ipynb` | auto-downloaded |

**Loading the final classifier:**

```python
import torch
from satellite_ship_pipeline import build_mobilenet   # or copy the function from cell B.12

model = build_mobilenet()
model.load_state_dict(torch.load("models/mobilenet_ship.pth", map_location="cpu"))
model.eval()
```

Or load the full object directly (no architecture code needed):

```python
model = torch.load("models/mobilenet_ship_full.pt", map_location="cpu")
model.eval()
```

## Setup

Requires Python 3.10 and [uv](https://github.com/astral-sh/uv).

```bash
# Install all dependencies (PyTorch with CUDA 12.4 included)
uv sync

# Launch Jupyter
uv run jupyter notebook
```

> **GPU:** Part B training is significantly faster on a CUDA GPU. The notebook auto-detects the device. CPU works but is slow for MobileNetV2 fine-tuning.

## Running the pipeline

### Quick start — detection only

1. Place a satellite image (PNG/JPG) in the `images/` folder.
2. Open `satellite_ship_pipeline.ipynb` and run cells **A.1 → A.9** in order.
3. Adjust sliders in **A.6** (v1) or **A.7-D** (v2) to refine parameters.
4. Run **A.7-E** to commit v2 detections, then **A.8** to save crops.

### Full pipeline — detection + classification

Continue from the crops and run cells **B.1 → B.21**.  
B.2 requires a valid `kaggle.json` credentials file in the project root.

### YOLO11 fine-tuning

Open `ship_detection_local_v3.ipynb` (current best, 120 epochs, imgsz=960) and follow the cells. Requires a Roboflow API key stored in a `.env` file:

```
ROBOFLOW_API_KEY=your_key_here
```

## Key dependencies

| Package | Purpose |
|---|---|
| `torch` + `torchvision` (CUDA 12.4) | CNN and MobileNetV2 training |
| `ultralytics<=8.3.40` | YOLO11 detection |
| `opencv-python` | Image processing for Part A |
| `scikit-learn` | Metrics, train/val/test split |
| `roboflow` / `inference` | Dataset download and Roboflow deployment |
| `ipywidgets` | Interactive parameter sliders |

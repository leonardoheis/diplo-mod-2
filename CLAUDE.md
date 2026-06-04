# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a YOLO11 object detection project for training a custom model on a Roboflow dataset. The primary workflow runs on **Google Colab** (GPU required); `main.py` is a local entrypoint stub.

The notebook `Copy_of_Copia_de_train_yolo11_object_detection_on_custom_dataset.ipynb` is the core artifact — it covers:
1. Pre-trained inference with `yolo11n.pt` on COCO
2. Dataset download from Roboflow Universe via API key
3. Fine-tuning `yolo11s.pt` on the custom dataset
4. Validation, inference on test images, and deployment back to Roboflow

## Environment

- Python 3.10 (pinned via `.python-version`)
- Package manager: `uv` (uses `pyproject.toml`)
- Runtime dependencies (`ultralytics<=8.3.40`, `supervision`, `roboflow`, `inference`) are installed inside Colab, not in `pyproject.toml`

## Commands

```bash
# Install a new dependency
uv add <package>

# Lint + format check (run this after every code or notebook change)
uv run poe check
```

## Quality Gate

**Always run `uv run poe check` after any change to Python files or notebooks.**
If it reports files that would be reformatted, run `uv run ruff format .` to fix them, then re-run `uv run poe check` to confirm it passes before considering the task done.

## Key Configuration

- **Roboflow workspace**: `liangdianzhong`, project `-qvdww`, version `1`
- **Dataset export format**: `yolov11`
- **Training**: `yolo11s.pt`, 10 epochs, 640px image size
- **Trained weights output**: `runs/detect/train<N>/weights/best.pt`
- **Roboflow API key**: stored as a Colab Secret named `ROBOFLOW_API_KEY`

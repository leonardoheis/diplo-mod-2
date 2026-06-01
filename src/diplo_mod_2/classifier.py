"""Part B — PyTorch dataset, models, training loop, and inference helpers."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import MobileNet_V2_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShipDataset(Dataset):
    """Wraps numpy (N, H, W, C) float32 arrays for use with DataLoader."""

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = X
        self.y = y.astype(np.float32)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        img = torch.from_numpy(self.X[idx].transpose(2, 0, 1))  # (H,W,C) → (C,H,W)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.y[idx])


class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


def build_mobilenet(trainable_base: bool = False, device: torch.device = DEVICE) -> nn.Module:
    """Build MobileNetV2 with a custom binary classification head."""
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = trainable_base
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )
    return model.to(device)


def train_model(
    model: nn.Module,
    train_ldr: DataLoader,
    val_ldr: DataLoader,
    epochs: int = 25,
    lr: float = 1e-3,
    patience: int = 5,
    model_path: str = "best_model.pth",
) -> dict[str, list]:
    """Generic training loop with early stopping and LR reduction on plateau."""
    device = next(model.parameters()).device
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-6, verbose=True
    )
    criterion = nn.BCEWithLogitsLoss()
    best_vloss = float("inf")
    patience_count = 0
    history: dict[str, list] = {
        k: [] for k in ("loss", "val_loss", "accuracy", "val_accuracy", "auc", "val_auc")
    }

    for epoch in range(epochs):
        model.train()
        t_loss: float = 0.0
        t_probs: list[float] = []
        t_labels: list[float] = []
        for X_b, y_b in train_ldr:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b).view(-1)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(y_b)
            t_probs.extend(torch.sigmoid(logits).detach().cpu().numpy())
            t_labels.extend(y_b.cpu().numpy())

        model.eval()
        v_loss: float = 0.0
        v_probs: list[float] = []
        v_labels: list[float] = []
        with torch.no_grad():
            for X_b, y_b in val_ldr:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits = model(X_b).view(-1)
                v_loss += criterion(logits, y_b).item() * len(y_b)
                v_probs.extend(torch.sigmoid(logits).cpu().numpy())
                v_labels.extend(y_b.cpu().numpy())

        t_loss /= len(train_ldr.dataset)  # type: ignore[arg-type]
        v_loss /= len(val_ldr.dataset)  # type: ignore[arg-type]
        t_acc = ((np.array(t_probs) >= 0.5) == np.array(t_labels)).mean()
        v_acc = ((np.array(v_probs) >= 0.5) == np.array(v_labels)).mean()
        t_auc = roc_auc_score(t_labels, t_probs)
        v_auc = roc_auc_score(v_labels, v_probs)

        for k, val in zip(history, (t_loss, v_loss, t_acc, v_acc, t_auc, v_auc)):
            history[k].append(val)

        scheduler.step(v_loss)

        if v_loss < best_vloss:
            best_vloss = v_loss
            patience_count = 0
            torch.save(copy.deepcopy(model.state_dict()), model_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(
            f"Ep {epoch + 1:3d}  loss={t_loss:.4f} val={v_loss:.4f}  "
            f"acc={t_acc:.4f} val_acc={v_acc:.4f}  auc={t_auc:.4f} val_auc={v_auc:.4f}"
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    return history


def predict_image(
    image_path_or_array: Any,
    model: nn.Module,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Predict whether a satellite image contains a ship.

    Args:
        image_path_or_array: path to PNG/JPG, or numpy array (80, 80, 3) float32 [0, 1]
        model: trained PyTorch model
        threshold: decision threshold (default 0.5)

    Returns:
        dict with label, probability, confidence
    """
    device = next(model.parameters()).device
    if isinstance(image_path_or_array, str):
        img = np.array(Image.open(image_path_or_array).resize((80, 80))) / 255.0
    else:
        img = image_path_or_array

    tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    model.eval()
    with torch.no_grad():
        prob = float(torch.sigmoid(model(tensor))[0])

    return {
        "label": "SHIP" if prob >= threshold else "NO SHIP",
        "probability": round(prob, 4),
        "confidence": f"{max(prob, 1 - prob) * 100:.1f}%",
    }

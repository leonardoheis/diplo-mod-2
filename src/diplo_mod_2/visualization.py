"""Part B — Visualization, evaluation, and Grad-CAM helpers."""

from __future__ import annotations

from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def plot_history(history: dict[str, list], title: str = "Learning curves") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=13)
    for ax, (metric, label) in zip(
        axes, [("loss", "Loss"), ("accuracy", "Accuracy"), ("auc", "AUC-ROC")]
    ):
        ax.plot(history[metric], label="Train", linewidth=2)
        ax.plot(history[f"val_{metric}"], label="Val", linewidth=2, linestyle="--")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()


def evaluate_model(
    model: nn.Module,
    loader: Any,
    y_true: np.ndarray,
    model_name: str = "Model",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader and print metrics + confusion matrix."""
    device = next(model.parameters()).device
    model.eval()
    all_probs: list[float] = []
    all_preds: list[int] = []
    with torch.no_grad():
        for X_b, _ in loader:
            logits = model(X_b.to(device)).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs >= 0.5).astype(int))

    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  {model_name} — Test Set Evaluation")
    print(f"{sep}")
    print(classification_report(y_true, y_pred, target_names=["No ship", "Ship"]))
    print(f"AUC-ROC: {roc_auc_score(y_true, y_prob):.4f}")

    fig, ax = plt.subplots(figsize=(4, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No ship", "Ship"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion matrix — {model_name}", fontsize=11)
    plt.tight_layout()
    plt.show()
    return y_prob, y_pred


def overlay_gradcam(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the original image."""
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap((255 * heatmap_resized).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted((255 * img).astype(np.uint8), 1 - alpha, heatmap_colored, alpha, 0)


def gradcam_pytorch(
    model: nn.Module,
    img_tensor: torch.Tensor,
    target_layer: nn.Module,
) -> np.ndarray:
    """Compute Grad-CAM for a single image tensor (1, C, H, W)."""
    activations: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}
    fh = target_layer.register_forward_hook(lambda m, i, o: activations.update({"v": o.detach()}))
    bh = target_layer.register_full_backward_hook(
        lambda m, gi, go: grads.update({"v": go[0].detach()})
    )

    model.eval()
    out = model(img_tensor)
    model.zero_grad()
    out[0].backward()
    fh.remove()
    bh.remove()

    a = activations["v"][0]  # (C, H, W)
    w = grads["v"][0].mean(dim=[1, 2])  # (C,)
    cam = (a * w[:, None, None]).sum(0).relu()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy()

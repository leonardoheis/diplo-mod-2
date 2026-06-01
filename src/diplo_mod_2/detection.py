"""Part A — OpenCV-based ship detection helpers."""

from __future__ import annotations

import cv2
import numpy as np

COLORS = [
    "#FF4444",
    "#44FF88",
    "#4488FF",
    "#FFAA00",
    "#FF44FF",
    "#00FFFF",
    "#FF8800",
    "#88FF00",
    "#FF0088",
    "#00FF88",
    "#8800FF",
    "#FFFF00",
    "#00FFFF",
    "#FF4444",
    "#44FF88",
    "#4488FF",
]


def detect_ships(
    img_rgb: np.ndarray,
    brightness_pct: float = 90,
    min_area: float = 60,
    max_area: float = 4000,
    min_aspect: float = 1.8,
    max_aspect: float = 18.0,
    use_sat: bool = True,
    kernel_size: int = 2,
) -> tuple[list[dict], np.ndarray]:
    """
    Detect ships in an RGB satellite image.

    Returns
    -------
    ships : list of dicts — each has 'id', 'label', 'bbox'=(x,y,w,h), 'area', 'aspect'
    mask  : uint8 ndarray — binary detection mask
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    thresh_v = np.percentile(v.astype(float), brightness_pct)
    mask_v = (v > thresh_v).astype(np.uint8) * 255

    if use_sat:
        thresh_s = np.percentile(s.astype(float), 92)
        mask_s = (s > thresh_s).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask_v, mask_s)
    else:
        mask = mask_v

    k = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ships: list[dict] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue

        x, y, w, h = cv2.boundingRect(contour)
        long_side = max(w, h)
        short_side = max(min(w, h), 1)
        aspect = long_side / short_side

        if not (min_aspect <= aspect <= max_aspect):
            continue
        if long_side < 8:
            continue

        ships.append(
            {
                "bbox": (x, y, w, h),
                "area": round(area, 1),
                "aspect": round(aspect, 2),
            }
        )

    ships.sort(key=lambda ship: -ship["area"])
    for i, ship in enumerate(ships):
        ship["id"] = i + 1
        ship["label"] = f"Ship {i + 1}"

    return ships, mask


def auto_params(img_rgb: np.ndarray) -> dict:
    """Select detection parameters based on mean image brightness."""
    mean_brightness = np.mean(img_rgb)

    if mean_brightness < 80:
        return dict(
            brightness_pct=88,
            min_area=50,
            max_area=5000,
            min_aspect=1.8,
            max_aspect=18,
            use_sat=True,
            kernel_size=2,
        )
    elif mean_brightness < 140:
        return dict(
            brightness_pct=93,
            min_area=80,
            max_area=2000,
            min_aspect=2.0,
            max_aspect=14,
            use_sat=False,
            kernel_size=2,
        )
    else:
        return dict(
            brightness_pct=95,
            min_area=60,
            max_area=3000,
            min_aspect=1.8,
            max_aspect=16,
            use_sat=False,
            kernel_size=3,
        )


def crop_ship(img_pil, bbox: tuple[int, int, int, int], padding: int = 12):
    """Crop a single ship from a PIL image with optional padding."""
    x, y, w, h = bbox
    iw, ih = img_pil.size
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(iw, x + w + padding)
    y2 = min(ih, y + h + padding)
    return img_pil.crop((x1, y1, x2, y2))


def nms_ships(ships: list[dict], iou_thresh: float = 0.3) -> list[dict]:
    """Non-maximum suppression: discard bboxes with IoU > iou_thresh."""
    if len(ships) < 2:
        return ships
    keep = []
    used: set[int] = set()
    for i, a in enumerate(ships):
        if i in used:
            continue
        xa, ya, wa, ha = a["bbox"]
        for j, b in enumerate(ships[i + 1 :], i + 1):
            xb, yb, wb, hb = b["bbox"]
            ix = max(0, min(xa + wa, xb + wb) - max(xa, xb))
            iy = max(0, min(ya + ha, yb + hb) - max(ya, yb))
            inter = ix * iy
            union = wa * ha + wb * hb - inter
            if union > 0 and inter / union > iou_thresh:
                used.add(j)
        keep.append(a)
    for i, ship in enumerate(keep):
        ship["id"] = i + 1
        ship["label"] = f"Ship {i + 1}"
    return keep


def auto_params_v2(img_rgb: np.ndarray) -> dict:
    """
    Estimate detection parameters for detect_ships_v2.
    Uses mean brightness and std-dev to distinguish cloudy images.
    """
    mean_brightness = np.mean(img_rgb)
    std_dev = np.std(img_rgb)
    has_clouds = std_dev > 55

    if mean_brightness < 80:
        return dict(
            min_area=250,
            max_area=8300,
            min_aspect=1.0,
            max_aspect=16.50,
            use_sat=False,
            use_canny=True,
            use_lab=True,
            kernel_open=3,
            kernel_close=7,
        )
    elif has_clouds:
        return dict(
            min_area=80,
            max_area=1500,
            min_aspect=2.0,
            max_aspect=12,
            use_sat=False,
            use_canny=True,
            use_lab=True,
            kernel_open=3,
            kernel_close=5,
        )
    else:
        return dict(
            min_area=60,
            max_area=3500,
            min_aspect=1.8,
            max_aspect=16,
            use_sat=True,
            use_canny=True,
            use_lab=True,
            kernel_open=3,
            kernel_close=5,
        )


def detect_ships_v2(
    img_rgb: np.ndarray,
    min_area: float = 250,
    max_area: float = 8300,
    min_aspect: float = 1.0,
    max_aspect: float = 16.50,
    use_sat: bool = False,
    use_canny: bool = True,
    use_lab: bool = True,
    kernel_open: int = 3,
    kernel_close: int = 7,
    nms_iou: float = 0.6,
) -> tuple[list[dict], np.ndarray]:
    """
    Improved ship detection pipeline v2.

    Changes vs v1
    -------------
    - CLAHE + Otsu instead of fixed-percentile threshold
    - Canny edge detection combined with the brightness mask
    - Asymmetric morphological kernels (open < close) + final dilate
    - Cloud / neutral-background exclusion via LAB colour space
    - NMS to remove duplicate bounding boxes for the same ship
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    _, mask_v = cv2.threshold(v_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if use_sat:
        thresh_s = np.percentile(s.astype(float), 92)
        mask_s = (s > thresh_s).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask_v, mask_s)
    else:
        mask = mask_v

    if use_canny:
        blur = cv2.GaussianBlur(v, (5, 5), 0)
        edges = cv2.Canny(blur, 30, 100)
        mask = cv2.bitwise_or(mask, edges)

    if use_lab:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        a_ch = lab[:, :, 1].astype(int)
        b_ch = lab[:, :, 2].astype(int)
        color_score = np.abs(a_ch - 128) + np.abs(b_ch - 128)
        mask_color = (color_score > 12).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, mask_color)

    k_open = np.ones((kernel_open, kernel_open), np.uint8)
    k_close = np.ones((kernel_close, kernel_close), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ships: list[dict] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        long_side = max(w, h)
        short_side = max(min(w, h), 1)
        aspect = long_side / short_side
        if not (min_aspect <= aspect <= max_aspect):
            continue
        if long_side < 8:
            continue
        ships.append({"bbox": (x, y, w, h), "area": round(area, 1), "aspect": round(aspect, 2)})

    ships.sort(key=lambda ship: -ship["area"])
    ships = nms_ships(ships, iou_thresh=nms_iou)

    return ships, mask

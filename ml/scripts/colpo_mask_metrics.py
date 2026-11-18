#!/usr/bin/env python3
"""Geometry / quality measurements on a lesion mask.

Not a detector. Acetic, green, and Lugol keep separate segmentation code.
These numbers describe a binary mask + the photo it came from.
Not gold. Not for lesion_boxes.jsonl.
"""

from __future__ import annotations

import cv2
import numpy as np


def _largest(mask: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num < 2:
        return np.zeros_like(mask, dtype=bool)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == i


def measure_lesion(
    lesion: np.ndarray,
    roi: np.ndarray,
    bgr: np.ndarray,
    extra: dict[str, np.ndarray] | None = None,
) -> dict[str, float | int]:
    """Area, opacity, border, shape, simple texture, quality.

    area_frac is relative to the cervix ROI, not the full frame.
    compactness: 1 = circle, lower = more elongated / ragged.
    border_sharp: contour gradient / ROI gradient (>1 = edge stands out).
    irregularity: perimeter / convex-hull perimeter (>=1).
    """
    extra = extra or {}
    h, w = lesion.shape
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    roi_n = max(int(roi.sum()), 1)
    out: dict[str, float | int] = {
        "present": int(bool(lesion.any())),
        "area_frac": float(lesion.sum() / roi_n),
        "mean_L": 0.0,
        "mean_S": 0.0,
        "dense_share": 0.0,
        "compactness": 0.0,
        "convexity": 0.0,
        "irregularity": 0.0,
        "border_sharp": 0.0,
        "texture_std": 0.0,
        "img_sharp": 0.0,
        "glare_frac": 0.0,
        "vessel_frac": 0.0,
        "vessel_in_lesion": 0.0,
    }
    glare = extra.get("glare")
    if glare is not None:
        out["glare_frac"] = float(glare[roi].mean()) if roi.any() else 0.0
    dense = extra.get("dense")
    if dense is not None and lesion.any():
        out["dense_share"] = float(dense[lesion].mean())
    ves = extra.get("vessels")
    if ves is not None:
        out["vessel_frac"] = float(ves[roi].mean()) if roi.any() else 0.0
        out["vessel_in_lesion"] = float(ves[lesion].mean()) if lesion.any() else 0.0

    lap = cv2.Laplacian(np.clip(L, 0, 255).astype(np.uint8), cv2.CV_32F)
    out["img_sharp"] = float(lap[roi].var()) if roi.any() else 0.0

    if not lesion.any():
        return out

    out["mean_L"] = float(L[lesion].mean())
    out["mean_S"] = float(sat[lesion].mean())
    out["texture_std"] = float(L[lesion].std())

    blob = _largest(lesion)
    if not blob.any():
        return out
    u8 = blob.astype(np.uint8)
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return out
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if peri > 1 and area > 1:
        out["compactness"] = float(4.0 * np.pi * area / (peri * peri))
        hull = cv2.convexHull(cnt)
        hull_a = float(cv2.contourArea(hull))
        hull_p = float(cv2.arcLength(hull, True))
        out["convexity"] = float(area / hull_a) if hull_a > 1 else 0.0
        out["irregularity"] = float(peri / hull_p) if hull_p > 1 else 0.0

    ring = cv2.dilate(u8, np.ones((5, 5), np.uint8)) ^ cv2.erode(u8, np.ones((5, 5), np.uint8))
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    edge_m = mag[ring.astype(bool)]
    roi_m = mag[roi]
    if edge_m.size and roi_m.size:
        out["border_sharp"] = float(edge_m.mean() / (float(roi_m.mean()) + 1e-3))
    return out

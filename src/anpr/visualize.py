"""Overlay drawing helpers (kept dependency-free apart from OpenCV)."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import PlateCandidate

GREEN = (0, 200, 0)
AMBER = (0, 200, 255)
WHITE = (255, 255, 255)


def draw_candidates(
    frame: np.ndarray,
    candidates: list[PlateCandidate],
    fps: float | None = None,
    stable_plate: str | None = None,
) -> np.ndarray:
    out = frame.copy()

    for cand in candidates:
        x1, y1, x2, y2 = cand.car_bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 2)

        if cand.plate_bbox is None:
            continue
        px1, py1, px2, py2 = cand.plate_bbox
        cv2.rectangle(out, (px1, py1), (px2, py2), AMBER, 2)

        if not cand.text:
            continue
        label = cand.text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(out, (px1, py1 - th - 8), (px1 + tw + 6, py1), AMBER, -1)
        cv2.putText(
            out, label, (px1 + 3, py1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA,
        )

    y = 28
    if fps is not None:
        cv2.putText(
            out, f"FPS: {fps:.1f}", (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA,
        )
        y += 30
    if stable_plate:
        cv2.putText(
            out, f"Plate: {stable_plate}", (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, AMBER, 2, cv2.LINE_AA,
        )
    return out

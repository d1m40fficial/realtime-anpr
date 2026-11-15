"""Plate anonymisation helpers (privacy-preserving publication).

A detector trained on vehicle crops misses plates when the vehicle is far away:
the plate inside a 340x230 crop is only ~80 px wide after resizing to the
network input. For anonymisation that is not acceptable — a missed plate is a
privacy leak — so the frame is additionally scanned with overlapping tiles at
native resolution, where the same plate stays ~150 px wide.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import CascadeDetector


def pixelate(frame: np.ndarray, bbox: tuple[int, int, int, int], blocks: int = 12) -> None:
    """Replace a region with a coarse mosaic (in place)."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    small = cv2.resize(
        roi,
        (max(1, (x2 - x1) // blocks), max(1, (y2 - y1) // blocks)),
        interpolation=cv2.INTER_LINEAR,
    )
    frame[y1:y2, x1:x2] = cv2.resize(
        small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
    )


def find_plates(
    detector: CascadeDetector,
    frame: np.ndarray,
    tile_size: int = 640,
    tile_step: int = 480,
    conf: float = 0.25,
    imgsz: int = 640,
) -> list[tuple[int, int, int, int]]:
    """All plate boxes found by the cascade plus a tiled full-frame scan."""
    boxes: list[tuple[int, int, int, int]] = []

    for cand in detector.detect(frame):
        if cand.plate_bbox is not None:
            boxes.append(cand.plate_bbox)

    height, width = frame.shape[:2]
    if height < tile_size or width < tile_size:
        tiles = [(0, 0, width, height)]
    else:
        tiles = [
            (x, y, x + tile_size, y + tile_size)
            for y in range(0, height - tile_size + 1, tile_step)
            for x in range(0, width - tile_size + 1, tile_step)
        ]
        # make sure the right/bottom edges are covered as well
        tiles.append((width - tile_size, 0, width, tile_size))
        tiles.append((0, height - tile_size, tile_size, height))
        tiles.append((width - tile_size, height - tile_size, width, height))

    for x1, y1, x2, y2 in set(tiles):
        patch = frame[y1:y2, x1:x2]
        result = detector.plate_detector.predict(
            source=patch,
            conf=conf,
            imgsz=imgsz,
            max_det=8,
            device=detector.device,
            verbose=False,
        )[0]
        for box in result.boxes:
            px1, py1, px2, py2 = map(int, box.xyxy[0])
            boxes.append((x1 + px1, y1 + py1, x1 + px2, y1 + py2))

    return boxes


def anonymize_frame(
    detector: CascadeDetector,
    frame: np.ndarray,
    blocks: int = 12,
    expand: float = 0.08,
    **kwargs,
) -> int:
    """Pixelate every plate in ``frame`` in place. Returns the number of regions."""
    count = 0
    for x1, y1, x2, y2 in find_plates(detector, frame, **kwargs):
        mx, my = int((x2 - x1) * expand), int((y2 - y1) * expand)
        pixelate(frame, (x1 - mx, y1 - my, x2 + mx, y2 + my), blocks)
        count += 1
    return count

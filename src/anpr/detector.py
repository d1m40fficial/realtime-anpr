"""Two-stage detector: vehicle on the full frame, plate inside the vehicle ROI.

Why a cascade instead of a single plate detector on the full frame?
  * the plate occupies a tiny fraction of a 2560x1920 surveillance frame, so
    downscaling the whole frame to the network input destroys the characters;
  * cropping to the vehicle keeps the plate at a usable resolution;
  * the second stage only ever sees car-like content, which removes most
    false positives coming from road signs and billboards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PipelineConfig


@dataclass
class PlateCandidate:
    car_bbox: tuple[int, int, int, int]
    plate_bbox: tuple[int, int, int, int] | None = None
    text: str = ""
    ocr_conf: float = 0.0
    plate_det_conf: float = 0.0
    area: float = 0.0
    center: tuple[int, int] = field(default=(0, 0))


class CascadeDetector:
    """YOLO11n (COCO vehicles) -> YOLO11n (plates) -> cropped plate."""

    def __init__(self, cfg: PipelineConfig) -> None:
        from ultralytics import YOLO

        self.cfg = cfg
        self.device, self.half = cfg.resolved_device()

        self.car_detector = YOLO(cfg.car_model)
        self.plate_detector = YOLO(cfg.plate_model)
        if self.device != "cpu":
            self.car_detector.to(self.device)
            self.plate_detector.to(self.device)

    # -- geometry ----------------------------------------------------------
    def plate_shape_ok(self, w: int, h: int, car_area: int | None = None) -> bool:
        cfg = self.cfg
        if w <= 0 or h <= 0:
            return False
        area = w * h
        aspect = w / max(h, 1)
        if area < cfg.min_plate_area:
            return False
        if not cfg.min_plate_aspect <= aspect <= cfg.max_plate_aspect:
            return False
        if car_area:
            ratio = area / car_area
            if not cfg.min_plate_area_ratio <= ratio <= cfg.max_plate_area_ratio:
                return False
        return True

    # -- detection ---------------------------------------------------------
    def _runtime_kwargs(self) -> dict:
        kwargs = {"device": self.device, "verbose": False}
        if self.half:
            kwargs["half"] = True
        return kwargs

    def detect(self, frame: np.ndarray) -> list[PlateCandidate]:
        cfg = self.cfg
        h, w = frame.shape[:2]
        frame_area = h * w
        found: list[PlateCandidate] = []

        car_results = self.car_detector.predict(
            source=frame,
            conf=cfg.conf_car,
            imgsz=cfg.car_imgsz,
            classes=list(cfg.vehicle_classes),
            max_det=cfg.max_cars,
            **self._runtime_kwargs(),
        )[0]

        cars: list[tuple[float, int, int, int, int]] = []
        for box in car_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            if area < frame_area * cfg.min_car_area_ratio:
                continue
            cars.append((area, x1, y1, x2, y2))
        cars.sort(reverse=True)

        for _, x1, y1, x2, y2 in cars[: cfg.max_cars]:
            car_crop = frame[y1:y2, x1:x2]
            if car_crop.size == 0:
                continue
            car_area = car_crop.shape[0] * car_crop.shape[1]

            plate_results = self.plate_detector.predict(
                source=car_crop,
                conf=cfg.conf_plate,
                imgsz=cfg.plate_imgsz,
                max_det=2,
                **self._runtime_kwargs(),
            )[0]
            if len(plate_results.boxes) == 0:
                continue

            candidates: list[tuple[float, int, int, int, int, int, float]] = []
            for box in plate_results.boxes:
                px1, py1, px2, py2 = map(int, box.xyxy[0])
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(car_crop.shape[1], px2), min(car_crop.shape[0], py2)
                if px2 <= px1 or py2 <= py1:
                    continue

                pw, ph = px2 - px1, py2 - py1
                area = pw * ph
                if area < cfg.min_plate_area:
                    continue
                if not self.plate_shape_ok(pw, ph, car_area):
                    continue

                det_conf = float(box.conf[0])
                aspect = pw / max(ph, 1)
                # Prefer confident, wide and large boxes: a plate is a wide
                # rectangle, so an unusually tall/large "plate" is usually a
                # headlight or a radiator grille.
                score = (
                    0.65 * det_conf
                    + 0.25 * min(aspect / 4.0, 1.5)
                    + 0.10 * min(area / 2500.0, 1.5)
                )
                candidates.append((score, area, px1, py1, px2, py2, det_conf))

            if not candidates:
                continue

            candidates.sort(reverse=True)
            _, area, px1, py1, px2, py2, det_conf = candidates[0]
            found.append(
                PlateCandidate(
                    car_bbox=(x1, y1, x2, y2),
                    plate_bbox=(x1 + px1, y1 + py1, x1 + px2, y1 + py2),
                    plate_det_conf=det_conf,
                    area=float(area),
                    center=((x1 + px1 + x1 + px2) // 2, (y1 + py1 + y1 + py2) // 2),
                )
            )

        return found

    @staticmethod
    def crop(frame: np.ndarray, cand: PlateCandidate) -> np.ndarray:
        x1, y1, x2, y2 = cand.plate_bbox
        return frame[y1:y2, x1:x2]

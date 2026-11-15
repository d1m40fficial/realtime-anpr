"""Configuration objects for the ANPR pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class PipelineConfig:
    """All tunable parameters in one place.

    The defaults correspond to the *balanced* configuration that won the
    tuning sweep described in the README (65.9% plate-level accuracy on the
    91-clip real-world test set at ~46 FPS on an RTX 5070).
    """

    # --- models -----------------------------------------------------------
    car_model: str = "yolo11n.pt"  # Ultralytics downloads the COCO weights on first use
    plate_model: str = "models/yolo11n_plate_best.pt"
    ocr_model: str = "models/ru_plate_cct_xs.onnx"
    ocr_config: str = "configs/ru_plate_config.yaml"

    # --- vehicle detector -------------------------------------------------
    vehicle_classes: tuple[int, ...] = (2, 3, 5, 7)  # car, motorcycle, bus, truck
    conf_car: float = 0.45
    car_imgsz: int = 960
    max_cars: int = 2
    min_car_area_ratio: float = 0.02

    # --- plate detector ---------------------------------------------------
    conf_plate: float = 0.60
    plate_imgsz: int = 320
    min_plate_area: int = 420
    min_plate_area_ratio: float = 0.004
    max_plate_area_ratio: float = 0.45
    min_plate_aspect: float = 1.6
    max_plate_aspect: float = 8.5

    # --- aggregation ------------------------------------------------------
    min_ocr_conf: float = 0.18
    area_keep_ratio: float = 0.35
    early_stop_votes: int = 3
    early_stop_score: float = 1.9

    # --- runtime ----------------------------------------------------------
    sample_every_n: int = 2
    use_clahe: bool = True
    device: str | None = None  # None -> auto ("cuda:0" if available else "cpu")
    half: bool | None = None  # None -> auto (True on CUDA)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        if "vehicle_classes" in data:
            data["vehicle_classes"] = tuple(data["vehicle_classes"])
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        data = asdict(self)
        data["vehicle_classes"] = list(self.vehicle_classes)
        Path(path).write_text(
            yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )

    def resolved_device(self) -> tuple[str, bool]:
        """Return (torch device string, use_half)."""
        device = self.device
        half = self.half
        if device is None:
            try:
                import torch

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:  # pragma: no cover - torch is optional for postprocessing
                device = "cpu"
        if half is None:
            half = device.startswith("cuda")
        return device, bool(half)

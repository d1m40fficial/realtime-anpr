"""Real-time ANPR: YOLO11n cascade + CCT-XS OCR for Russian plates.

Heavy dependencies (Ultralytics, ONNX Runtime) are imported lazily, so the
post-processing and aggregation modules can be used — and tested — without a
GPU stack installed.
"""

from __future__ import annotations

from .config import PipelineConfig
from .postprocess import normalize_plate_text, postprocess_plate

__version__ = "1.0.0"

__all__ = [
    "ALPRPipeline",
    "CascadeDetector",
    "ClipAggregator",
    "ClipResult",
    "PipelineConfig",
    "PlateCandidate",
    "PlateOCR",
    "StreamStats",
    "StreamingAggregator",
    "normalize_plate_text",
    "postprocess_plate",
]

_LAZY = {
    "ALPRPipeline": ("pipeline", "ALPRPipeline"),
    "ClipResult": ("pipeline", "ClipResult"),
    "StreamStats": ("pipeline", "StreamStats"),
    "CascadeDetector": ("detector", "CascadeDetector"),
    "PlateCandidate": ("detector", "PlateCandidate"),
    "PlateOCR": ("ocr", "PlateOCR"),
    "ClipAggregator": ("aggregation", "ClipAggregator"),
    "StreamingAggregator": ("aggregation", "StreamingAggregator"),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        import importlib

        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

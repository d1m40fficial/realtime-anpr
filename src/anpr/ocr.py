"""FastPlate CCT-XS inference over ONNX Runtime.

The model is a 9-slot classifier, not a CTC network: the output is
``(1, 9, num_chars)`` and every slot is decoded independently. Applying greedy
CTC collapsing here (as one would for CRNN/LPRNet) silently deletes repeated
characters and breaks plates such as ``A111AA``.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import yaml

from .postprocess import normalize_plate_text, postprocess_plate

INPUT_SIZE = (140, 70)  # (width, height) expected by the CCT-XS model


class PlateOCR:
    """ONNX wrapper for the fine-tuned Russian plate recogniser."""

    def __init__(
        self,
        model_path: str | Path,
        config_path: str | Path,
        providers: list[str] | None = None,
        use_clahe: bool = True,
    ) -> None:
        import onnxruntime as ort

        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.alphabet: str = cfg["alphabet"]
        self.pad_char: str = cfg.get("pad_char", "_")
        self.pad_index = (
            self.alphabet.index(self.pad_char)
            if self.pad_char in self.alphabet
            else len(self.alphabet) - 1
        )
        self.use_clahe = use_clahe

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        options.inter_op_num_threads = 1

        if providers is None:
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = self.session.get_inputs()[0].type
        self.providers = self.session.get_providers()

    # -- preprocessing -----------------------------------------------------
    def _prepare(self, plate_crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
        small = cv2.resize(gray, INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        if "float" in self.input_type:
            return small.astype(np.float32)[None, :, :, None] / 255.0
        return small[None, :, :, None]

    # -- decoding ----------------------------------------------------------
    def decode(self, logits: np.ndarray) -> tuple[str, float]:
        """Per-slot argmax decoding with a mean-confidence score."""
        slot_probs = _softmax(logits[0])
        indices = slot_probs.argmax(axis=-1)
        chars: list[str] = []
        confidences: list[float] = []
        for step, idx in enumerate(indices):
            if idx == self.pad_index:
                continue
            chars.append(self.alphabet[idx])
            confidences.append(float(slot_probs[step, idx]))
        score = float(np.mean(confidences)) if confidences else 0.0
        return "".join(chars), score

    # -- public API --------------------------------------------------------
    def recognize(self, plate_crop: np.ndarray) -> tuple[str, float]:
        """Return ``(normalised_plate, confidence)`` for a BGR plate crop."""
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0
        try:
            tensor = self._prepare(plate_crop)
            logits = self.session.run(None, {self.input_name: tensor})[0]
            raw, score = self.decode(logits)
            return postprocess_plate(normalize_plate_text(raw)), score
        except Exception:  # pragma: no cover - defensive: a bad crop must not kill the loop
            return "", 0.0


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=-1, keepdims=True)

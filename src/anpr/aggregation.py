"""Temporal aggregation of per-frame OCR results.

A single frame is a noisy observation: motion blur, glare or a partially
occluded plate can produce a plausible-looking but wrong string. Two different
aggregators are used for the two operating modes:

``ClipAggregator``      - offline: weighted voting over the whole clip with a
                          two-tier candidate pool (strict GOST-valid strings
                          first, then looser ones) plus a per-character vote.
``StreamingAggregator`` - live: a sliding window of the last N readings, so a
                          car that left the scene is forgotten automatically
                          and a new plate can be reported without an explicit
                          "clip boundary" event.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass

from .postprocess import is_soft_plate, is_strict_plate, postprocess_plate


@dataclass
class Observation:
    text: str
    area: float
    ocr_conf: float


class ClipAggregator:
    """Offline aggregation over all detections of one video clip."""

    def __init__(self, min_ocr_conf: float = 0.18, area_keep_ratio: float = 0.35) -> None:
        self.min_ocr_conf = min_ocr_conf
        self.area_keep_ratio = area_keep_ratio
        self.observations: list[Observation] = []

    def add(self, text: str, area: float, ocr_conf: float) -> None:
        text = postprocess_plate(text)
        if text and is_soft_plate(text):
            self.observations.append(Observation(text, float(area), float(ocr_conf)))

    def best(self) -> str:
        prepared = self.observations
        if not prepared:
            return "NOTDETECTED"

        # Tier 1: valid GOST plates read confidently. Tier 2: confident-ish
        # readings of any length. Tier 3: everything we have.
        strict = [o for o in prepared if is_strict_plate(o.text) and o.ocr_conf >= self.min_ocr_conf]
        soft = [o for o in prepared if o.ocr_conf >= max(0.08, self.min_ocr_conf * 0.5)]
        working = strict or soft or prepared

        max_area = max((o.area for o in working), default=0.0)
        large = [o for o in working if max_area <= 0 or o.area >= max_area * self.area_keep_ratio]
        large = large or working

        weights: dict[str, float] = defaultdict(float)
        counts: Counter[str] = Counter()
        for o in large:
            area_norm = o.area / max(max_area, 1.0)
            weight = 0.5 * area_norm + 0.5 * max(o.ocr_conf, 0.05)
            weights[o.text] += weight
            counts[o.text] += 1

        # A string seen twice already beats any character-level reconstruction.
        regex_candidates = [t for t in weights if is_strict_plate(t)]
        if regex_candidates:
            best_full = max(regex_candidates, key=lambda t: (counts[t], weights[t], len(t)))
            if counts[best_full] >= 2:
                return best_full

        # Otherwise vote character by character, weighting by confidence and
        # plate area (a larger plate means a sharper crop).
        max_len = max(len(o.text) for o in large)
        merged: list[str] = []
        for pos in range(max_len):
            votes: dict[str, float] = defaultdict(float)
            for o in large:
                if pos >= len(o.text):
                    continue
                area_norm = o.area / max(max_area, 1.0)
                votes[o.text[pos]] += 0.55 * max(o.ocr_conf, 0.05) + 0.45 * area_norm
            if votes:
                merged.append(max(votes.items(), key=lambda kv: kv[1])[0])

        candidate = postprocess_plate("".join(merged))
        if is_strict_plate(candidate):
            return candidate

        fallback = max(
            weights,
            key=lambda t: (is_strict_plate(t), counts[t], weights[t], len(t)),
        )
        return fallback or "NOTDETECTED"


class StreamingAggregator:
    """Sliding-window voting for a continuous RTSP stream."""

    def __init__(self, window: int = 30, stable_votes: int = 3) -> None:
        self.window = window
        self.stable_votes = stable_votes
        self.recent: deque[str] = deque(maxlen=window)
        self.votes: Counter[str] = Counter()
        self.current = ""

    def add(self, text: str) -> str:
        """Feed one reading; returns the currently stable plate (or "")."""
        if not text or len(text) < 8:
            return self.current

        if len(self.recent) == self.recent.maxlen:
            self._drop(self.recent[0])
        self.recent.append(text)
        self.votes[text] += 1

        if self.votes[text] >= self.stable_votes:
            self.current = text
        return self.current

    def _drop(self, text: str) -> None:
        self.votes[text] -= 1
        if self.votes[text] <= 0:
            del self.votes[text]

    def top(self, n: int = 3) -> dict[str, int]:
        return dict(self.votes.most_common(n))

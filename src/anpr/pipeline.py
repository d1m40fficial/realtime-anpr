"""End-to-end ALPR pipelines: offline clip processing and live RTSP.

The two modes share the same detector, OCR and post-processing; only the input
and the aggregation strategy differ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .aggregation import ClipAggregator, StreamingAggregator
from .config import PipelineConfig
from .detector import CascadeDetector
from .ocr import PlateOCR
from .visualize import draw_candidates


@dataclass
class ClipResult:
    clip: str
    plate: str
    frames_seen: int
    frames_processed: int
    elapsed_sec: float
    char_accuracy: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames_processed / self.elapsed_sec if self.elapsed_sec > 0 else 0.0


@dataclass
class StreamStats:
    frames: int = 0
    last_plate: str = ""
    history: list[str] = field(default_factory=list)


class ALPRPipeline:
    """Cascade detector + OCR + post-processing in one object."""

    def __init__(self, cfg: PipelineConfig | None = None) -> None:
        self.cfg = cfg or PipelineConfig()
        self.detector = CascadeDetector(self.cfg)
        self.ocr = PlateOCR(
            self.cfg.ocr_model,
            self.cfg.ocr_config,
            use_clahe=self.cfg.use_clahe,
        )

    # -- one frame ---------------------------------------------------------
    def read_frame(self, frame):
        """Run the cascade on a single frame, return candidates with text."""
        candidates = self.detector.detect(frame)
        for cand in candidates:
            if cand.plate_bbox is None:
                continue
            crop = self.detector.crop(frame, cand)
            cand.text, cand.ocr_conf = self.ocr.recognize(crop)
        return candidates

    # -- offline -----------------------------------------------------------
    def process_clip(
        self,
        clip_path: str | Path,
        save_video: str | Path | None = None,
        max_frames: int | None = None,
    ) -> ClipResult:
        cfg = self.cfg
        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            raise FileNotFoundError(clip_path)

        writer = None
        if save_video is not None:
            fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
            size = (int(cap.get(3)), int(cap.get(4)))
            writer = cv2.VideoWriter(
                str(save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, fps_in / cfg.sample_every_n), size,
            )

        aggregator = ClipAggregator(cfg.min_ocr_conf, cfg.area_keep_ratio)
        frame_idx = processed = 0
        elapsed = 0.0
        stable = ""

        while True:
            ok = cap.grab()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame_idx % cfg.sample_every_n != 0:
                frame_idx += 1
                continue

            ok, frame = cap.retrieve()
            if not ok:
                frame_idx += 1
                continue

            t0 = time.perf_counter()
            candidates = self.read_frame(frame)
            elapsed += time.perf_counter() - t0
            processed += 1
            frame_idx += 1

            for cand in candidates:
                if not cand.text:
                    continue
                aggregator.add(cand.text, cand.area, cand.ocr_conf)
                if writer is not None and cand.text:
                    stable = cand.text

            if writer is not None:
                writer.write(draw_candidates(frame, candidates, stable_plate=stable))

        cap.release()
        if writer is not None:
            writer.release()

        return ClipResult(
            clip=str(clip_path),
            plate=aggregator.best(),
            frames_seen=frame_idx,
            frames_processed=processed,
            elapsed_sec=elapsed,
        )

    # -- live --------------------------------------------------------------
    def run_stream(self, url: str, on_frame=None, max_seconds: float | None = None) -> StreamStats:
        """Process an RTSP stream until Ctrl+C (or ``max_seconds``)."""
        from .rtsp import RTSPReader

        cfg = self.cfg
        reader = RTSPReader(url).start()
        aggregator = StreamingAggregator()
        stats = StreamStats()
        last_idx = -1
        start = time.time()

        try:
            if reader.wait_first_frame()[0] is None:
                raise RuntimeError("No frames received from the camera")

            while max_seconds is None or time.time() - start < max_seconds:
                frame, idx = reader.read()
                if frame is None or idx == last_idx:
                    time.sleep(0.01)
                    continue
                if last_idx >= 0 and idx - last_idx < cfg.sample_every_n:
                    continue
                last_idx = idx

                candidates = self.read_frame(frame)
                for cand in candidates:
                    if cand.text:
                        stats.last_plate = aggregator.add(cand.text) or stats.last_plate
                stats.frames += 1

                if on_frame is not None:
                    on_frame(
                        frame,
                        candidates,
                        stats.last_plate,
                        aggregator.top(),
                    )
        except KeyboardInterrupt:
            pass
        finally:
            reader.stop()

        if stats.last_plate:
            stats.history.append(stats.last_plate)
        return stats

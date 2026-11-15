#!/usr/bin/env python
"""Anonymise a video by pixelating every detected license plate.

Useful when footage has to be shown publicly (portfolio, reports, demos):
the vehicle stays visible, the plate does not.

    python scripts/anonymize_clip.py --clip raw.avi --out redacted.mp4
    python scripts/anonymize_clip.py --clip raw.avi --out redacted.mp4 --blocks 16

The script scans each frame twice — through the vehicle cascade and with
overlapping full-resolution tiles — because a cascade tuned for recognition
misses distant plates. Only the plate detector is loaded, so the OCR model is
not required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anpr import CascadeDetector, PipelineConfig  # noqa: E402
from anpr.anonymize import anonymize_frame  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pixelate license plates in a video")
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=12, help="mosaic block size in pixels")
    parser.add_argument("--expand", type=float, default=0.08, help="extra margin around the plate")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--tile-step", type=int, default=480)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = PipelineConfig()
    # An anonymiser should over-detect: masking a random bumper is cheap,
    # leaving a readable plate in the background is not.
    cfg.max_cars = 16
    cfg.conf_car = 0.20
    cfg.conf_plate = 0.25
    cfg.min_plate_area = 120
    cfg.min_plate_area_ratio = 0.0005
    if args.device:
        cfg.device = args.device
        cfg.half = False
    detector = CascadeDetector(cfg)

    cap = cv2.VideoCapture(str(args.clip))
    if not cap.isOpened():
        raise FileNotFoundError(args.clip)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(cap.get(3)), int(cap.get(4)))
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    masked = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break
        frame_idx += 1

        masked += anonymize_frame(
            detector,
            frame,
            blocks=args.blocks,
            expand=args.expand,
            tile_size=args.tile_size,
            tile_step=args.tile_step,
        )
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"frames: {frame_idx}, plate regions pixelated: {masked} -> {args.out}")


if __name__ == "__main__":
    main()

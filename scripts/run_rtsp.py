#!/usr/bin/env python
"""Live recognition from an IP camera over RTSP.

Credentials are read from the environment, never from source:

    set RTSP_HOST=192.168.1.64:554
    set RTSP_PATH=/Streaming/Channels/101
    set RTSP_USER=admin
    set RTSP_PASS=secret
    python scripts/run_rtsp.py

Alternatively pass a complete URL: ``python scripts/run_rtsp.py --url rtsp://...``

The window shows the vehicle box (green), the plate box (amber) and the plate
value that has been confirmed by the voting window. Press ``q`` to stop.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anpr import ALPRPipeline, PipelineConfig  # noqa: E402
from anpr.rtsp import build_rtsp_url  # noqa: E402
from anpr.visualize import draw_candidates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Live ANPR over RTSP")
    parser.add_argument("--url", default=None, help="full rtsp:// URL (overrides env)")
    parser.add_argument("--host", default=os.getenv("RTSP_HOST", "127.0.0.1:554"))
    parser.add_argument("--path", default=os.getenv("RTSP_PATH", "/live/main"))
    parser.add_argument("--user", default=os.getenv("RTSP_USER", ""))
    parser.add_argument("--password", default=os.getenv("RTSP_PASS", ""))
    parser.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    parser.add_argument("--no-window", action="store_true", help="headless mode")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    url = args.url or build_rtsp_url(args.host, args.path, args.user, args.password)
    pipeline = ALPRPipeline(PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig())
    print("Connecting to", url.split("@")[-1])

    def on_frame(frame, candidates, plate, votes) -> None:
        if args.no_window:
            if plate:
                print("PLATE:", plate, "| votes:", votes)
            return
        import cv2

        cv2.imshow("ANPR", draw_candidates(frame, candidates, stable_plate=plate))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            raise KeyboardInterrupt

    stats = pipeline.run_stream(url, on_frame=on_frame, max_seconds=args.seconds)
    print(f"frames processed: {stats.frames}, last stable plate: {stats.last_plate or '---'}")

    if not args.no_window:
        import cv2

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

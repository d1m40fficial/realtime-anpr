#!/usr/bin/env python
"""Process a single video file (or a whole folder) through the ALPR pipeline.

Examples
--------
    python scripts/process_clip.py --clip data/clips/01_A123BC177.avi
    python scripts/process_clip.py --clip sample.mp4 --save-video out/demo.mp4
    python scripts/process_clip.py --dir data/clips --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anpr import ALPRPipeline, PipelineConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ANPR pipeline on video files")
    parser.add_argument("--clip", type=Path, help="single video file")
    parser.add_argument("--dir", type=Path, help="folder with video files")
    parser.add_argument("--limit", type=int, default=None, help="max files from --dir")
    parser.add_argument("--save-video", type=Path, default=None, help="annotated output (single clip)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None, help="pipeline YAML config")
    parser.add_argument("--device", default=None, help="cuda:0 / cpu (default: auto)")
    parser.add_argument("--json", type=Path, default=None, help="write results to JSON")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    if args.device:
        cfg.device = args.device
        cfg.half = args.device.startswith("cuda")
    pipeline = ALPRPipeline(cfg)

    clips: list[Path] = []
    if args.clip:
        clips.append(args.clip)
    elif args.dir:
        clips = sorted(
            p for p in args.dir.iterdir() if p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
        )
        if args.limit:
            clips = clips[: args.limit]
    else:
        parser.error("pass --clip or --dir")

    results = []
    for clip in clips:
        save = args.save_video if len(clips) == 1 else None
        res = pipeline.process_clip(clip, save_video=save, max_frames=args.max_frames)
        print(
            f"{clip.name:>40} -> {res.plate:<12} "
            f"frames={res.frames_processed:>5} time={res.elapsed_sec:6.2f}s fps={res.fps:5.1f}"
        )
        results.append(res.__dict__)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    if clips:
        total_time = sum(r["elapsed_sec"] for r in results)
        total_frames = sum(r["frames_processed"] for r in results)
        print(f"\nprocessed {total_frames} frames in {total_time:.1f}s "
              f"(avg {total_frames / max(total_time, 1e-6):.1f} FPS)")


if __name__ == "__main__":
    main()

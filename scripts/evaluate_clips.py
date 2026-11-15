#!/usr/bin/env python
"""Reproduce the clip-level evaluation reported in the README.

The dataset convention is ``NN_PLATE.avi`` (e.g. ``01_A123BC177.avi``): the
ground truth is encoded in the file name, one file = one entry/exit event.
Clips named ``*_UNKNOWN.avi`` have no ground truth and are skipped.

    python scripts/evaluate_clips.py --dir data/clips --out results.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anpr import ALPRPipeline, PipelineConfig  # noqa: E402


def ground_truth(path: Path) -> str | None:
    name = path.stem
    gt = name.split("_", 1)[-1] if "_" in name else name
    gt = re.sub(r"[^A-Z0-9]", "", gt.upper())
    return None if gt in {"", "UNKNOWN", "NOTDETECTED"} else gt


def char_accuracy(pred: str, gt: str) -> float:
    if not gt or not pred or pred == "NOTDETECTED":
        return 0.0
    matches = sum(p == g for p, g in zip(pred, gt))
    return 100.0 * matches / max(len(gt), len(pred), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the pipeline on a clip set")
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results.csv"))
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    pipeline = ALPRPipeline(PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig())
    clips = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in {".avi", ".mp4", ".mkv"})

    rows = []
    for i, clip in enumerate(clips, start=1):
        gt = ground_truth(clip)
        if gt is None:
            continue
        res = pipeline.process_clip(clip)
        acc = char_accuracy(res.plate, gt)
        rows.append(
            {
                "clip": clip.name,
                "gt": gt,
                "recognized": res.plate,
                "char_acc": round(acc, 1),
                "frames_processed": res.frames_processed,
                "elapsed_sec": round(res.elapsed_sec, 3),
            }
        )
        print(f"[{i:02d}/{len(clips)}] {clip.name} -> {res.plate} (gt {gt}) char_acc={acc:.1f}%")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    total = len(df)
    correct = int((df["gt"] == df["recognized"]).sum())
    missed = int((df["recognized"] == "NOTDETECTED").sum())
    print("-" * 60)
    print(f"clips            : {total}")
    print(f"correct          : {correct} ({correct / max(total, 1) * 100:.1f}%)")
    print(f"wrong / missed   : {total - correct - missed} / {missed}")
    print(f"mean char acc    : {df['char_acc'].mean():.1f}%")
    print(f"mean FPS         : {df['frames_processed'].sum() / max(df['elapsed_sec'].sum(), 1e-6):.1f}")


if __name__ == "__main__":
    main()

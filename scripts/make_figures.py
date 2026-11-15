#!/usr/bin/env python
"""Regenerate every chart used in the README.

All numbers come from the recorded experiments (see README tables); nothing is
synthetic. Run from the repository root:

    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "assets"
OUT.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 170,
        "savefig.dpi": 170,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)

BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
GREEN = "#2f855a"
RED = "#c53030"
GRAY = "#718096"
LIGHT = "#e2e8f0"


# ---------------------------------------------------------------------------
# 1. OCR engine benchmark
# ---------------------------------------------------------------------------
def fig_ocr() -> None:
    engines = ["Tesseract\nPSM 8", "EasyOCR", "RapidOCR", "FastPlate\n(base)", "CCT-XS\n(fine-tuned)"]
    plate_gpu = [65.2, 45.4, 93.6, 52.0, 99.09]
    plate_cpu = [67.2, 45.4, 95.2, 46.2, 97.8]
    char_gpu = [93.2, 85.2, 97.81, 84.32, 99.89]
    ms_gpu = [195.2, 25.8, 38.34, 7.62, 2.10]
    ms_cpu = [194.1, 520.5, 65.87, 11.51, 9.67]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4))

    x = np.arange(len(engines))
    w = 0.38
    b1 = ax1.bar(x - w / 2, plate_gpu, w, label="GPU", color=BLUE)
    b2 = ax1.bar(x + w / 2, plate_cpu, w, label="CPU", color="#90cdf4")
    ax1.set_xticks(x, engines)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel("Plate accuracy, %")
    ax1.set_title("Full-plate accuracy on 45 511 Russian plates")
    ax1.legend(frameon=False, loc="upper left")
    ax1.bar_label(b1, fmt="%.1f", padding=2, fontsize=7.5)
    ax1.bar_label(b2, fmt="%.1f", padding=2, fontsize=7.5)
    ax1.axhline(99.09, color=GREEN, lw=1, ls=":", alpha=0.8)

    # accuracy vs latency, both axes logarithmic
    ax2.set_xscale("log")
    for eng, mg, mc, pg in zip(engines, ms_gpu, ms_cpu, plate_gpu):
        ax2.scatter(mg, pg, s=70, color=ORANGE, zorder=3, edgecolor="white", linewidth=0.8)
        ax2.scatter(mc, pg, s=70, color=BLUE, zorder=3, marker="s", edgecolor="white", linewidth=0.8)
        ax2.annotate(eng.replace("\n", " "), (mg, pg), textcoords="offset points",
                     xytext=(6, 6), fontsize=7)
    ax2.set_xlabel("Latency per plate, ms (log)")
    ax2.set_ylabel("Plate accuracy, %")
    ax2.set_title("Accuracy / latency trade-off")
    ax2.axvline(33.3, color=RED, lw=1, ls="--", alpha=0.6)
    ax2.text(34.5, 50, "real-time budget\n(30 FPS)", color=RED, fontsize=7.5)
    ax2.grid(True, which="both", alpha=0.2, ls="--")

    fig.tight_layout()
    fig.savefig(OUT / "fig_ocr_benchmark.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. YOLO plate detector benchmark
# ---------------------------------------------------------------------------
def fig_yolo() -> None:
    models = ["YOLOv8n", "YOLOv10n", "YOLO11n", "YOLO12n"]
    params = [3.01, 2.27, 2.58, 2.56]
    express_map50 = [0.9668, 0.9556, 0.9667, 0.9645]
    express_map5095 = [0.6739, 0.6603, 0.6662, 0.6679]
    ms_express = [2.67, 2.76, 2.75, 4.00]

    final = {
        "YOLOv8n": (0.983, 0.944, 0.974, 0.713, 1.7),
        "YOLO11n": (0.988, 0.951, 0.975, 0.713, 1.8),
        "YOLO12n": (0.983, 0.949, 0.975, 0.715, 2.3),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.6))

    x = np.arange(len(models))
    ax1.bar(x - 0.2, express_map50, 0.38, label="mAP@50", color=BLUE)
    ax1.bar(x + 0.2, express_map5095, 0.38, label="mAP@50-95", color=ORANGE)
    ax1.set_xticks(x, [f"{m}\n{p}M / {t} ms" for m, p, t in zip(models, params, ms_express)])
    ax1.set_ylim(0, 1.12)
    ax1.set_title("5-epoch express benchmark (Tesla T4)")
    ax1.set_ylabel("mAP")
    ax1.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    for xi, v in zip(x - 0.2, express_map50):
        ax1.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + 0.2, express_map5095):
        ax1.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.5)

    names = list(final.keys())
    metrics = {
        "Precision": [final[n][0] for n in names],
        "Recall": [final[n][1] for n in names],
        "mAP@50-95": [final[n][3] for n in names],
    }
    xs = np.arange(len(names))
    width = 0.26
    colors = [BLUE, ORANGE, GREEN]
    for i, (label, values) in enumerate(metrics.items()):
        offset = (i - 1) * width
        bars = ax2.bar(xs + offset, values, width, label=label, color=colors[i])
        ax2.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5, rotation=90)
    ax2.set_xticks(xs, [f"{n}\n{final[n][4]} ms" for n in names])
    ax2.set_ylim(0.60, 1.06)
    ax2.set_title("Final models after full training")
    ax2.set_ylabel("Score")
    ax2.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    ax2.axvspan(-0.5, 0.5, color=GREEN, alpha=0.06)
    ax2.text(0, 1.03, "chosen", ha="center", fontsize=8.5, color=GREEN, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "fig_yolo_benchmark.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Pipeline tuning sweep
# ---------------------------------------------------------------------------
def fig_tuning() -> None:
    configs = ["Base\n(all frames)", "Aggressive\nspeed-up", "Hard\nfiltering", "Balanced\n(chosen)", "Fine-tuned"]
    correct = np.array([63, 46, 36, 60, 59])
    wrong = np.array([24, 44, 1, 22, 23])
    missed = np.array([4, 1, 54, 9, 9])
    fps = [24.9, 37.7, 47.5, 46.6, 47.8]
    char_acc = [87.9, 80.1, 40.5, 80.7, 81.0]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13.0, 4.8), gridspec_kw={"width_ratios": [1.45, 1]}
    )
    x = np.arange(len(configs))
    ax1.bar(x, correct, 0.58, label="Recognised correctly", color=GREEN)
    ax1.bar(x, wrong, 0.58, bottom=correct, label="Recognised wrongly", color=RED)
    ax1.bar(x, missed, 0.58, bottom=correct + wrong, label="No result", color=LIGHT)
    ax1.set_xticks(x, configs)
    ax1.set_ylabel("Clips out of 91")
    ax1.set_ylim(0, 100)
    ax1.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    ax1.set_title("Tuning the cascade on 91 parking-camera clips")

    for xi, c, wv, mv in zip(x, correct, wrong, missed):
        ax1.text(xi, c / 2, f"{c}\n{c / 91 * 100:.1f}%", ha="center", va="center",
                 color="white", fontsize=9.5, fontweight="bold")
        ax1.text(xi, c + wv / 2, str(wv), ha="center", va="center", color="white", fontsize=8.5)
        ax1.text(xi, c + wv + mv / 2, str(mv), ha="center", va="center", color="#4a5568", fontsize=8.5)

    ax2.plot(x, fps, color=BLUE, marker="o", lw=2, label="Throughput, FPS")
    ax2.set_ylabel("Throughput, FPS (RTX 5070)", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax2.set_ylim(0, 58)
    ax2.set_xticks(x, ["Base", "Aggressive", "Hard filter", "Balanced", "Fine-tuned"],
                  rotation=18, ha="right")
    for xi, f in zip(x, fps):
        ax2.annotate(f"{f:.1f}", (xi, f), textcoords="offset points", xytext=(0, 8),
                     ha="center", color=BLUE, fontsize=8.5)

    ax3 = ax2.twinx()
    ax3.plot(x, char_acc, color=ORANGE, marker="s", lw=2, ls="--", label="Character accuracy, %")
    ax3.set_ylabel("Mean character accuracy, %", color=ORANGE)
    ax3.tick_params(axis="y", colors=ORANGE)
    ax3.set_ylim(30, 95)
    ax3.grid(False)
    ax2.set_title("Speed and character accuracy")
    ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.28))
    ax3.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.45, -0.28))

    fig.tight_layout()
    fig.savefig(OUT / "fig_pipeline_tuning.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Error analysis
# ---------------------------------------------------------------------------
def fig_errors() -> None:
    labels = [
        "Camera angle / perspective",
        "One character wrong",
        "2-3 characters wrong",
        "Fast pass / motion blur",
        "Dirty or unreadable plate",
        "Non-standard plate format",
        "Other",
    ]
    counts = [10, 6, 5, 4, 3, 2, 1]

    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, counts, color=[BLUE] + [GRAY] * (len(labels) - 1), height=0.62)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Clips")
    ax.set_xlim(0, 11.8)
    ax.set_title("Why 31 of 91 clips failed (manual review)")
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.text(6.4, y[0], "32% of all errors", va="center", ha="center", fontsize=8.5, color="white")
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_error_analysis.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Dataset + label audit
# ---------------------------------------------------------------------------
def fig_dataset() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.2), gridspec_kw={"width_ratios": [1, 1.25]})

    sizes = [36_000, 4_800, 4_711]
    labels = ["Train\n36 000", "Val\n4 800", "Test\n4 711"]
    colors = [BLUE, "#90cdf4", "#cbd5e0"]
    wedges, _ = ax1.pie(sizes, colors=colors, startangle=90, counterclock=False,
                        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2})
    ax1.text(0, 0.1, "45 511", ha="center", fontsize=17, fontweight="bold")
    ax1.text(0, -0.18, "plate crops", ha="center", fontsize=9, color=GRAY)
    ax1.legend(wedges, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    ax1.set_title("Dataset: AY000554 Russian plate OCR")

    audit_labels = ["Model errors\n(hard images)", "Typos in labels\n(1-2 chars)", "Fully wrong\nlabels"]
    audit_counts = [400, 180, 120]
    x = np.arange(len(audit_labels))
    bars = ax2.bar(x, audit_counts, 0.5, color=[ORANGE, RED, "#9b2c2c"])
    ax2.set_xticks(x, audit_labels)
    ax2.set_ylabel("Disagreements")
    ax2.set_title("Audit of 758 model/label disagreements")
    ax2.set_ylim(0, 470)
    ax2.bar_label(bars, padding=4, fontsize=9)
    ax2.annotate(
        "300 labels fixed by hand\n-> +1.07 pp accuracy without retraining",
        xy=(1.5, 300), xytext=(0.55, 350), fontsize=8.5, color=GREEN,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1),
    )

    fig.tight_layout()
    fig.savefig(OUT / "fig_dataset_audit.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    fig_ocr()
    fig_yolo()
    fig_tuning()
    fig_errors()
    fig_dataset()
    print("figures written to", OUT)


if __name__ == "__main__":
    main()

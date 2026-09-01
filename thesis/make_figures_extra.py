#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qualitative comparison figures: the system's own overlay beside the
expert's hand-marked photograph of the same plate.

Both panels are unretouched program output; only the frame and the panel
labels are added here.

  python3 thesis/make_figures_extra.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for ttf in sorted(OUT.glob("Tinos-*.ttf")):
    fm.fontManager.addfont(str(ttf))
INSTALLED = {f.name for f in fm.fontManager.ttflist}
SERIF = "Times New Roman" if "Times New Roman" in INSTALLED else "Tinos"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [SERIF, "Liberation Serif", "DejaVu Serif"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
})
INK = "#1a1a1a"


def comparison(tag, marked, out_name, title):
    expert = Image.open(REPO / f"ground_truth/marked_images/{marked}")
    system = Image.open(REPO / f"ground_truth/pipeline_overlays/{tag}_pipeline_halo.png")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2))
    panels = ((expert, "Expert reading, marked on a printed copy"),
              (system, "System output, unretouched"))
    for ax, (im, label) in zip(axes, panels):
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
            sp.set_linewidth(0.9)
        ax.set_xlabel(label, fontsize=9.5, color=INK, labelpad=7)
    fig.suptitle(title, fontsize=11, color=INK, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / out_name, facecolor="white", bbox_inches="tight",
                pad_inches=0.08)
    plt.close(fig)
    print(f"  wrote {out_name}")


if __name__ == "__main__":
    print(f"serif face: {SERIF}")
    comparison("gt_03", "marked_03.jpg", "fig_5_8_best_case.png",
               "Plate 03 — all nine zones found, MAE 1.89 mm")
    comparison("gt_08", "marked_08.jpg", "fig_5_9_worst_case.png",
               "Plate 08 — all nine zones found, MAE 7.13 mm")

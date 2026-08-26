#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remaining thesis figures: the Chapter-3 architecture diagram and the
Chapter-5 qualitative comparisons (system output beside the expert's
hand-marked reading). Print figures, 300 dpi, Persian labels."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

font_manager.fontManager.addfont(str(OUT / "Vazirmatn.ttf"))
plt.rcParams["font.family"] = "Vazirmatn"
plt.rcParams["axes.unicode_minus"] = False

SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
STAGE_FILL, STAGE_EDGE = "#eaf1fb", "#2a78d6"
NOTE_FILL, NOTE_EDGE = "#fdeee7", "#eb6834"

# ══════════════ FIGURE 3-1: pipeline architecture ══════════════
STAGES = [
    ("تصویر ورودی", "یک یا چند ظرف پتری در یک قاب", "io"),
    ("مرحله ۱ — تشخیص ظرف",
     "ماژول ۴ و ۴.۱ · تصحیح روشنایی، Multi-Otsu،\nماسک دقیق هر ظرف", "stage"),
    ("مرحله ۲ — تشخیص دیسک",
     "ماژول ۵ تا ۱۵ · شاخه هاف (اصلی) + شاخه بلاب/واترشد\n(پوششی)، سپس ادغام دوشاخه‌ای و NMS", "stage"),
    ("مرحله ۳ — مرزیابی هاله",
     "ماژول ۱۶ همگرایی دائمی به پس‌زمینه + دو سقف هندسی ·\n۱۶.۵ بسط جهت‌به‌جهت · ۱۶.۶ رفع رخداد زاویه‌ای", "stage"),
    ("مرحله ۴ — رخدادهای درون هاله",
     "ماژول ۱۷ · ماسک حلقه‌ای واقعی + تحلیل لکه", "stage"),
    ("گزارش نهایی", "ماژول ۱۸ · قطر دیسک و هاله بر حسب میلی‌متر", "io"),
]

fig, ax = plt.subplots(figsize=(6.9, 8.6), dpi=300, facecolor=SURFACE)
ax.set_xlim(0, 10); ax.set_ylim(0, len(STAGES) * 2.05)
ax.axis("off"); ax.set_facecolor(SURFACE)

for i, (title, body, kind) in enumerate(STAGES):
    y = (len(STAGES) - 1 - i) * 2.05 + 0.25
    fill, edge = (NOTE_FILL, NOTE_EDGE) if kind == "io" else (STAGE_FILL, STAGE_EDGE)
    ax.add_patch(FancyBboxPatch(
        (0.45, y), 9.1, 1.45, boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=2))
    ax.text(5.0, y + 1.05, title, ha="center", va="center",
            fontsize=11.5, color=INK, zorder=3)
    ax.text(5.0, y + 0.45, body, ha="center", va="center",
            fontsize=8.6, color=INK_2, linespacing=1.5, zorder=3)
    if i < len(STAGES) - 1:
        ax.add_patch(FancyArrowPatch(
            (5.0, y - 0.02), (5.0, y - 0.55), arrowstyle="-|>",
            mutation_scale=15, color=INK_2, linewidth=1.3, zorder=1))

# The "independently per dish" note is carried in the figure caption rather
# than as rotated text: rotating Persian here loses its bidi ordering.

fig.tight_layout()
fig.savefig(OUT / "fig_3_1_architecture.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# ═════════ FIGURES 5-3 / 5-4: system output vs expert marking ═════════
def comparison(tag, marked_name, out_name, caption_left, caption_right):
    sys_img = Image.open(REPO / f"ground_truth/pipeline_overlays/{tag}_pipeline_halo.png")
    exp_img = Image.open(REPO / f"ground_truth/marked_images/{marked_name}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.3), dpi=300, facecolor=SURFACE)
    for ax_, im, cap in zip(axes, (exp_img, sys_img), (caption_right, caption_left)):
        ax_.imshow(im)
        ax_.set_xticks([]); ax_.set_yticks([])
        for sp in ax_.spines.values():
            sp.set_color("#dcdcd8"); sp.set_linewidth(0.8)
        ax_.set_xlabel(cap, fontsize=9.5, color=INK, labelpad=6)
    fig.tight_layout()
    fig.savefig(OUT / out_name, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

# Right panel is drawn first because the figure reads right-to-left.
comparison("gt_03", "marked_03.jpg", "fig_5_3_best_case_gt03.png",
           "خروجی سامانه (ماژول ۱۶.۶)", "اندازه‌گیری کارشناس")
comparison("gt_04", "marked_04.jpg", "fig_5_4_worst_case_gt04.png",
           "خروجی سامانه (ماژول ۱۶.۶)", "اندازه‌گیری کارشناس")

print("wrote:", *(p.name for p in sorted(OUT.glob("fig_*.png"))))

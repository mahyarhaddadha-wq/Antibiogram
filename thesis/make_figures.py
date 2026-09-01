#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build every chart used in the thesis.

All chart text is English in a Times-metric serif face, per the agreed
figure convention. Times New Roman itself is used when the machine has it
(Windows, Office installs); otherwise Tinos, which is metrically identical
to it, is loaded from thesis/figures/.

  python3 thesis/make_figures.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

THESIS = Path(__file__).resolve().parent
FIGS = THESIS / "figures"
ROOT = THESIS.parent
FIGS.mkdir(exist_ok=True)

for ttf in sorted(FIGS.glob("Tinos-*.ttf")):
    fm.fontManager.addfont(str(ttf))
INSTALLED = {f.name for f in fm.fontManager.ttflist}
SERIF = "Times New Roman" if "Times New Roman" in INSTALLED else "Tinos"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [SERIF, "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "custom",
    "mathtext.rm": SERIF,
    "mathtext.it": f"{SERIF}:italic",
    "mathtext.bf": f"{SERIF}:bold",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

INK = "#1a1a1a"
ACCENT = "#0b5394"
WARN = "#b45309"
BAD = "#a02020"
GOOD = "#1d6f42"


def save(fig, name):
    path = FIGS / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.name}")


# ───────────────────────── data ─────────────────────────

def load_results():
    """The single authorised source of every measured number below."""
    rows = list(csv.DictReader(open(ROOT / "ground_truth" / "evaluation_results.csv")))

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    matched = [r for r in rows if num(r["disk_diff_mm"]) is not None]
    both = [r for r in matched
            if num(r["gt_halo_mm"]) is not None and num(r["sys_halo_mm"]) is not None]
    gt = np.array([num(r["gt_halo_mm"]) for r in both])
    sys_ = np.array([num(r["sys_halo_mm"]) for r in both])
    img = np.array([r["image"].replace(".jpg", "") for r in both])
    fp = np.array([num(r["sys_halo_mm"]) for r in matched
                   if num(r["gt_halo_mm"]) is None and num(r["sys_halo_mm"]) is not None])
    return rows, matched, gt, sys_, img, fp


ROWS, MATCHED, GT, SYS, IMG, FP = load_results()
DIFF = SYS - GT
BIAS, SD = DIFF.mean(), DIFF.std(ddof=1)
LO, HI = BIAS - 1.96 * SD, BIAS + 1.96 * SD


# ───────────────────── figure 3-1 ─────────────────────

def fig_architecture():
    fig, ax = plt.subplots(figsize=(7.4, 7.9))
    ax.set_xlim(0, 10)
    ax.set_ylim(2.1, 22.55)
    ax.axis("off")
    ax.grid(False)

    def box(y, h, text, fc, ec, w=7.4, x=1.3, size=10.5):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.18",
            linewidth=1.1, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, color=INK, linespacing=1.5)

    def arrow(y_from, y_to, x=5.0):
        ax.add_patch(FancyArrowPatch(
            (x, y_from), (x, y_to), arrowstyle="-|>", mutation_scale=13,
            linewidth=1.1, color=INK, shrinkA=0, shrinkB=0))

    def stage(y_label, n, title):
        ax.text(0.95, y_label, n, ha="right", va="center", fontsize=14,
                weight="bold", color=ACCENT)
        ax.text(1.30, y_label, title, ha="left", va="center", fontsize=10.5,
                weight="bold", color=ACCENT)

    box(21.3, 1.0, "Plate photograph   (one frame, ordinary room lighting)",
        "#f2f2f2", "#999999")
    arrow(21.3, 20.75)

    stage(20.35, "1", "Locate the petri dish")
    box(18.8, 1.2, "Illumination-field correction  ·  Otsu segmentation\n"
                   "circularity and size screening",
        "#eaf1f8", ACCENT)
    arrow(18.8, 18.25)

    stage(17.85, "2", "Locate the antibiotic disks")
    box(15.7, 1.8, "Disk-specific feature image (white top-hat)\n"
                   "Hough branch  +  blob branch with watershed splitting\n"
                   "score fusion  ·  non-maximum suppression",
        "#eaf1f8", ACCENT)
    arrow(15.7, 15.15)
    box(13.85, 1.3, "Shape verification\n"
                    "arc continuity  ·  radial edge step  ·  rotational symmetry",
        "#dcebdc", GOOD)
    arrow(13.85, 13.3)

    stage(12.9, "3", "Delineate the inhibition zone")
    box(11.35, 1.1, "Agar canvas   ·   far-field lawn reference", "#eaf1f8", ACCENT)

    bus = 10.75
    top = 10.25
    ax.plot([5.0, 5.0], [11.35, bus], linewidth=1.1, color=INK)
    ax.plot([2.45, 7.55], [bus, bus], linewidth=1.1, color=INK)

    trio = [("Radial profile\nbranch", 1.30), ("Threshold region\nbranch", 3.85),
            ("Controlled-growth\nregion branch", 6.40)]
    for text, x in trio:
        cx = x + 1.15
        ax.add_patch(FancyArrowPatch((cx, bus), (cx, top), arrowstyle="-|>",
                                     mutation_scale=11, linewidth=1.1,
                                     color=INK, shrinkA=0, shrinkB=0))
        ax.add_patch(FancyBboxPatch(
            (x, 8.75), 2.30, 1.5, boxstyle="round,pad=0.08,rounding_size=0.16",
            linewidth=1.1, edgecolor=ACCENT, facecolor="#f7fbff"))
        ax.text(cx, 9.5, text, ha="center", va="center", fontsize=9.8,
                color=INK, linespacing=1.5)
        ax.plot([cx, cx], [8.75, 8.25], linewidth=1.1, color=INK)

    ax.plot([2.45, 7.55], [8.25, 8.25], linewidth=1.1, color=INK)
    arrow(8.25, 7.75)
    box(6.55, 1.2, "Division-of-labour fusion\n"
                   "a region branch measures  ·  the radial branch decides presence",
        "#dcebdc", GOOD)
    arrow(6.55, 6.0)

    stage(5.6, "4", "Interpret the reading clinically")
    box(4.05, 1.2, "Pixel-to-millimetre calibration from the 6 mm disks\n"
                   "EUCAST v16.0 breakpoint lookup",
        "#eaf1f8", ACCENT)
    arrow(4.05, 3.5)
    box(2.3, 1.2, "Zone diameter in millimetres\n"
                  "S / I / R category  ·  technical-uncertainty flag",
        "#f2f2f2", "#999999")
    save(fig, "fig_3_1_architecture.png")


# ───────────────────── figure 3-2 ─────────────────────

def fig_branch_tradeoff():
    """Table 6 of EVALUATION.md, drawn as the trade-off it describes."""
    names = ["Radial\nprofile", "Threshold\nregion", "Controlled-growth\nregion"]
    fires = [57 + 14, 26 + 1, 11 + 2]
    mae = [5.49, 3.66, 1.08]
    falsepos = [14, 1, 2]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    x = np.arange(len(names))

    ax = axes[0]
    bars = ax.bar(x, fires, width=0.55, color=["#9ec3e6", "#5b93c7", ACCENT],
                  edgecolor=INK, linewidth=0.7)
    ax.set_xticks(x, names)
    ax.set_ylabel("Disks the branch fires on")
    ax.set_title("Coverage")
    ax.set_ylim(0, 84)
    for b, v, f in zip(bars, fires, falsepos):
        ax.text(b.get_x() + b.get_width() / 2, v + 2.0, f"{v}", ha="center",
                va="bottom", fontsize=10)
        word = "positive" if f == 1 else "positives"
        ax.text(b.get_x() + b.get_width() / 2, v / 2, f"{f} false\n{word}",
                ha="center", va="center", fontsize=9, color="white")

    ax = axes[1]
    bars = ax.bar(x, mae, width=0.55, color=["#f0b9b9", "#d98080", BAD],
                  edgecolor=INK, linewidth=0.7)
    ax.set_xticks(x, names)
    ax.set_ylabel("MAE where the branch fires (mm)")
    ax.set_title("Measurement error")
    ax.set_ylim(0, 6.6)
    for b, v in zip(bars, mae):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center",
                va="bottom", fontsize=10)
    fig.tight_layout()
    save(fig, "fig_5_3_branch_tradeoff.png")


# ───────────────────── figure 5-1 ─────────────────────

def fig_bland_altman():
    mean = (GT + SYS) / 2
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    right = mean.max() + 1.0
    pad = right + 11.5
    ax.set_xlim(mean.min() - 1.2, pad)

    ax.axhspan(-2, 2, xmax=(right - (mean.min() - 1.2)) / (pad - (mean.min() - 1.2)),
               color=GOOD, alpha=0.12, zorder=0)
    for y, c, ls in ((0, "#666666", ":"), (BIAS, ACCENT, "-"), (HI, BAD, "--"),
                     (LO, BAD, "--")):
        ax.plot([mean.min() - 1.2, right], [y, y], color=c,
                linewidth=0.9 if ls == ":" else 1.3, linestyle=ls, zorder=2)
    ax.scatter(mean, DIFF, s=34, facecolor="#7fb0d8", edgecolor=ACCENT,
               linewidth=0.7, zorder=3)

    lx = right + 0.7
    ax.text(lx, HI, f"+1.96 SD\n{HI:+.2f} mm", ha="left", va="center",
            fontsize=9.5, color=BAD, linespacing=1.4)
    ax.text(lx, BIAS + 1.0, f"bias {BIAS:+.2f} mm", ha="left", va="bottom",
            fontsize=9.5, color=ACCENT)
    ax.text(lx, BIAS - 1.0, "expert band\n±2 mm", ha="left", va="top",
            fontsize=9.5, color=GOOD, linespacing=1.4)
    ax.text(lx, LO, f"−1.96 SD\n{LO:+.2f} mm", ha="left", va="center",
            fontsize=9.5, color=BAD, linespacing=1.4)
    ax.axvline(right + 0.35, color="#cccccc", linewidth=0.8)

    ax.set_xlabel("Mean of the two readings (mm)")
    ax.set_ylabel("System − expert (mm)")
    ax.set_title(f"Bland–Altman agreement, n = {len(GT)} zones")
    ax.set_xticks([10, 15, 20, 25, 30, 35])
    save(fig, "fig_5_1_bland_altman.png")


# ───────────────────── figure 5-2 ─────────────────────

def fig_system_vs_expert():
    slope, intercept = np.polyfit(GT, SYS, 1)
    r = np.corrcoef(GT, SYS)[0, 1]
    lim = (4, 38)

    fig, ax = plt.subplots(figsize=(5.9, 5.5))
    ax.plot(lim, lim, color="#666666", linewidth=1.0, linestyle=":",
            label="perfect agreement", zorder=2)
    xs = np.linspace(*lim, 50)
    ax.plot(xs, slope * xs + intercept, color=BAD, linewidth=1.4,
            label=f"least squares, slope {slope:.2f}", zorder=3)
    ax.fill_between(xs, xs - 2, xs + 2, color=GOOD, alpha=0.10, zorder=1,
                    label="within ±2 mm of the expert")
    ax.scatter(GT, SYS, s=34, facecolor="#7fb0d8", edgecolor=ACCENT,
               linewidth=0.7, zorder=4)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Expert reading (mm)")
    ax.set_ylabel("System measurement (mm)")
    ax.set_title(f"Zone diameter, n = {len(GT)}   (r = {r:.2f})")
    ax.legend(loc="upper left", framealpha=0.95)
    save(fig, "fig_5_2_system_vs_expert.png")


# ───────────────────── figure 5-3 ─────────────────────

def fig_error_vs_size():
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    edges = [(9.5, 16), (16, 20), (20, 25), (25, 36)]
    ax.set_xlim(9.0, 36.4)
    ax.set_ylim(-23.5, 17.0)
    ax.axhspan(-2, 2, color=GOOD, alpha=0.12, zorder=0)
    for k, (lo, hi) in enumerate(edges):
        if k % 2:
            ax.axvspan(lo, hi, color="#f0f0f0", zorder=-1)
    ax.axhline(0, color="#666666", linewidth=0.9, linestyle=":", zorder=1)
    ax.scatter(GT, DIFF, s=34, facecolor="#7fb0d8", edgecolor=ACCENT,
               linewidth=0.7, zorder=3)

    ax.axhline(12.2, color="#bbbbbb", linewidth=0.8)
    for lo, hi in edges:
        m = (GT >= lo) & (GT < hi)
        if not m.any():
            continue
        mid = (lo + hi) / 2
        ax.plot([lo + 0.3, hi - 0.3], [DIFF[m].mean()] * 2, color=BAD,
                linewidth=2.6, solid_capstyle="butt", zorder=4)
        ax.text(mid, 15.4, f"{DIFF[m].mean():+.1f} mm", ha="center", va="center",
                fontsize=10, color=BAD)
        ax.text(mid, 13.3, f"n = {int(m.sum())}", ha="center", va="center",
                fontsize=9.5, color=INK)
    ax.text(9.35, 15.4, "mean", ha="left", va="center", fontsize=9.5,
            color="#666666")
    ax.plot([], [], color=BAD, linewidth=2.6, label="group mean error")
    ax.set_xlabel("Expert reading (mm)")
    ax.set_ylabel("System − expert (mm)")
    ax.set_title("The error turns negative as the zone grows")
    ax.legend(loc="lower left", framealpha=0.96)
    save(fig, "fig_5_5_error_vs_size.png")


# ───────────────────── figure 5-4 ─────────────────────

def fig_error_distribution():
    a = np.abs(DIFF)
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    ax.hist(a, bins=np.arange(0, 23, 1.0), color="#9ec3e6", edgecolor=INK,
            linewidth=0.7, zorder=2)
    ax.axvspan(0, 2, color=GOOD, alpha=0.12, zorder=0)
    ax.axvline(2, color=GOOD, linewidth=1.6, zorder=3,
               label=f"expert band, 2 mm — {(a <= 2).mean() * 100:.0f}% of zones inside")
    ax.axvline(np.median(a), color=ACCENT, linewidth=1.6, linestyle="--", zorder=3,
               label=f"median error  {np.median(a):.2f} mm")
    ax.axvline(a.mean(), color=BAD, linewidth=1.6, linestyle="-.", zorder=3,
               label=f"mean error  {a.mean():.2f} mm")
    ax.set_xlabel("Absolute error (mm)")
    ax.set_ylabel("Number of zones")
    ax.set_xlim(0, 22.5)
    ax.set_title("A tight core with a heavy tail")
    ax.legend(loc="upper right", framealpha=0.96)
    save(fig, "fig_5_4_error_distribution.png")


# ───────────────────── figure 5-5 ─────────────────────

def fig_per_image():
    names = sorted(set(IMG))
    mae = [np.abs(DIFF[IMG == n]).mean() for n in names]
    cnt = [int((IMG == n).sum()) for n in names]
    order = np.argsort(mae)
    names = [names[i].replace("gt_", "plate ") for i in order]
    mae = [mae[i] for i in order]
    cnt = [cnt[i] for i in order]

    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    bars = ax.bar(np.arange(len(names)), mae, width=0.62,
                  color=["#9ec3e6" if v <= 3 else "#e2a8a8" for v in mae],
                  edgecolor=INK, linewidth=0.7)
    ax.axhline(np.abs(DIFF).mean(), color=BAD, linewidth=1.2, linestyle="--")
    ax.text(len(names) - 0.4, np.abs(DIFF).mean() + 0.18,
            f"overall MAE {np.abs(DIFF).mean():.2f} mm", ha="right", va="bottom",
            fontsize=9.5, color=BAD)
    ax.axhline(2, color=GOOD, linewidth=1.2)
    ax.text(-0.4, 2.18, "expert band 2 mm", ha="left", va="bottom", fontsize=9.5,
            color=GOOD)
    ax.set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
    ax.set_ylabel("MAE on that plate (mm)")
    ax.set_ylim(0, 12.2)
    ax.set_title("Plate-by-plate error: the difficulty is not evenly spread")
    for b, v, c in zip(bars, mae, cnt):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.16, f"n={c}", ha="center",
                va="bottom", fontsize=9)
    save(fig, "fig_5_7_per_image.png")


# ───────────────────── figure 5-6 ─────────────────────

def fig_false_positives():
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.set_xlim(4.0, 35.0)
    ax.set_ylim(0.12, 2.35)

    ax.axvline(6.0, color=INK, linewidth=1.5, zorder=2)
    ax.text(6.35, 2.24, "6 mm — the disk itself, where a borderline\n"
                        "false report would have to sit",
            fontsize=9.5, color=INK, va="top", linespacing=1.4)

    ax.scatter(GT, 1.72 + rng.uniform(-0.10, 0.10, GT.size), s=42,
               facecolor="#9ec3e6", edgecolor=ACCENT, linewidth=0.7, zorder=3)
    ax.scatter(FP, 0.72 + rng.uniform(-0.10, 0.10, FP.size), s=42,
               facecolor="#e2a8a8", edgecolor=BAD, linewidth=0.7, zorder=3)

    ax.plot([np.median(GT)] * 2, [1.42, 1.56], color=ACCENT, linewidth=1.8,
            zorder=4)
    ax.plot([np.median(FP)] * 2, [0.42, 0.56], color=BAD, linewidth=1.8,
            zorder=4)

    ax.set_yticks([1.72, 0.72],
                  [f"zones the expert\nconfirmed  (n = {GT.size})",
                   f"false zone reports\n(n = {FP.size})"])
    ax.text(np.median(GT) + 0.45, 1.38, f"median {np.median(GT):.1f} mm",
            fontsize=9.5, color=ACCENT, va="top")
    ax.text(np.median(FP) + 0.45, 0.38, f"median {np.median(FP):.1f} mm",
            fontsize=9.5, color=BAD, va="top")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Zone diameter (mm)")
    ax.set_title("The false zone reports are not borderline cases")
    save(fig, "fig_5_6_false_positives.png")


# ───────────────────── figure 5-7 ─────────────────────

def fig_clinical_target():
    """Engineering-target table of ground_truth/eucast/categorical_agreement_run.txt."""
    mae = [3.85, 2.89, 1.93, 1.35, 0.96, 0.58, 0.39]
    ca = [83.2, 86.8, 91.5, 94.2, 96.1, 97.5, 98.1]
    vme = [6.19, 4.09, 2.36, 1.19, 0.50, 0.12, 0.07]
    me = [9.24, 7.78, 5.11, 3.78, 2.80, 1.96, 1.51]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))

    ax = axes[0]
    ax.set_xlim(4.25, 0.15)
    ax.set_ylim(81.5, 99.5)
    ax.axhspan(92, 96, color=GOOD, alpha=0.13, zorder=0)
    ax.plot(mae, ca, marker="o", color=ACCENT, linewidth=1.6, markersize=5,
            zorder=3)
    ax.text(4.10, 94.0, "expert-versus-expert ceiling", fontsize=9.5,
            color=GOOD, va="center", ha="left")
    ax.scatter([3.85], [83.2], s=95, facecolor="white", edgecolor=BAD,
               linewidth=1.7, zorder=5)
    ax.text(3.60, 83.2, "  system today", fontsize=9.5, color=BAD, va="center",
            ha="left")
    ax.set_xlabel("MAE (mm)")
    ax.set_ylabel("Categorical agreement (%)")
    ax.set_title("Agreement against measurement error")

    ax = axes[1]
    ax.set_xlim(4.25, 0.15)
    ax.set_ylim(-0.7, 11.4)
    ax.axhline(1.5, color=BAD, linewidth=1.0, linestyle=":", zorder=1)
    ax.axhline(3.0, color=WARN, linewidth=1.0, linestyle=":", zorder=1)
    ax.text(4.15, 1.72, "1.5% limit", fontsize=9, color=BAD, ha="left")
    ax.text(4.15, 3.22, "3.0% limit", fontsize=9, color=WARN, ha="left")
    ax.plot(mae, vme, marker="o", color=BAD, linewidth=1.6, markersize=5,
            label="very major error", zorder=3)
    ax.plot(mae, me, marker="s", color=WARN, linewidth=1.6, markersize=5,
            label="major error", zorder=3)
    ax.axvline(0.96, color=INK, linewidth=1.1, linestyle="--", zorder=2)
    ax.text(0.88, 6.4, "both limits met\nbelow 0.96 mm", fontsize=9.5,
            color=INK, ha="left", va="center", linespacing=1.4)
    ax.set_xlabel("MAE (mm)")
    ax.set_ylabel("Error rate (%)")
    ax.set_title("Clinical error limits")
    ax.legend(loc="lower left", framealpha=0.96)
    fig.tight_layout()
    save(fig, "fig_5_10_clinical_target.png")


# ───────────────────── figure 6-1 ─────────────────────

def fig_separability():
    """Measured contrast between zone interior and lawn (ROADMAP, p2 probe)."""
    labels = ["Reference taken from\nrings next to the disk",
              "Reference taken inside the\nexpert-marked zone"]
    critical = [0.37, 0.29]
    easy = [0.85, 0.67]
    x = np.arange(len(labels))
    w = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    b1 = ax.bar(x - w / 2, critical, w, color="#e2a8a8", edgecolor=INK,
                linewidth=0.7, label="33 zones left to the radial branch")
    b2 = ax.bar(x + w / 2, easy, w, color="#9ec3e6", edgecolor=INK,
                linewidth=0.7, label="27 zones a region branch solved")
    ax.axhline(2.0, color=GOOD, linewidth=1.4)
    ax.text(-0.47, 2.05, "contrast the region branches need", ha="left",
            va="bottom", fontsize=9.5, color=GOOD)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.52, 1.52)
    ax.set_ylim(0, 2.28)
    ax.set_ylabel("Zone-to-lawn separation\n(standard deviations of the lawn)")
    ax.set_title("On the hard half of the disks the boundary is not in the image")
    ax.legend(loc="center", bbox_to_anchor=(0.5, 0.63), framealpha=0.96)
    save(fig, "fig_6_1_separability.png")


# ───────────────────── figure 6-2 ─────────────────────

def fig_filter_study():
    """Spatial-averaging study of ROADMAP section 7.1."""
    n = np.array([1, 9, 25, 81, 361, 1225])
    measured = np.array([0.36, 0.42, 0.47, 0.54, 0.59, 0.71])
    ideal = np.array([0.36, 1.08, 1.80, 3.24, 6.84, 12.60])
    blur = np.array([0.0, 0.23, 0.38, 0.68, 1.43, 2.63])

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

    ax = axes[0]
    ax.plot(n, ideal, marker="s", color="#999999", linewidth=1.4, markersize=5,
            linestyle="--", label="if the lawn noise were independent")
    ax.plot(n, measured, marker="o", color=ACCENT, linewidth=1.7, markersize=5,
            label="measured")
    ax.axhline(2.0, color=GOOD, linewidth=1.2)
    ax.text(1.15, 2.12, "contrast needed", fontsize=9.5, color=GOOD)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
    ax.set_yticks([0.5, 1, 2, 5, 10], ["0.5", "1", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_xlabel("Pixels averaged")
    ax.set_ylabel("Zone-to-lawn separation (SD)")
    ax.set_title("Averaging buys far less than noise theory promises")
    ax.legend(loc="upper left", framealpha=0.95)

    ax = axes[1]
    ax.set_xlim(0.05, 3.05)
    ax.set_ylim(0.395, 0.745)
    ax.plot(blur[1:], measured[1:], marker="o", color=ACCENT, linewidth=1.7,
            markersize=5)
    for b, m, k in zip(blur[1:], measured[1:], n[1:]):
        dx, dy = (-6, -11) if k == 1225 else (7, -8)
        ha = "right" if k == 1225 else "left"
        ax.annotate(f"{k}", (b, m), textcoords="offset points", xytext=(dx, dy),
                    fontsize=9, color=INK, ha=ha)
    ax.axvline(2.0, color=BAD, linewidth=1.2, linestyle="--")
    ax.text(1.93, 0.415, "expert band 2 mm", rotation=90, ha="right", va="bottom",
            fontsize=9.5, color=BAD)
    ax.set_xlabel("Boundary blur the filter introduces (mm)")
    ax.set_ylabel("Zone-to-lawn separation (SD)")
    ax.set_title("and it blurs the boundary it was meant to find")
    fig.tight_layout()
    save(fig, "fig_6_2_filter_study.png")


# ───────────────────── figure 6-3 ─────────────────────

def fig_sigmoid_projection():
    """Paired offline study of the logistic model (ROADMAP, p4 feasibility)."""
    labels = ["Deployed fusion", "Logistic model\n(inflection point)",
              "Best of the two,\nper disk"]
    mae = [3.78, 2.79, 1.37]
    med = [2.10, 1.03, None]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))

    ax = axes[0]
    x = np.arange(3)
    bars = ax.bar(x, mae, width=0.55, color=["#e2a8a8", "#9ec3e6", "#cfe3d3"],
                  edgecolor=INK, linewidth=0.7)
    ax.axhline(2.0, color=GOOD, linewidth=1.3)
    ax.text(2.42, 2.12, "expert band 2 mm", ha="right", va="bottom",
            fontsize=9.5, color=GOOD)
    for b, v in zip(bars, mae):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.10, f"{v:.2f}", ha="center",
                va="bottom", fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 4.6)
    ax.set_ylabel("MAE on the 57 paired disks (mm)")
    ax.set_title("Measured on the same disks, offline")

    ax = axes[1]
    x2 = np.arange(2)
    bars = ax.bar(x2, med[:2], width=0.45, color=["#e2a8a8", "#9ec3e6"],
                  edgecolor=INK, linewidth=0.7)
    ax.axhline(2.0, color=GOOD, linewidth=1.3)
    ax.text(1.32, 2.06, "expert band 2 mm", ha="right", va="bottom",
            fontsize=9.5, color=GOOD)
    for b, v in zip(bars, med[:2]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.06, f"{v:.2f}", ha="center",
                va="bottom", fontsize=10)
    ax.set_xticks(x2, labels[:2])
    ax.set_xlim(-0.6, 1.45)
    ax.set_ylim(0, 2.9)
    ax.set_ylabel("Median error (mm)")
    ax.set_title("The typical case halves")
    fig.tight_layout()
    save(fig, "fig_6_3_sigmoid_projection.png")


if __name__ == "__main__":
    print(f"serif face: {SERIF}")
    fig_architecture()
    fig_branch_tradeoff()
    fig_bland_altman()
    fig_system_vs_expert()
    fig_error_vs_size()
    fig_error_distribution()
    fig_per_image()
    fig_false_positives()
    fig_clinical_target()
    fig_separability()
    fig_filter_study()
    fig_sigmoid_projection()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chapter-5 figures: Bland-Altman and system-vs-expert scatter.
Print figures (300 dpi PNG) for the Word thesis. Persian labels."""
import csv, statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

SCRATCH = Path(__file__).resolve().parent / "figures"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "thesis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Persian text shaping ----------------------------------------------
font_manager.fontManager.addfont(str(SCRATCH / "Vazirmatn.ttf"))
FA = "Vazirmatn"
plt.rcParams["font.family"] = FA
plt.rcParams["axes.unicode_minus"] = False

def fa(s):
    """This matplotlib build shapes and bidi-orders Arabic script natively,
    so Persian strings are passed through unchanged. Applying an external
    reshaper/bidi pass here would double-process and reverse the text."""
    return s

# Persian-Indic digits, so figure numerals match the thesis body text.
_DIG = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
def fnum(x, dec=0, sign=False):
    """Persian-Indic digits wrapped in an LTR isolate, so a sign or decimal
    point keeps its correct side when embedded in right-to-left text."""
    s = f"{x:+.{dec}f}" if sign else f"{x:.{dec}f}"
    return "\u2066" + s.translate(_DIG).replace("-", "\u2212") + "\u2069"

# ---- Palette (validated: dataviz reference, slots 1-2 + neutrals) -------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
GRID      = "#dcdcd8"
SERIES_1  = "#2a78d6"   # data points
SERIES_2  = "#eb6834"   # reference lines

# ---- Data ---------------------------------------------------------------
rows = list(csv.DictReader(open(REPO / "ground_truth/evaluation_results.csv", encoding="utf-8")))
def num(v): return float(v) if v not in (None, "") else None

pairs = []
for r in rows:
    if r["match_dist_px"] in (None, ""):
        continue
    g, s = num(r["gt_halo_mm"]), num(r["sys_halo_mm"])
    if g is not None and s is not None:
        pairs.append((g, s))

gt   = [p[0] for p in pairs]
sysv = [p[1] for p in pairs]
diff = [s - g for g, s in pairs]
mean = [(s + g) / 2 for g, s in pairs]

bias = st.mean(diff)
sd   = st.stdev(diff)
loa_hi, loa_lo = bias + 1.96 * sd, bias - 1.96 * sd
n = len(pairs)

def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)

# ======================= FIGURE 1: Bland-Altman ==========================
fig, ax = plt.subplots(figsize=(6.6, 4.4), dpi=300, facecolor=SURFACE)
style(ax)

ax.scatter(mean, diff, s=34, facecolor=SERIES_1, edgecolor=SURFACE,
           linewidth=0.8, alpha=0.85, zorder=3)

for y, dash, lbl in [
    (loa_hi, (5, 3), f"حد بالای توافق {fnum(loa_hi,2,sign=True)}"),
    (bias,   None,   f"سوگیری {fnum(bias,2,sign=True)}"),
    (loa_lo, (5, 3), f"حد پایین توافق {fnum(loa_lo,2,sign=True)}"),
]:
    ax.axhline(y, color=SERIES_2, linewidth=1.6 if dash is None else 1.3,
               linestyle="-" if dash is None else (0, dash), zorder=2)
    ax.text(0.995, y, fa(lbl), transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8.5, color=SERIES_2)

ax.axhline(0, color=INK_2, linewidth=0.9, linestyle=(0, (1, 3)), zorder=1)

ax.set_xlabel(fa("میانگین دو اندازه‌گیری (میلی‌متر)"), fontsize=10, color=INK, labelpad=7)
ax.set_ylabel(fa("سامانه − کارشناس (میلی‌متر)"), fontsize=10, color=INK, labelpad=7)
ax.set_title(fa(f"نمودار بلاند–آلتمن قطر هاله  (n = {fnum(n)})"),
             fontsize=11.5, color=INK, pad=11)
ax.set_xticks(ax.get_xticks()); ax.set_xticklabels([fnum(t) for t in ax.get_xticks()])
ax.set_yticks(ax.get_yticks()); ax.set_yticklabels([fnum(t).replace("-", "−") for t in ax.get_yticks()])
fig.tight_layout()
fig.savefig(OUT / "fig_5_1_bland_altman.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# ============ FIGURE 2: system vs expert, identity + regression ==========
mx, my = st.mean(gt), st.mean(sysv)
sxy = sum((a - mx) * (b - my) for a, b in zip(gt, sysv))
sxx = sum((a - mx) ** 2 for a in gt)
slope = sxy / sxx
inter = my - slope * mx
syy = sum((b - my) ** 2 for b in sysv)
r2 = sxy ** 2 / (sxx * syy)

fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=300, facecolor=SURFACE)
style(ax)

lim = (0, max(max(gt), max(sysv)) * 1.08)
ax.plot(lim, lim, color=INK_2, linewidth=1.2, linestyle=(0, (5, 3)), zorder=2)
ax.text(lim[1] * 0.97, lim[1] * 0.97, fa("توافق کامل"), fontsize=8.5,
        color=INK_2, ha="right", va="bottom", rotation=45,
        rotation_mode="anchor")

xs = [lim[0], lim[1]]
ax.plot(xs, [slope * x + inter for x in xs], color=SERIES_2,
        linewidth=1.8, zorder=2)
ax.text(lim[1] * 0.98, slope * lim[1] * 0.98 + inter - 1.6,
        fa(f"برازش خطی: شیب {fnum(slope,2)}،  r = {fnum(r2**0.5,2)}"), fontsize=8.5,
        color=SERIES_2, ha="right", va="top")

ax.scatter(gt, sysv, s=34, facecolor=SERIES_1, edgecolor=SURFACE,
           linewidth=0.8, alpha=0.85, zorder=3)

ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel(fa("اندازه‌گیری کارشناس (میلی‌متر)"), fontsize=10, color=INK, labelpad=7)
ax.set_ylabel(fa("گزارش سامانه (میلی‌متر)"), fontsize=10, color=INK, labelpad=7)
ax.set_title(fa(f"سامانه در برابر کارشناس  (n = {fnum(n)})"),
             fontsize=11.5, color=INK, pad=11)
ax.set_xticks(ax.get_xticks()); ax.set_xticklabels([fnum(t) for t in ax.get_xticks()])
ax.set_yticks(ax.get_yticks()); ax.set_yticklabels([fnum(t) for t in ax.get_yticks()])
fig.tight_layout()
fig.savefig(OUT / "fig_5_2_system_vs_expert.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print(f"n={n}  bias={bias:.2f}  sd={sd:.2f}  LoA=[{loa_lo:.2f}, {loa_hi:.2f}]")
print(f"regression: sys = {slope:.3f}*gt + {inter:.2f}   R2={r2:.3f}")
print("wrote:", *(p.name for p in sorted(OUT.glob("*.png"))))

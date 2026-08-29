"""
پ۱+پ۳ گامِ ۱ — جاروبِ آستانه‌ی Hough رویِ هر ۱۱ عکس.

پرسش: با پایین آوردنِ `disk_hough_param2`، چند دیسکِ ازدست‌رفته بازیافته می‌شود و
به ازای آن چند کاندیدِ کاذب اضافه می‌شود؟

**نکته‌ی روشی که در نسخه‌ی اولِ این آزمون اشتباه کردم:** تطبیق فقط با *مکان* کافی
نیست. Hough گاهی رویِ خودِ **هاله** قفل می‌شود و در همان مرکز دایره‌ای با شعاعِ
دوبرابر می‌دهد (رویِ gt_10 دیسکِ ۸: شعاعِ ۹۰ در برابرِ ۳۷–۴۳ برایِ دیسک‌هایِ واقعی).
چنین دایره‌ای «بازیافتِ دیسک» نیست و قاعده‌ی سازگاریِ شعاعِ ماژولِ ۱۴ هم به‌درستی
ردش می‌کند. پس تطبیق باید هم مکان و هم شعاع را بسنجد، وگرنه نتیجه گمراه‌کننده است.

خروجی: ground_truth/diagnostics/p13_hough_sweep.csv
"""
import copy
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "disk_detection_pipeline_modular.ipynb"
RAW = REPO / "ground_truth" / "raw_images"
GT_CSV = REPO / "ground_truth" / "ground_truth_expert_readings.csv"
OUT = REPO / "ground_truth" / "diagnostics" / "p13_hough_sweep.csv"

CFG_MARKER = "cfg.halo_fusion_otsu_percentile"
FUSION_MARKER = "ماژول ۱۴ (بازطراحی)"
PARAM2_GRID = [20, 16, 13, 11, 9]
R_TOL = 0.30            # رواداریِ شعاع نسبت به میانه‌ی دیسک‌هایِ واقعیِ همان عکس

DEBUG = r'''
_grid = %(grid)s
for _d in dishes:
    _ox, _oy = _d["roi_offset_xy"]
    for _p2 in _grid:
        _old = cfg.disk_hough_param2
        cfg.disk_hough_param2 = _p2
        try:
            _c = run_disk_hough(_d["disk_feature_image"], _d["disk_edge_image"],
                                _d["processing_mask_roi"], _d["center_roi_xy"],
                                _d["radius_px"], cfg)
        except Exception as _e:
            print("ERR|%%d|%%s" %% (_p2, _e)); cfg.disk_hough_param2 = _old; continue
        cfg.disk_hough_param2 = _old
        for _k in _c:
            print("H|%%d|%%.1f|%%.1f|%%.1f" %% (_p2, _k["x"]+_ox, _k["y"]+_oy, _k["r"]))
    for _k in _d["final_candidates"]:
        print("F|%%.1f|%%.1f|%%.1f" %% (_k["x"]+_ox, _k["y"]+_oy, _k["r"]))
'''

H_RE = re.compile(r"H\|(\d+)\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)")
F_RE = re.compile(r"F\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)")


def build(base, img):
    nb = copy.deepcopy(base)
    cells = list(nb["cells"])
    i = next(k for k, c in enumerate(cells) if CFG_MARKER in "".join(c["source"]))
    cells.insert(i + 1, nbformat.v4.new_code_cell(f'cfg.image_path = r"{img}"'))
    j = next(k for k, c in enumerate(cells) if FUSION_MARKER in "".join(c["source"]))
    nb["cells"] = cells[: j + 1] + [
        nbformat.v4.new_code_cell(DEBUG % {"grid": PARAM2_GRID})]
    return nb


def main():
    base = nbformat.read(NB, as_version=4)
    gt = defaultdict(list)
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            gt[row["image_file"]].append(
                {"num": int(row["disk_number"]), "x": float(row["x_px"]),
                 "y": float(row["y_px"])})

    rows = []
    for img in sorted(RAW.glob("gt_*.jpg")):
        nb = build(base, img)
        NotebookClient(nb, kernel_name="antibiogram-test", timeout=2400).execute()
        text = "".join(o.get("text", "") for c in nb["cells"]
                       for o in c.get("outputs", []) if o.get("output_type") == "stream")

        gd = gt[img.name]
        rad = 0.5 * min((math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                         for i, a in enumerate(gd) for b in gd[i + 1:]), default=200.0)
        fin = [(float(a), float(b), float(c)) for a, b, c in F_RE.findall(text)]
        r_ref = sorted(c for _, _, c in fin)[len(fin) // 2] if fin else 0.0

        by = defaultdict(list)
        for p2, x, y, r in H_RE.findall(text):
            by[int(p2)].append((float(x), float(y), float(r)))

        cur_fn = [g for g in gd if not any(
            math.hypot(fx - g["x"], fy - g["y"]) <= rad for fx, fy, _ in fin)]

        for p2 in PARAM2_GRID:
            pts = by.get(p2, [])
            hit, extra = 0, 0
            for px, py, pr in pts:
                g = next((g for g in gd
                          if math.hypot(px - g["x"], py - g["y"]) <= rad), None)
                ok_r = bool(r_ref) and abs(pr - r_ref) / r_ref <= R_TOL
                if g and ok_r:
                    hit += 1
                elif not g:
                    extra += 1
            rec = sum(1 for g in cur_fn
                      if any(math.hypot(px - g["x"], py - g["y"]) <= rad
                             and r_ref and abs(pr - r_ref) / r_ref <= R_TOL
                             for px, py, pr in pts))
            rows.append({"image": img.name, "param2": p2, "n_gt": len(gd),
                         "hough_total": len(pts), "hough_on_gt": hit,
                         "hough_extra": extra, "current_fn": len(cur_fn),
                         "fn_recovered": rec})
        print(f"[done] {img.name}", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 70)
    print("جاروبِ آستانه‌ی Hough رویِ هر ۱۱ عکس")
    print("=" * 70)
    print(f"  {'param2':>7}{'پوششِ دیسکِ واقعی':>18}{'کاندیدِ کاذبِ خام':>18}"
          f"{'FN بازیافته':>14}")
    n_gt = sum(r["n_gt"] for r in rows if r["param2"] == PARAM2_GRID[0])
    fn0 = sum(r["current_fn"] for r in rows if r["param2"] == PARAM2_GRID[0])
    for p2 in PARAM2_GRID:
        sub = [r for r in rows if r["param2"] == p2]
        print(f"  {p2:>7}{sum(r['hough_on_gt'] for r in sub):>10}/{n_gt:<7}"
              f"{sum(r['hough_extra'] for r in sub):>18}"
              f"{sum(r['fn_recovered'] for r in sub):>10}/{fn0}")
    print(f"\nجزئیات: {OUT}")


if __name__ == "__main__":
    main()

"""
آیا با فیلتر رویِ تصویرِ ورودی می‌توان جدایشِ (σ) هاله از لَون را بهتر کرد؟

اندازه‌گیریِ قبلی نشان داد رویِ نیمی از دیسک‌ها شدتِ متوسطِ داخلِ هاله با لَون فقط
۰.۳σ فرق دارد. این اسکریپت می‌سنجد که آیا این سقف با پردازشِ تصویر شکستنی است.

## تفکیکِ نظری که مسیرِ آزمون را تعیین می‌کند

**تبدیلِ نقطه‌ای** (گاما، لگاریتم، کوبلکا–مونک، هر تابعِ پیکسل‌به‌پیکسل): اگر یکنوا و
وارون‌پذیر باشد، ترتیبِ پیکسل‌ها را حفظ می‌کند، پس **AUC را دقیقاً تغییر نمی‌دهد**.
می‌تواند عددِ «اندازه‌اثر» را باد کند بدونِ این‌که یک پیکسل واقعاً تفکیک‌پذیرتر شود.
این‌جا تجربی هم بررسی می‌شود، نه فقط ادعا.

**فیلترِ فضایی** (میانگین رویِ همسایگی): نقطه‌ای نیست، پس *می‌تواند* کمک کند. اگر
نویزِ بینِ پیکسل‌ها **مستقل** باشد، میانگین‌گیری رویِ N پیکسل نویز را √N برابر کم
می‌کند؛ برایِ رفتن از ۰.۳σ به ۲σ فقط N≈۴۴ پیکسل لازم است.

**ولی این فقط با نویزِ مستقل درست است.** اگر آن‌چه «نویز» می‌نامیم ساختارِ کم‌بسامد
باشد (شیبِ نور، بافتِ خودِ لَون، ناهمواریِ آگار)، میانگین‌گیری حذفش نمی‌کند و
اندازه‌اثر **اشباع** می‌شود.

پس پرسشِ تعیین‌کننده: **نویزِ لَون مستقل است یا ساختاریافته؟**

    اگر اندازه‌اثر با √N بالا برود -> نویز مستقل -> فیلتر جواب می‌دهد
    اگر زود اشباع شود              -> نویز ساختاریافته -> فیلتر جواب نمی‌دهد

خروجی: ground_truth/diagnostics/sigma_filter_study.csv
"""
import copy
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import nbformat
import numpy as np
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "disk_detection_pipeline_modular.ipynb"
RAW = REPO / "ground_truth" / "raw_images"
GT_CSV = REPO / "ground_truth" / "ground_truth_expert_readings.csv"
OUT = REPO / "ground_truth" / "diagnostics" / "sigma_filter_study.csv"

CFG_MARKER = "cfg.halo_fusion_otsu_percentile"
CANVAS_MARKER = "ماژول ۱۵.۵ (جدید)"

# اندازه‌ی کرنل به‌صورتِ کسری از شعاعِ دیسک (نسبت‌محور، مثلِ کلِ پروژه).
SCALES = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80]

DEBUG = r'''
import numpy as _np, cv2 as _cv2

_GT = %(gt)s          # [(x, y, zone_diameter_mm), ...] در مختصاتِ تصویرِ کامل
_SC = %(scales)s

def _auc(a, b):
    """سطحِ زیرِ ROC از رویِ رتبه‌ها (من-ویتنی) -- بدونِ scipy."""
    n1, n2 = len(a), len(b)
    if n1 < 20 or n2 < 20:
        return float("nan")
    allv = _np.concatenate([a, b])
    order = _np.argsort(allv, kind="mergesort")
    ranks = _np.empty(len(allv), dtype=_np.float64)
    ranks[order] = _np.arange(1, len(allv) + 1, dtype=_np.float64)
    r1 = float(_np.sum(ranks[:n1]))
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n2)

for _d in dishes:
    _ox, _oy = _d["roi_offset_xy"]
    _canvas = _d["agar_canvas"].astype(_np.float32)
    _agar = _d["agar_mask"] > 0
    # px_per_mm در ماژولِ ۱۶ محاسبه می‌شود که بعد از نقطه‌ی توقفِ ماست، پس همان
    # قاعده این‌جا درجا بازسازی می‌شود: خوشه‌ی کوچک‌ترین قطرها (تا ۳۰٪ بالاتر از
    # کمینه)، میانگین، تقسیم بر قطرِ اسمیِ ۶ میلی‌متر.
    _ds = [2.0 * float(_c["r"]) for _c in _d["final_candidates"]]
    _ppm = (float(_np.mean([_x for _x in _ds if _x <= min(_ds) * 1.30])) / 6.0) if _ds else 0.0
    if not _ppm or _canvas.size == 0:
        continue
    _h, _w = _canvas.shape[:2]
    _yy, _xx = _np.mgrid[0:_h, 0:_w]

    # مرجعِ میدانِ دور: آگارِ دورترین ۲۰٪ از هر دیسک -- نمی‌تواند داخلِ هیچ هاله‌ای باشد
    _dist = _np.full((_h, _w), 1e9, dtype=_np.float32)
    for _c in _d["final_candidates"]:
        _dist = _np.minimum(_dist, _np.hypot(_xx - _c["x"], _yy - _c["y"]).astype(_np.float32))
    _far = _agar & (_dist >= _np.percentile(_dist[_agar], 80))

    for _i, _c in enumerate(_d["final_candidates"], start=1):
        _cx, _cy, _rd = float(_c["x"]), float(_c["y"]), float(_c["r"])
        _gx, _gy = _cx + _ox, _cy + _oy
        # تطبیق با نزدیک‌ترین دیسکِ مرجع که هاله‌ی ثبت‌شده دارد
        _best, _bd = None, 1e18
        for _g in _GT:
            _dd = _np.hypot(_gx - _g[0], _gy - _g[1])
            if _dd < _bd:
                _bd, _best = _dd, _g
        if _best is None or _bd > _rd * 2.0 or _best[2] <= 0:
            continue
        _rzone = 0.5 * _best[2] * _ppm          # شعاعِ هاله‌ی مرجع بر حسبِ پیکسل
        if _rzone <= 1.4 * _rd:
            continue
        _rr = _np.hypot(_xx - _cx, _yy - _cy)
        # داخلِ هاله: از بیرونِ ماسکِ دیسک تا ۷۰٪ شعاعِ هاله‌ی مرجع
        _inzone = _agar & (_rr <= 0.70 * _rzone)
        if int(_np.count_nonzero(_inzone)) < 200:
            continue

        for _s in _SC:
            if _s <= 0:
                _img = _canvas
            else:
                _k = max(3, (int(round(_s * _rd)) | 1))
                _img = _cv2.blur(_canvas, (_k, _k))
            _a = _img[_inzone].astype(_np.float64)
            _b = _img[_far].astype(_np.float64)
            _sd = float(_np.std(_b, ddof=1))
            _eff = abs(float(_np.mean(_b)) - float(_np.mean(_a))) / _sd if _sd > 1e-9 else 0.0
            # نمونه‌برداری برایِ AUC تا هزینه‌ی مرتب‌سازی کنترل شود (گامِ ثابت = قطعی)
            _as = _a[:: max(1, len(_a) // 4000)]
            _bs = _b[:: max(1, len(_b) // 4000)]
            _n_eff = (max(3, (int(round(_s * _rd)) | 1)) ** 2) if _s > 0 else 1
            print("S|%%.1f|%%.1f|%%.2f|%%d|%%.4f|%%.4f|%%.4f" %% (
                _gx, _gy, _s, _n_eff, _eff, _auc(_as, _bs), _sd))
'''

ROW_RE = re.compile(
    r"S\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)\|(\d+)\|([-\d.]+)\|([-\d.nan]+)\|([-\d.]+)")


def build(base, img, gt):
    nb = copy.deepcopy(base)
    cells = list(nb["cells"])
    i = next(k for k, c in enumerate(cells) if CFG_MARKER in "".join(c["source"]))
    cells.insert(i + 1, nbformat.v4.new_code_cell(f'cfg.image_path = r"{img}"'))
    j = next(k for k, c in enumerate(cells) if CANVAS_MARKER in "".join(c["source"]))
    nb["cells"] = cells[: j + 1] + [nbformat.v4.new_code_cell(
        DEBUG % {"gt": repr(gt), "scales": SCALES})]
    return nb


def main():
    base = nbformat.read(NB, as_version=4)
    gt = defaultdict(list)
    with open(GT_CSV) as f:
        for r in csv.DictReader(f):
            d = r["halo_diameter_mm_expert"].strip()
            gt[r["image_file"]].append(
                (float(r["x_px"]), float(r["y_px"]), float(d) if d else 0.0))

    rows = []
    for img in sorted(RAW.glob("gt_*.jpg")):
        nb = build(base, img, gt[img.name])
        NotebookClient(nb, kernel_name="antibiogram-test", timeout=2400).execute()
        text = "".join(o.get("text", "") for c in nb["cells"]
                       for o in c.get("outputs", []) if o.get("output_type") == "stream")
        for m in ROW_RE.findall(text):
            rows.append({"image": img.name, "x": float(m[0]), "y": float(m[1]),
                         "scale": float(m[2]), "n_px": int(m[3]),
                         "effect": float(m[4]),
                         "auc": float(m[5]) if m[5] != "nan" else float("nan"),
                         "lawn_sd": float(m[6])})
        print(f"[done] {img.name}", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[r["scale"]].append(r)
    n_disks = len({(r["image"], r["x"], r["y"]) for r in rows})

    print("\n" + "=" * 78)
    print(f"اثرِ فیلترِ فضایی بر جدایشِ هاله از لَون  (n={n_disks} دیسکِ دارایِ هاله)")
    print("=" * 78)
    print(f"  {'کرنل (×r_disk)':<16}{'پیکسل در کرنل':>15}{'اندازه‌اثر میانه':>18}"
          f"{'AUC میانه':>12}{'رشدِ مورد انتظارِ √N':>22}")
    base_eff = float(np.median([r["effect"] for r in by[0.0]]))
    for s in SCALES:
        sub = by[s]
        if not sub:
            continue
        eff = float(np.median([r["effect"] for r in sub]))
        auc = float(np.nanmedian([r["auc"] for r in sub]))
        n = int(np.median([r["n_px"] for r in sub]))
        print(f"  {s:<16.2f}{n:>15}{eff:>18.2f}{auc:>12.3f}"
              f"{base_eff * math.sqrt(n):>22.2f}")

    print("\n  ستونِ آخر یعنی «اگر نویز کاملاً مستقل بود، اندازه‌اثر باید این می‌شد».")
    print("  فاصله‌ی ستونِ اندازه‌اثر از آن، سهمِ نویزِ **ساختاریافته** را نشان می‌دهد.")
    print(f"\nجزئیات: {OUT}")


if __name__ == "__main__":
    main()

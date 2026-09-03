"""
آیا «مرز در تصویر ثبت نشده» است؟ -- آزمونِ مستقیمِ همین ادعا.

## چرا این آزمون لازم است

اندازه‌گیریِ قبلی (۰.۳σ) چیزِ دیگری را سنجیده بود: «داخلِ هاله» در برابرِ «لَونِ
میدانِ دور» -- یعنی دو ناحیه‌ی *دور از هم*. ولی ادعایِ «مرز ثبت نشده» درباره‌ی
**خودِ مرز** است، نه درباره‌ی دو ناحیه‌ی دور. این دو یکی نیستند:

  • ممکن است دو ناحیه‌ی دور کم‌تفاوت باشند ولی درست سرِ مرز یک پرش وجود داشته باشد
    (مثلاً وقتی شیبِ نور، تفاوتِ ناحیه‌ای را خنثی کرده باشد).
  • و برعکس، ممکن است دو ناحیه تفاوت داشته باشند ولی گذار آن‌قدر تدریجی باشد که
    هیچ مرزِ مکان‌یابی‌شدنی وجود نداشته باشد.

پس ادعا باید **سرِ خودِ مرزِ مرجع** سنجیده شود.

## روش

برایِ هر دیسکِ دارایِ هاله‌ی کارشناسی، دو نوارِ باریک بلافاصله در دو سویِ شعاعِ
مرجع برداشته می‌شود (پهنایِ هر نوار نسبت به شعاعِ دیسک، نه پیکسلِ مطلق) و
تفکیک‌پذیریِ آن دو با AUC سنجیده می‌شود:

    AUC ≈ ۰.۵  ->  مرز واقعاً در تصویر نیست
    AUC >> ۰.۵ ->  مرز ثبت شده، و شکستِ ما شکستِ *روش* بوده نه نبودِ اطلاعات

هم‌چنین یک **آزمونِ اوراکل** انجام می‌شود: بهترین آستانه‌ی شدت که *با دانستنِ خودِ
جواب* انتخاب شود چه خطایی می‌دهد. این کرانِ بالایِ کارِ هر روشِ آستانه‌ایِ ممکن است.
اگر اوراکل هم شکست بخورد، ادعا درست است؛ اگر اوراکل خوب جواب دهد، ادعا غلط است.

خروجی: ground_truth/diagnostics/boundary_recorded_test.csv
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
OUT = REPO / "ground_truth" / "diagnostics" / "boundary_recorded_test.csv"

CFG_MARKER = "cfg.halo_fusion_otsu_percentile"
CANVAS_MARKER = "ماژول ۱۵.۵ (جدید)"

BAND = 0.35        # پهنایِ هر نوار، × شعاعِ دیسک

DEBUG = r'''
import numpy as _np

_GT = %(gt)s
_BAND = %(band)f

def _auc(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 30 or n2 < 30:
        return float("nan")
    allv = _np.concatenate([a, b])
    o = _np.argsort(allv, kind="mergesort")
    rk = _np.empty(len(allv), dtype=_np.float64)
    rk[o] = _np.arange(1, len(allv) + 1, dtype=_np.float64)
    r1 = float(_np.sum(rk[:n1]))
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n2)

for _d in dishes:
    _ox, _oy = _d["roi_offset_xy"]
    _canvas = _d["agar_canvas"].astype(_np.float64)
    _agar = _d["agar_mask"] > 0
    _ds = [2.0 * float(_c["r"]) for _c in _d["final_candidates"]]
    _ppm = (float(_np.mean([_x for _x in _ds if _x <= min(_ds) * 1.30])) / 6.0) if _ds else 0.0
    if not _ppm or _canvas.size == 0:
        continue
    _h, _w = _canvas.shape[:2]
    _yy, _xx = _np.mgrid[0:_h, 0:_w]

    for _c in _d["final_candidates"]:
        _cx, _cy, _rd = float(_c["x"]), float(_c["y"]), float(_c["r"])
        _gx, _gy = _cx + _ox, _cy + _oy
        _best, _bd = None, 1e18
        for _g in _GT:
            _dd = _np.hypot(_gx - _g[0], _gy - _g[1])
            if _dd < _bd:
                _bd, _best = _dd, _g
        if _best is None or _bd > _rd * 2.0 or _best[2] <= 0:
            continue
        _rz = 0.5 * _best[2] * _ppm            # شعاعِ مرزِ مرجع، پیکسل
        if _rz <= 1.3 * _rd:
            continue
        _rr = _np.hypot(_xx - _cx, _yy - _cy)
        _bw = _BAND * _rd

        # دو نوارِ باریک، دقیقاً در دو سویِ مرزِ مرجع
        _inb = _agar & (_rr >= _rz - _bw) & (_rr < _rz) & (_rr > 1.05 * _rd)
        _outb = _agar & (_rr >= _rz) & (_rr < _rz + _bw)
        _a = _canvas[_inb]; _b = _canvas[_outb]
        if len(_a) < 30 or len(_b) < 30:
            continue
        _sd = float(_np.std(_b, ddof=1))
        _eff = abs(float(_np.mean(_b)) - float(_np.mean(_a))) / _sd if _sd > 1e-9 else 0.0
        _au = _auc(_a[:: max(1, len(_a)//4000)], _b[:: max(1, len(_b)//4000)])

        # آزمونِ اوراکل: بهترین آستانه‌ی ممکن رویِ پروفایلِ شعاعی
        _rmax = min(_rz * 2.2, float(_rr[_agar].max()) if _agar.any() else _rz * 2.2)
        _edges = _np.arange(1.05 * _rd, _rmax, max(2.0, 0.05 * _rd))
        _prof = []
        for _k in range(len(_edges) - 1):
            _m = _agar & (_rr >= _edges[_k]) & (_rr < _edges[_k + 1])
            _prof.append(float(_np.mean(_canvas[_m])) if int(_np.count_nonzero(_m)) >= 20 else _np.nan)
        _prof = _np.array(_prof); _ctr = 0.5 * (_edges[:-1] + _edges[1:])
        _ok = ~_np.isnan(_prof)
        _bestr = _np.nan
        if int(_ok.sum()) >= 6:
            _p = _prof[_ok]; _cc = _ctr[_ok]
            _cands = _np.linspace(_p.min(), _p.max(), 60)
            _errs = []
            for _t in _cands:
                _cross = _np.where(_p >= _t)[0]
                _r_t = _cc[_cross[0]] if len(_cross) else _np.nan
                _errs.append(abs(_r_t - _rz) if _np.isfinite(_r_t) else 1e9)
            _bestr = float(_np.min(_errs))
        # کنترلِ پوچ: همان اوراکل، ولی به سمتِ یک هدفِ *غلط*. اگر اوراکل هر هدفی را
        # به‌راحتی بزند، عددش بی‌معنی است و چیزی درباره‌ی وجودِ مرز نمی‌گوید.
        _others = [_g[2] for _g in _GT if _g[2] > 0 and abs(_g[2] - _best[2]) > 1e-6]
        _null = _np.nan
        if _others and int(_ok.sum()) >= 6:
            _rz_f = 0.5 * float(_np.median(_others)) * _ppm
            _e2 = []
            for _t in _cands:
                _cr = _np.where(_p >= _t)[0]
                _rt = _cc[_cr[0]] if len(_cr) else _np.nan
                _e2.append(abs(_rt - _rz_f) if _np.isfinite(_rt) else 1e9)
            _null = float(_np.min(_e2))
        print("B|%%.1f|%%.1f|%%.2f|%%.4f|%%.4f|%%.3f|%%.3f|%%.3f" %% (
            _gx, _gy, _best[2], _eff, _au, _bestr * 2.0 / _ppm, _rd,
            _null * 2.0 / _ppm if _np.isfinite(_null) else -1.0))
'''

ROW = re.compile(r"B\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)\|([-\d.nan]+)\|([-\d.nan]+)\|([-\d.]+)\|([-\d.nan]+)")


def build(base, img, gt):
    nb = copy.deepcopy(base)
    cells = list(nb["cells"])
    i = next(k for k, c in enumerate(cells) if CFG_MARKER in "".join(c["source"]))
    cells.insert(i + 1, nbformat.v4.new_code_cell(f'cfg.image_path = r"{img}"'))
    j = next(k for k, c in enumerate(cells) if CANVAS_MARKER in "".join(c["source"]))
    nb["cells"] = cells[: j + 1] + [nbformat.v4.new_code_cell(
        DEBUG % {"gt": repr(gt), "band": BAND})]
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
        for m in ROW.findall(text):
            rows.append({"image": img.name, "x": float(m[0]), "y": float(m[1]),
                         "gt_mm": float(m[2]), "edge_effect": float(m[3]),
                         "edge_auc": float(m[4]) if m[4] != "nan" else float("nan"),
                         "oracle_err_mm": float(m[5]) if m[5] != "nan" else float("nan"),
                         "r_disk": float(m[6]),
                         "null_oracle_mm": float(m[7]) if m[7] != "nan" else float("nan")})
        print(f"[done] {img.name}", flush=True)

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    eff = np.array([r["edge_effect"] for r in rows])
    auc = np.array([r["edge_auc"] for r in rows]); auc = np.maximum(auc, 1 - auc)
    orc = np.array([r["oracle_err_mm"] for r in rows])
    orc = orc[np.isfinite(orc) & (orc < 900)]

    print("\n" + "=" * 74)
    print(f"تفکیک‌پذیری **سرِ خودِ مرزِ مرجع**  (n={len(rows)} دیسک)")
    print("=" * 74)
    print(f"  اندازه‌اثرِ لبه : میانه {np.median(eff):.2f}σ   "
          f"چارک۱ {np.percentile(eff,25):.2f}   چارک۳ {np.percentile(eff,75):.2f}")
    print(f"  AUC لبه        : میانه {np.nanmedian(auc):.3f}   "
          f"چارک۱ {np.nanpercentile(auc,25):.3f}   چارک۳ {np.nanpercentile(auc,75):.3f}")
    print(f"  کسری با AUC>۰.۶: {100*np.nanmean(auc>0.60):.0f}%   "
          f"AUC>۰.۷: {100*np.nanmean(auc>0.70):.0f}%")
    print()
    print("=" * 74)
    print("آزمونِ اوراکل: بهترین آستانه‌ی ممکن، با دانستنِ خودِ جواب")
    print("=" * 74)
    print(f"  خطایِ قطر: میانه {np.median(orc):.2f} mm   میانگین {orc.mean():.2f} mm   "
          f"n={len(orc)}")
    print(f"  کسری با خطایِ ≤۲mm: {100*np.mean(orc<=2):.0f}%   "
          f"≤۴mm: {100*np.mean(orc<=4):.0f}%")
    nul = np.array([r["null_oracle_mm"] for r in rows])
    nul = nul[np.isfinite(nul) & (nul >= 0) & (nul < 900)]
    print()
    print("  کنترلِ پوچ -- همان اوراکل به سمتِ هدفِ *غلط* (هاله‌ی دیسکِ دیگر):")
    print(f"    خطا: میانه {np.median(nul):.2f} mm   میانگین {nul.mean():.2f} mm   n={len(nul)}")
    print(f"    کسری با خطایِ ≤۲mm: {100*np.mean(nul<=2):.0f}%")
    print("\n  اگر کنترلِ پوچ هم به‌خوبیِ اوراکلِ واقعی باشد، اوراکل صرفاً نشان می‌دهد")
    print("  «هر شعاعی در بازه‌ی جست‌وجو با یک آستانه قابلِ‌دستیابی است» و چیزی درباره‌ی")
    print("  وجودِ مرز ثابت نمی‌کند. تفاوتِ معنادارِ این دو است که ادعا را می‌سنجد.")
    print(f"\nجزئیات: {OUT}")


if __name__ == "__main__":
    main()

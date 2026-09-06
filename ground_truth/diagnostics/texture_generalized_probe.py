"""
تعمیمِ آزمونِ بافتِ فوریه: ماسکِ مرجعِ جایگزین + چند تبدیلِ تصویر.

آزمونِ قبلی (`texture_fft_probe.py`) نشان داد بافتِ فوریه رویِ **تصویرِ خام** بدتر از
روشناییِ خام است (AUC=۰.۶۳۲ در برابرِ ۰.۷۱۱). کاربر سه تعمیم خواست:

  ۱) تبدیل‌هایِ مختلفِ تصویر را پیش از FFT امتحان کن -- شاید تصویرِ خام نیست که باید
     تحلیل شود، بلکه نسخه‌ای که فرکانسِ نویز/باکتری در آن بهتر جدا می‌شود.
  ۲) ماسکِ ناحیه‌ی «رشدِ قطعی» را از میدانِ دورِ صدکی به **بیرونِ دایره‌ی ۳۵mm قطر
     دورِ هر دیسک** تغییر بده.
  ۳) رویِ همان ناحیه، چند تبدیل/تابع را بیازما و ببین کدام نویز/باکتری را بهتر جدا
     می‌کند؛ همان را به‌جایِ FFTِ خام به‌عنوانِ معیارِ مقایسه استفاده کن.

## طراحی

هندسه (ماسکِ آگار، ماسکِ رشدِ مرجع، نوارهایِ این‌باند/اوت‌باندِ کنارِ مرزِ مرجع) یک‌بار
با همان مختصات محاسبه می‌شود؛ سپس **همان پیکسل‌ها** زیرِ هر تبدیل نمونه‌برداری می‌شوند
تا مقایسه‌ی تبدیل‌ها دقیقاً روی یک مجموعه‌ی هندسیِ ثابت باشد، نه روی داده‌هایِ متفاوت.

**تبدیل‌هایِ آزموده‌شده:**
  raw       تصویرِ خام (خطِ پایه، همان آزمونِ قبلی با ماسکِ جدید)
  clahe     یکسان‌سازیِ هیستوگرامِ محلی (کنتراستِ محلی را بزرگ‌نمایی می‌کند)
  highpass  تصویر منهایِ نسخه‌یِ محوشده‌یِ خودش (شیبِ نور را حذف، فقط بافتِ محلی می‌ماند)
  log       تبدیلِ لگاریتمی (فشرده‌سازیِ غیرخطیِ دامنه)
  localstd  نقشه‌یِ انحرافِ‌معیارِ محلی (بافت را به‌عنوانِ یک سیگنالِ مستقل می‌سازد،
            نه شدت را)

نکته‌ی مهم: برخلافِ آزمونِ AUCِ آستانه‌ای (بندِ ۶ که در آن تبدیل‌هایِ یکنوا نمی‌توانستند
AUC را عوض کنند)، این‌جا تبدیل رویِ **طیفِ فوریه** اثر می‌گذارد، نه رتبه‌یِ پیکسل‌ها --
پس highpass/localstd/log واقعاً می‌توانند طیف را عوض کنند و آزمودنشان بی‌معنی نیست.

## ماسکِ مرجعِ جدید

`growth_mask = agar_mask AND NOT (اجتماعِ دایره‌هایِ ۱۷.۵mm‌شعاعی دورِ هر final_candidate)`

۳۵mm از بزرگ‌ترین قطرِ هاله‌یِ مشاهده‌شده در دیتاست (`gt_08`#۵ = ۳۵mm) به‌عنوانِ سقفِ
ایمن انتخاب شده -- هر نقطه‌ی بیرونِ این دایره، مستقل از آنتی‌بیوتیک، قطعاً رشدکرده
است. این ماسک نسبت به میدانِ دورِ صدکی **پوششِ پیکسلیِ بیشتری** می‌دهد.

خروجی: ground_truth/diagnostics/texture_generalized_probe.csv
"""
import copy
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import nbformat
import numpy as np
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]
NB = REPO / "disk_detection_pipeline_modular.ipynb"
RAW = REPO / "ground_truth" / "raw_images"
GT_CSV = REPO / "ground_truth" / "ground_truth_expert_readings.csv"
OUT = REPO / "ground_truth" / "diagnostics" / "texture_generalized_probe.csv"

CFG_MARKER = "cfg.halo_fusion_otsu_percentile"
CANVAS_MARKER = "ماژول ۱۵.۵ (جدید)"

BAND = 0.35
N_REF_PATCHES = 250
N_TEST_PATCHES = 50
EXCLUDE_RADIUS_MM = 17.5     # نیمِ ۳۵mm

TRANSFORMS = ["raw", "clahe", "highpass", "log", "localstd"]

DEBUG = r'''
import numpy as _np
import cv2 as _cv2

_GT = %(gt)s
_BAND = %(band)f
_NREF = %(nref)d
_NTEST = %(ntest)d
_EXCL_MM = %(excl)f
_TRANSFORMS = %(transforms)s
_SEED = 20260906

def _auc(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 15 or n2 < 15:
        return float("nan")
    allv = _np.concatenate([a, b])
    o = _np.argsort(allv, kind="mergesort")
    rk = _np.empty(len(allv), dtype=_np.float64)
    rk[o] = _np.arange(1, len(allv) + 1, dtype=_np.float64)
    r1 = float(_np.sum(rk[:n1]))
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n2)

def _apply_transform(canvas_u8, name):
    if name == "raw":
        return canvas_u8.astype(_np.float64)
    if name == "clahe":
        return _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(canvas_u8).astype(_np.float64)
    if name == "highpass":
        blur = _cv2.GaussianBlur(canvas_u8, (0, 0), sigmaX=canvas_u8.shape[0] * 0.02 + 3)
        return canvas_u8.astype(_np.float64) - blur.astype(_np.float64) + 128.0
    if name == "log":
        return _np.log1p(canvas_u8.astype(_np.float64)) * (255.0 / _np.log1p(255.0))
    if name == "localstd":
        f = canvas_u8.astype(_np.float64)
        k = (5, 5)
        mean = _cv2.boxFilter(f, -1, k)
        meansq = _cv2.boxFilter(f * f, -1, k)
        var = _np.maximum(meansq - mean * mean, 0.0)
        return _np.sqrt(var)
    raise ValueError(name)

def _radial_psd(patch, win, freq_r, n_bins):
    f = _np.fft.fftshift(_np.fft.fft2(patch * win))
    p = _np.abs(f) ** 2
    out = _np.zeros(n_bins, dtype=_np.float64)
    cnt = _np.zeros(n_bins, dtype=_np.float64)
    fr_flat = freq_r.ravel()
    _np.add.at(out, fr_flat, p.ravel())
    _np.add.at(cnt, fr_flat, 1.0)
    return out / _np.maximum(cnt, 1.0)

for _d in dishes:
    _ox, _oy = _d["roi_offset_xy"]
    _canvas_u8 = _d["agar_canvas"]
    _agar = _d["agar_mask"] > 0
    _ds = [2.0 * float(_c["r"]) for _c in _d["final_candidates"]]
    _ppm = (float(_np.mean([_x for _x in _ds if _x <= min(_ds) * 1.30])) / 6.0) if _ds else 0.0
    if not _ppm or _canvas_u8.size == 0:
        continue
    _h, _w = _canvas_u8.shape[:2]
    _yy, _xx = _np.mgrid[0:_h, 0:_w]

    _nk = max(9, int(round(_ppm)) | 1)
    _win = _np.outer(_np.hanning(_nk), _np.hanning(_nk))
    _cy0, _cx0 = _np.mgrid[0:_nk, 0:_nk] - _nk // 2
    _fr = _np.round(_np.hypot(_cy0, _cx0)).astype(int)
    _fr = _np.clip(_fr, 0, _nk // 2)
    _nbins = _nk // 2 + 1
    _rng = _np.random.default_rng(_SEED)

    # ── ماسکِ مرجعِ جدید: بیرونِ دایره‌ی ۳۵mm‌قطر دورِ هر دیسک ─────────────────
    _excl_px = _EXCL_MM * _ppm
    _near_any_disk = _np.zeros((_h, _w), dtype=bool)
    for _c in _d["final_candidates"]:
        _near_any_disk |= (_np.hypot(_xx - _c["x"], _yy - _c["y"]) < _excl_px)
    _growth_mask = _agar & (~_near_any_disk)

    def _sample_patches(img, mask, n):
        ys, xs = _np.nonzero(mask)
        if len(ys) < 5:
            return []
        idx = _rng.choice(len(ys), size=min(n, len(ys)), replace=False)
        out = []
        for i in idx:
            y, x = int(ys[i]), int(xs[i])
            y0, x0 = y - _nk // 2, x - _nk // 2
            if y0 < 0 or x0 < 0 or y0 + _nk > _h or x0 + _nk > _w:
                continue
            if not _agar[y0:y0 + _nk, x0:x0 + _nk].all():
                continue
            out.append(_radial_psd(img[y0:y0 + _nk, x0:x0 + _nk], _win, _fr, _nbins))
        return out

    for _tname in _TRANSFORMS:
        _timg = _apply_transform(_canvas_u8, _tname)
        _refs = _sample_patches(_timg, _growth_mask, _NREF)
        if len(_refs) < 20:
            continue
        _sig = _np.median(_np.stack(_refs), axis=0)
        _sig_n = _sig / max(_np.linalg.norm(_sig), 1e-9)

        def _score(patches):
            out = []
            for p in patches:
                pn = p / max(_np.linalg.norm(p), 1e-9)
                out.append(float(_np.dot(pn, _sig_n)))
            return _np.array(out)

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
            _rz = 0.5 * _best[2] * _ppm
            if _rz <= 1.3 * _rd:
                continue
            _rr = _np.hypot(_xx - _cx, _yy - _cy)
            _bw = _BAND * _rd
            _inb = _agar & (_rr >= _rz - _bw) & (_rr < _rz) & (_rr > 1.05 * _rd)
            _outb = _agar & (_rr >= _rz) & (_rr < _rz + _bw)

            _pin = _sample_patches(_timg, _inb, _NTEST)
            _pout = _sample_patches(_timg, _outb, _NTEST)
            if len(_pin) < 15 or len(_pout) < 15:
                continue
            _sin = _score(_pin); _sout = _score(_pout)
            _sd = float(_np.std(_sout, ddof=1))
            _eff = abs(float(_np.mean(_sout) - _np.mean(_sin))) / _sd if _sd > 1e-9 else 0.0
            _au = _auc(_sin, _sout)
            print("G|%%s|%%.1f|%%.1f|%%.4f|%%.4f" %% (
                _tname, _gx, _gy, _eff, _au if _np.isfinite(_au) else -1.0))
'''

ROW = re.compile(r"G\|(\w+)\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)")


def build(base, img, gt):
    nb = copy.deepcopy(base)
    cells = list(nb["cells"])
    i = next(k for k, c in enumerate(cells) if CFG_MARKER in "".join(c["source"]))
    cells.insert(i + 1, nbformat.v4.new_code_cell(f'cfg.image_path = r"{img}"'))
    j = next(k for k, c in enumerate(cells) if CANVAS_MARKER in "".join(c["source"]))
    nb["cells"] = cells[: j + 1] + [nbformat.v4.new_code_cell(
        DEBUG % {"gt": repr(gt), "band": BAND, "nref": N_REF_PATCHES,
                "ntest": N_TEST_PATCHES, "excl": EXCLUDE_RADIUS_MM,
                "transforms": TRANSFORMS})]
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
            rows.append({"image": img.name, "transform": m[0], "x": float(m[1]),
                         "y": float(m[2]), "effect": float(m[3]),
                         "auc": float(m[4]) if float(m[4]) >= 0 else float("nan")})
        print(f"[done] {img.name}", flush=True)

    if not rows:
        print("داده‌ای تولید نشد.")
        return

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 74)
    print("مقایسه‌ی تبدیل‌ها -- ماسکِ مرجع: بیرونِ دایره‌ی ۳۵mm‌قطر دورِ هر دیسک")
    print("=" * 74)
    print(f"  {'تبدیل':<12}{'n':>5}{'اندازه‌اثر (میانه)':>20}{'AUC (میانه)':>14}")
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["transform"]].append(r)
    for t in TRANSFORMS:
        sub = by_t.get(t, [])
        if not sub:
            print(f"  {t:<12}   داده‌ای نبود")
            continue
        eff = np.array([r["effect"] for r in sub])
        auc = np.array([r["auc"] for r in sub]); auc = np.maximum(auc, 1 - auc)
        print(f"  {t:<12}{len(sub):>5}{np.nanmedian(eff):>20.2f}{np.nanmedian(auc):>14.3f}")

    print("\n  مقایسه: AUC روشناییِ خام با ماسکِ میدانِ دورِ قبلی = 0.711")
    print(f"\nجزئیات: {OUT}")


if __name__ == "__main__":
    main()

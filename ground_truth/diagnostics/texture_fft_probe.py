"""
آزمونِ پیش‌فرضِ پیشنهادِ کاربر: تطبیقِ بافت/فرکانس برایِ اصلاحِ مرزِ هاله.

## شهودِ کاربر — و کجایش درست است

کاربر می‌گوید: «من و هر کسی مرزِ هاله را راحت می‌بینیم، پس سیستم هم باید بتواند».
این شهود **درست** است و دقیقاً با آزمونِ اوراکلِ قبلی هم‌خوان: بهترین آستانه‌ی ممکن
رویِ پروفایلِ شعاعی، مرز را با خطایِ میانه‌یِ ۰.۱۵mm بازمی‌یابد
(`boundary_recorded_test.py`). یعنی اطلاعات هست.

**نکته‌ی ظریفی که تناقضِ ظاهری را حل می‌کند:** آن اوراکل رویِ *پروفایلِ شعاعی* کار
می‌کند -- که خودش میانگینِ ۷۲ زاویه در هر شعاع است، یعنی از قبل یک میانگین‌گیریِ
ناحیه‌ای دارد. چشمِ انسان هم وقتی به یک حلقه نگاه می‌کند، آن را یک‌جا و به‌صورتِ
ناحیه‌ای می‌بیند، نه پیکسل‌به‌پیکسل. پس «راحت دیدن» با «۰.۳σ در سطحِ پیکسل» متناقض
نیست -- هر دو با هم سازگارند، چون ادراکِ انسان و اوراکل هر دو رویِ *میانگینِ ناحیه*
کار می‌کنند، نه رویِ یک پیکسلِ تنها.

## پیشنهادِ کاربر، به‌طورِ دقیق

ساختِ یک «کرنلِ معیارِ رشد» از طیفِ فوریه‌یِ ناحیه‌یِ قطعاً-رشدکرده، و تطبیقِ آن با
ناحیه‌یِ داخلِ هر دایره‌یِ هاله برایِ کوچک‌کردنِ دایره‌هایی که تا داخلِ ناحیه‌یِ رشد
پیش رفته‌اند.

## دو اصلاحِ روشی، پیش از آزمون

۱. **مرجعِ «قطعاً رشدکرده» را نمی‌توان از «داخلِ پتری منهایِ دایره‌هایِ هاله» ساخت.**
   این تعریف به خودِ خروجیِ سیستم وابسته است -- اگر دایره‌ای اشتباه (کوچک) کشیده
   شده باشد، ناحیه‌یِ «رشدِ قطعی» آلوده به لَونِ واقعیِ **بیرونِ همان دایره‌یِ غلط**
   می‌شود، ولی چون اسمش «قطعی» است، اشتباه هیچ‌وقت تصحیح نمی‌شود -- یک حلقه‌یِ
   خودتاییدکننده. مرجعِ امن همان چیزی است که خودِ پایپ‌لاین از قبل به این نام دارد:
   **میدانِ دور** (`far_field_ref`) -- پیکسل‌هایِ آگار که از *همه‌ی* دیسک‌ها دورترند
   و به‌لحاظِ هندسی نمی‌توانند داخلِ هیچ هاله‌ای باشند، مستقل از این‌که سیستم چه
   دایره‌ای کشیده باشد.

۲. **یک کرنلِ ۱×۱ میلی‌متر (~۱۳×۱۳ پیکسل در این رزولوشن) برایِ FFT بسیار کوچک است.**
   تبدیلِ فوریه‌یِ یک پنجره‌یِ ۱۳ پیکسلی فقط چند بینِ فرکانسیِ قابلِ‌اعتماد می‌دهد و
   به لبه‌هایِ خودِ پنجره و بلوک‌بندیِ فشرده‌سازیِ JPEG (بلوک‌هایِ ۸×۸) به‌شدت حساس
   است. **راه‌حل: میانگین‌گیری رویِ ده‌ها پنجره** به‌جایِ اعتماد به یک پنجره‌یِ تنها --
   دقیقاً همان اصلی که در بندِ بالا برایِ حلِ تناقضِ «دیدنِ راحت» به‌کار رفت. هر پنجره
   هم با پنجره‌یِ Hann ضرب می‌شود تا نشتِ لبه کم شود.

## روش

برایِ هر دیسکِ دارایِ هاله‌یِ مرجع (همان دیتاستِ `boundary_recorded_test.py`):

  ۱. از میدانِ دور، دهه‌ها پنجره‌یِ ۱×۱mm نمونه‌برداری و طیفِ توانِ فوریه‌یِ
     شعاعی‌میانگین‌شده‌یِ هرکدام محاسبه می‌شود؛ میانگینِ آن‌ها = «امضایِ فرکانسیِ رشد».
  ۲. از دو نوارِ باریکِ **درست کنارِ مرزِ مرجع** (همان این‌باند/اوت‌باندِ آزمونِ قبلی)
     چند ده پنجره نمونه‌برداری و امتیازِ «شباهت به امضایِ رشد» (فاصله‌یِ منفیِ اقلیدسی
     تا امضا) برایِ هرکدام محاسبه می‌شود.
  ۳. اگر بافت اطلاعاتِ واقعی دارد، امتیازِ شباهت باید بینِ این‌باند (نزدیک‌تر به رشد)
     و اوت‌باند (نزدیک‌تر به لَون) تفکیک‌پذیر باشد -- دقیقاً همان AUC/اندازه‌اثری که
     برایِ روشناییِ خام سنجیده شد (میانه‌یِ AUC=۰.۷۱۱).

**معیارِ پذیرش:** اگر AUCِ بافتی به‌طورِ معناداری از AUCِ روشناییِ خام **بهتر** باشد،
پیشنهاد اطلاعاتِ تازه‌ای اضافه می‌کند و ارزشِ ساختن دارد. اگر تفاوتِ معناداری نداشته
باشد یا بدتر باشد، نتیجه با یافته‌یِ قبلیِ کانالِ بافت (تسکِ #۴۴: هم‌پوشانیِ ۹۹-۱۰۰٪
توزیع‌ها) هم‌خوان است و پیشنهاد چیزِ تازه‌ای که روشِ فعلی نداشته باشد اضافه نمی‌کند.

خروجی: ground_truth/diagnostics/texture_fft_probe.csv
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
OUT = REPO / "ground_truth" / "diagnostics" / "texture_fft_probe.csv"

CFG_MARKER = "cfg.halo_fusion_otsu_percentile"
CANVAS_MARKER = "ماژول ۱۵.۵ (جدید)"

BAND = 0.35          # همان پهنایِ نوار در boundary_recorded_test.py
N_REF_PATCHES = 250   # سقفِ پنجره‌هایِ مرجع از میدانِ دور
N_TEST_PATCHES = 50   # سقفِ پنجره‌ها در هر نوار

DEBUG = r'''
import numpy as _np

_GT = %(gt)s
_BAND = %(band)f
_NREF = %(nref)d
_NTEST = %(ntest)d
_SEED = 20260906           # قطعیت: بذرِ ثابت برایِ نمونه‌برداریِ تصادفی

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

def _radial_psd(patch, win, freq_r, n_bins):
    """طیفِ توانِ فوریه‌یِ شعاعی‌میانگین‌شده‌یِ یک پنجره -- توصیفگرِ بافتِ ناوردا به‌جهت."""
    f = _np.fft.fftshift(_np.fft.fft2(patch.astype(_np.float64) * win))
    p = _np.abs(f) ** 2
    out = _np.zeros(n_bins, dtype=_np.float64)
    cnt = _np.zeros(n_bins, dtype=_np.float64)
    fr_flat = freq_r.ravel()
    _np.add.at(out, fr_flat, p.ravel())
    _np.add.at(cnt, fr_flat, 1.0)
    return out / _np.maximum(cnt, 1.0)

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

    # اندازه‌ی کرنل: ۱mm، فرد، حداقل ۹ پیکسل تا FFT بی‌معنا نشود
    _nk = max(9, int(round(_ppm)) | 1)
    _win = _np.outer(_np.hanning(_nk), _np.hanning(_nk))
    _cy0, _cx0 = _np.mgrid[0:_nk, 0:_nk] - _nk // 2
    _fr = _np.round(_np.hypot(_cy0, _cx0)).astype(int)
    _fr = _np.clip(_fr, 0, _nk // 2)
    _nbins = _nk // 2 + 1
    _rng = _np.random.default_rng(_SEED)

    def _sample_patches(mask, n):
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
            patch = _canvas[y0:y0 + _nk, x0:x0 + _nk]
            if not _agar[y0:y0 + _nk, x0:x0 + _nk].all():
                continue
            out.append(_radial_psd(patch, _win, _fr, _nbins))
        return out

    # مرجعِ رشد: میدانِ دور -- همان تعریفِ ایمن و مستقل از خروجیِ سیستم
    _dist = _np.full((_h, _w), 1e9, dtype=_np.float64)
    for _c in _d["final_candidates"]:
        _dist = _np.minimum(_dist, _np.hypot(_xx - _c["x"], _yy - _c["y"]))
    _far = _agar & (_dist >= _np.percentile(_dist[_agar], 80))
    _refs = _sample_patches(_far, _NREF)
    if len(_refs) < 20:
        continue
    _sig = _np.median(_np.stack(_refs), axis=0)          # امضایِ فرکانسیِ رشد
    _sig_n = _sig / max(_np.linalg.norm(_sig), 1e-9)

    def _score(patches):
        out = []
        for p in patches:
            pn = p / max(_np.linalg.norm(p), 1e-9)
            out.append(float(_np.dot(pn, _sig_n)))          # شباهتِ کسینوسی به امضا
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

        _pin = _sample_patches(_inb, _NTEST)
        _pout = _sample_patches(_outb, _NTEST)
        if len(_pin) < 15 or len(_pout) < 15:
            continue
        _sin = _score(_pin); _sout = _score(_pout)
        _sd = float(_np.std(_sout, ddof=1))
        _eff = abs(float(_np.mean(_sout) - _np.mean(_sin))) / _sd if _sd > 1e-9 else 0.0
        _au = _auc(_sin, _sout)
        print("T|%%.1f|%%.1f|%%.4f|%%.4f|%%d|%%d" %% (
            _gx, _gy, _eff, _au if _np.isfinite(_au) else -1.0, len(_pin), len(_pout)))
'''

ROW = re.compile(r"T\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)\|([-\d.]+)\|(\d+)\|(\d+)")


def build(base, img, gt):
    nb = copy.deepcopy(base)
    cells = list(nb["cells"])
    i = next(k for k, c in enumerate(cells) if CFG_MARKER in "".join(c["source"]))
    cells.insert(i + 1, nbformat.v4.new_code_cell(f'cfg.image_path = r"{img}"'))
    j = next(k for k, c in enumerate(cells) if CANVAS_MARKER in "".join(c["source"]))
    nb["cells"] = cells[: j + 1] + [nbformat.v4.new_code_cell(
        DEBUG % {"gt": repr(gt), "band": BAND, "nref": N_REF_PATCHES, "ntest": N_TEST_PATCHES})]
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
                         "texture_effect": float(m[2]),
                         "texture_auc": float(m[3]) if float(m[3]) >= 0 else float("nan"),
                         "n_in": int(m[4]), "n_out": int(m[5])})
        print(f"[done] {img.name}", flush=True)

    if not rows:
        print("هیچ دیسکی به اندازه‌یِ کافی پنجره نداشت -- آزمون قابلِ اجرا نبود.")
        return

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    eff = np.array([r["texture_effect"] for r in rows])
    auc = np.array([r["texture_auc"] for r in rows]); auc = np.maximum(auc, 1 - auc)

    print("\n" + "=" * 74)
    print(f"تفکیک‌پذیریِ بافتیِ فوریه، سرِ خودِ مرزِ مرجع  (n={len(rows)} دیسک)")
    print("=" * 74)
    print(f"  اندازه‌اثرِ بافتی : میانه {np.nanmedian(eff):.2f}   "
          f"چارک۱ {np.nanpercentile(eff,25):.2f}   چارک۳ {np.nanpercentile(eff,75):.2f}")
    print(f"  AUC بافتی        : میانه {np.nanmedian(auc):.3f}   "
          f"چارک۱ {np.nanpercentile(auc,25):.3f}   چارک۳ {np.nanpercentile(auc,75):.3f}")
    print()
    print("  مقایسه با روشناییِ خام (boundary_recorded_test.py):")
    print("    AUC روشناییِ خام (میانه)  = 0.711")
    print(f"    AUC بافتیِ فوریه (میانه) = {np.nanmedian(auc):.3f}")
    n = len(auc); k = int(np.sum(auc > 0.711))
    z = (k - n / 2) / math.sqrt(n * 0.25)
    print(f"\n  تعدادِ دیسک‌هایی که بافت از روشناییِ خام بهتر است: {k} از {n}  ->  z={z:+.2f}")
    print(f"\nجزئیات: {OUT}")


if __name__ == "__main__":
    main()

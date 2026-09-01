"""
آماده‌سازیِ تصویر/پروفایل پیش از تبدیلِ نوری (پیگیریِ بندِ ۶ نقشه‌ی راه).

پس‌زمینه: آزمونِ قبلی نشان داد کوبلکا–مونک R² را بهتر می‌کند ولی MAE را بدتر.
تشخیصِ اولیه‌ی من «واگراییِ (1-R)^2/(2R) در R→0» بود. اندازه‌گیریِ دامنه‌ی واقعیِ R
آن تشخیص را رد می‌کند:

    کمینه‌ی R  — میانه ۰٫۹۰۷ ، صدکِ ۱۰ ۰٫۸۳۸ ، مطلقاً کمینه ۰٫۳۲
    بیشینه‌ی R — میانه ۱٫۰۵۷ ، تا ۱٫۲۷

یعنی R اصلاً به صفر نزدیک نمی‌شود. مشکلِ واقعی دو چیزِ دیگر است:

  (۱) بریدنِ سقف: به‌طور میانه **۳۵٪** از هر پروفایل R ≥ 1 دارد و
      clip(R, EPS, 1-EPS) آن را دقیقاً صفر می‌کند — یعنی همان فلاتِ لَون که
      مجانبِ بالاییِ سیگموئید (B) رویش تکیه دارد، صاف می‌شود.
  (۲) فشرده‌سازی: با R≈0.9 مقدارِ (1-R)^2 برابر ۰٫۰۱ است، پس دامنه‌ی دینامیکیِ
      KM به میانه ۰٫۰۰۵ می‌رسد؛ تبدیل سیگنال را *فشرده* می‌کند نه خطی.

ریشه‌ی هر دو یکی است: KM به بازتابِ **مطلق** نیاز دارد (نسبت به سفیدِ مرجع)، ولی
ما فقط بازتابِ **نسبی به لَون** را داریم. ضریبِ مجهول R_lawn (بازتابِ مطلقِ خودِ
لَون) یک درجه‌ی آزادیِ گم‌شده است — و دقیقاً همان چیزی که یک وصله‌ی سفیدِ مرجع در
کادرِ عکس به ما می‌داد.

پس به‌جای حدس زدنِ یک پیش‌پردازش، R_lawn را **جاروب** می‌کنیم: اگر برای هیچ مقدارِ
معقولی از R_lawn تبدیل بهتر از خطِ پایه نشود، ایده ذاتاً کمکی نمی‌کند؛ اگر برای
بازه‌ای بشود، آن بازه همان شرطِ تصویربرداریِ پیشنهادی است.

پنج پیش‌پردازش سنجیده می‌شود (هرکدام با کنترلِ خودش):
  raw       شدتِ خام                                   (خطِ پایه)
  raw+dn    شدتِ خام + میانه‌ی متحرکِ ۳ حلقه‌ای          (کنترلِ نویزگیری)
  km_scan   KM با R_lawn جاروب‌شده                      (رفعِ بریدن + فشردگی)
  bl_scan   بیر–لامبرت با R_lawn جاروب‌شده
  inv_bl    وارونه‌سازیِ قطبیت (شبیه‌سازیِ نورِ پس‌زمینه‌ای) + بیر–لامبرت

کنترلِ raw+dn ضروری است: بدونِ آن، اگر نویزگیری کمک کند به‌اشتباه به حسابِ تبدیلِ
نوری گذاشته می‌شود.
"""
import json
import math
from collections import defaultdict

import numpy as np

PROFILES = "/tmp/radial_profiles.json"
OUT = "/tmp/optical_precond.json"
EPS = 1e-6

# جاروبِ بازتابِ مطلقِ لَون. کفِ ۰٫۲۰ و سقفِ ۰٫۹۵: لَونِ باکتریایی یک پراکنده‌ی قوی
# است ولی هرگز سفیدِ کامل نیست، و زیرِ ۰٫۲ دیگر در تصویرِ ۸ بیتی تفکیک‌پذیر نیست.
RLAWN_GRID = np.round(np.arange(0.20, 0.96, 0.05), 2)


# ─────────────────────────── برازنده‌ی سیگموئید ────────────────────────────
def fit(r, y, B, r_disk):
    """برازشِ لجستیک با مجانبِ بالاییِ B ثابت و A خطی — همان برازنده‌ی ماژولِ ۱۵.۹."""
    r = np.asarray(r, float)
    y = np.asarray(y, float)
    if len(r) < 8:
        return None
    r0g = np.linspace(r[0], r[-1], 60)
    wg = np.exp(np.linspace(math.log(max(0.05 * r_disk, 1e-3)),
                            math.log(max(3.0 * r_disk, 1e-2)), 40))
    best = None
    for w in wg:
        for r0 in r0g:
            s = 1.0 / (1.0 + np.exp(-np.clip((r - r0) / w, -60, 60)))
            den = float(np.sum((1 - s) ** 2))
            if den < 1e-9:
                continue
            A = float(np.sum((y - B * s) * (1 - s)) / den)
            e = float(np.sum((y - (A + (B - A) * s)) ** 2))
            if best is None or e < best[0]:
                best = (e, A, float(r0), float(w))
    if best is None:
        return None
    sse, A, r0, w = best
    sr, sw = (r[-1] - r[0]) / 60.0, w * 0.5
    for _ in range(40):
        imp = False
        for dr, dw in ((sr, 0), (-sr, 0), (0, sw), (0, -sw)):
            r0n, wn = r0 + dr, max(w + dw, 1e-4)
            s = 1.0 / (1.0 + np.exp(-np.clip((r - r0n) / wn, -60, 60)))
            den = float(np.sum((1 - s) ** 2))
            if den < 1e-9:
                continue
            An = float(np.sum((y - B * s) * (1 - s)) / den)
            e = float(np.sum((y - (An + (B - An) * s)) ** 2))
            if e < sse:
                sse, A, r0, w = e, An, r0n, wn
                imp = True
        if not imp:
            sr *= 0.5
            sw *= 0.5
            if sr < 1e-4 and sw < 1e-4:
                break
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {"r0": r0, "w": w, "r2": 1.0 - sse / sst if sst > 1e-9 else 0.0}


# ─────────────────────────── پیش‌پردازش‌ها ────────────────────────────
def denoise(y):
    """میانه‌ی متحرکِ ۳ حلقه‌ای.

    چرا میانه و نه میانگین: فیلترِ میانه لبه‌ی پله‌ای را جابه‌جا نمی‌کند (خاصیتِ
    کلاسیکِ حفظِ لبه)، درحالی‌که میانگین مرزِ هاله را پهن و مکانش را مبهم می‌کند.
    پنجره‌ی ۳ کمینه‌ی ممکن است: تنها تک‌حلقه‌های پرت را حذف می‌کند.
    """
    y = np.asarray(y, float)
    if len(y) < 3:
        return y.copy()
    p = np.concatenate([y[:1], y, y[-1:]])
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def km(y, lawn, r_lawn):
    """کوبلکا–مونک با بازتابِ مطلق: R_abs = (y/lawn)·R_lawn.

    R_lawn بازتابِ مطلقِ لَون است — همان چیزی که یک وصله‌ی سفیدِ مرجع می‌دهد.
    با R_lawn < 1، مقادیرِ بالای لَون هم زیرِ ۱ می‌مانند و دیگر بریده نمی‌شوند.
    """
    R = np.clip(np.asarray(y, float) / max(lawn, EPS) * r_lawn, 1e-4, 1.0 - 1e-4)
    return (1.0 - R) ** 2 / (2.0 * R)


def bl(y, lawn, r_lawn):
    """بیر–لامبرت روی همان بازتابِ مطلق: OD = -ln(R_abs)."""
    R = np.clip(np.asarray(y, float) / max(lawn, EPS) * r_lawn, 1e-4, 1.0 - 1e-4)
    return -np.log(R)


def inv_bl(y, lawn):
    """وارونه‌سازیِ قطبیت: شبیه‌سازیِ هندسه‌ی نورِ پس‌زمینه‌ای.

    در بازتاب، لَون روشن و هاله تاریک است. در عبور (نورِ پشتِ ظرف) برعکس: هاله
    شفاف و روشن، لَون کدر و تاریک. با وارونه کردن نسبت به بیشینه‌ی پروفایل،
    همان قطبیت را می‌سازیم و بعد بیر–لامبرت را — که قانونِ *عبور* است — اعمال
    می‌کنیم.

    نکته‌ی مهم که بارِ اول اشتباه کردم: در این فضا مجانبِ لَون **صفر نیست**، بلکه
    بیشترین چگالیِ نوری است. مقدارِ B را هم با همان `top` برمی‌گردانیم تا برازنده
    مجانبِ درست را ببندد، وگرنه برازش بی‌معنی می‌شود.
    """
    y = np.asarray(y, float)
    top = float(np.max(y)) * 1.02          # کمی بالاتر از بیشینه تا لگاریتم معتبر بماند
    T = np.clip((top - y) / max(top, EPS), 1e-4, 1.0 - 1e-4)
    T_lawn = min(max((top - lawn) / max(top, EPS), 1e-4), 1.0 - 1e-4)
    return -np.log(T), -math.log(T_lawn)


# ─────────────────────────── داده ────────────────────────────
def ppm_map(recs):
    by = defaultdict(list)
    for x in recs:
        by[x["image"]].append(2.0 * x["r_disk"])
    return {k: float(np.mean([d for d in v if d <= min(v) * 1.30])) / 6.0
            for k, v in by.items()}


def clean(x):
    """حذفِ حلقه‌هایِ کم‌پوشش از دو سر (همان اصلاحِ پوششِ حلقه در P4)."""
    c = np.asarray(x["counts"], float)
    rc = np.asarray(x["ring_centers"], float)
    pr = np.asarray(x["profile"], float)
    ok = c >= 20
    if int(np.count_nonzero(ok)) < 8:
        return None
    f = int(np.argmax(ok))
    l = len(ok) - 1 - int(np.argmax(ok[::-1]))
    rc, pr, c = rc[f:l + 1], pr[f:l + 1], c[f:l + 1]
    k = c >= 20
    return (rc[k], pr[k]) if int(np.count_nonzero(k)) >= 8 else None


def summarise(name, rows, key_mm, key_r2, base_err):
    e = np.array([abs(r[key_mm] - r["gt"]) for r in rows if r["gt"] is not None])
    r2 = np.median([r[key_r2] for r in rows])
    n = len(e)
    w = int(np.sum(e < base_err))
    z = (w - n / 2) / math.sqrt(n * 0.25) if n else 0.0
    d = base_err - e
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    print(f"{name:<26}{r2:>9.3f}{e.mean():>8.2f}{np.median(e):>8.2f}"
          f"{w:>6}/{n:<4}{z:>+7.2f}{t:>+7.2f}")
    return e


def main():
    recs = json.load(open(PROFILES))
    pm = ppm_map(recs)

    rows = []
    for x in recs:
        if x.get("gt_num") is None or not x.get("lawn_mean"):
            continue
        cl = clean(x)
        if cl is None:
            continue
        rc, pr = cl
        lawn, rd, p = x["lawn_mean"], x["r_disk"], pm[x["image"]]
        prd = denoise(pr)

        rec = {"image": x["image"], "gt": x["gt_halo"], "ppm": p}

        f = fit(rc, pr, lawn, rd)
        fd = fit(rc, prd, lawn, rd)
        _iy, _ib = inv_bl(prd, lawn)
        fi = fit(rc, _iy, _ib, rd)              # مجانبِ لَون در فضایِ وارونه، نه صفر
        if not (f and fd and fi):
            continue
        rec["raw_mm"], rec["raw_r2"] = 2 * f["r0"] / p, f["r2"]
        rec["dn_mm"], rec["dn_r2"] = 2 * fd["r0"] / p, fd["r2"]
        rec["inv_mm"], rec["inv_r2"] = 2 * fi["r0"] / p, fi["r2"]

        for rl in RLAWN_GRID:
            fk = fit(rc, km(prd, lawn, rl), km(np.array([lawn]), lawn, rl)[0], rd)
            fb = fit(rc, bl(prd, lawn, rl), bl(np.array([lawn]), lawn, rl)[0], rd)
            if fk:
                rec[f"km{rl:.2f}_mm"], rec[f"km{rl:.2f}_r2"] = 2 * fk["r0"] / p, fk["r2"]
            if fb:
                rec[f"bl{rl:.2f}_mm"], rec[f"bl{rl:.2f}_r2"] = 2 * fb["r0"] / p, fb["r2"]
        rows.append(rec)

    has = [r for r in rows if r["gt"] is not None]
    base = np.array([abs(r["raw_mm"] - r["gt"]) for r in has])

    print(f"n={len(rows)} پروفایل، {len(has)} با هاله‌ی مرجع\n")
    print("=" * 76)
    print("پیش‌پردازش‌هایِ بدونِ پارامترِ آزاد")
    print("=" * 76)
    print(f"{'روش':<26}{'R² میانه':>9}{'MAE':>8}{'میانه':>8}{'بهتر':>11}{'z':>7}{'t':>7}")
    summarise("raw (خطِ پایه)", has, "raw_mm", "raw_r2", base)
    summarise("raw + میانه‌ی ۳ حلقه‌ای", has, "dn_mm", "dn_r2", base)
    summarise("وارونه‌سازی + بیر–لامبرت", has, "inv_mm", "inv_r2", base)

    print()
    print("=" * 76)
    print("جاروبِ بازتابِ مطلقِ لَون (R_lawn) — کوبلکا–مونک")
    print("=" * 76)
    print(f"{'R_lawn':<26}{'R² میانه':>9}{'MAE':>8}{'میانه':>8}{'بهتر':>11}{'z':>7}{'t':>7}")
    km_best = None
    for rl in RLAWN_GRID:
        k = f"km{rl:.2f}"
        if f"{k}_mm" not in has[0]:
            continue
        e = summarise(f"  R_lawn = {rl:.2f}", has, f"{k}_mm", f"{k}_r2", base)
        if km_best is None or e.mean() < km_best[1]:
            km_best = (rl, e.mean())

    print()
    print("=" * 76)
    print("جاروبِ بازتابِ مطلقِ لَون (R_lawn) — بیر–لامبرت")
    print("=" * 76)
    print(f"{'R_lawn':<26}{'R² میانه':>9}{'MAE':>8}{'میانه':>8}{'بهتر':>11}{'z':>7}{'t':>7}")
    bl_best = None
    for rl in RLAWN_GRID:
        k = f"bl{rl:.2f}"
        if f"{k}_mm" not in has[0]:
            continue
        e = summarise(f"  R_lawn = {rl:.2f}", has, f"{k}_mm", f"{k}_r2", base)
        if bl_best is None or e.mean() < bl_best[1]:
            bl_best = (rl, e.mean())

    print()
    print("=" * 76)
    print("جمع‌بندی")
    print("=" * 76)
    print(f"  خطِ پایه (شدتِ خام):        MAE = {base.mean():.2f} mm")
    if km_best:
        print(f"  بهترین KM در جاروب:        MAE = {km_best[1]:.2f} mm  "
              f"(R_lawn={km_best[0]:.2f})")
    if bl_best:
        print(f"  بهترین BL در جاروب:        MAE = {bl_best[1]:.2f} mm  "
              f"(R_lawn={bl_best[0]:.2f})")
    print("\n  هشدار: «بهترین» در جاروب روی همان داده انتخاب شده، پس یک برآوردِ")
    print("  خوش‌بینانه است. اگر حتی این برآوردِ خوش‌بینانه از خطِ پایه بهتر نباشد،")
    print("  ایده قطعاً کمکی نمی‌کند و نیازی به اعتبارسنجیِ جداگانه نیست.")

    json.dump(rows, open(OUT, "w"))
    print(f"\nجزئیات: {OUT}")


if __name__ == "__main__":
    main()

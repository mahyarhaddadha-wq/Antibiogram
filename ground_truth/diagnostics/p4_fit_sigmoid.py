"""
گامِ ۲ از پ۴: برازشِ سیگموئید به پروفایل‌ها و آزمونِ سه فرضیه.

مدل:  y(r) = A + (B - A) / (1 + exp(-(r - r0)/w))
  A  = مجانبِ داخلی (سطحِ آگارِ شفافِ داخلِ هاله)
  B  = مجانبِ بیرونی (سطحِ لَون) -- *پارامترِ آزاد نیست*، از میدانِ دور مستقلاً اندازه‌گیری شده
  r0 = مرکزِ گذار
  w  = پهنایِ گذار

فقط دو پارامترِ آزاد (r0, w) با جست‌وجویِ شبکه‌ای + پالایشِ محلی، فقط با numpy.
بدونِ scipy: هم قطعی است (مستقل از نقطه‌ی شروع) هم مستقیماً به C++ ترجمه می‌شود.

مرزِ گزارشی طبقِ تعریفِ کاربر:  *نقطه‌ی عطف نیست* (آن نقطه‌ی میانیِ گذار است).
مرز جایی است که منحنی از مجانبِ هاله جدا می‌شود -- شعاعی که در آن به کسرِ
DEPART_FRAC از راهِ A تا B رسیده‌ایم.
"""
import json
import math
from pathlib import Path

import numpy as np

PROFILES = Path("/tmp/radial_profiles.json")
DEPART_FRAC = 0.05          # «جدا شدن از مجانبِ هاله» = ۵٪ راه به‌سویِ لَون
INNER_RINGS_FOR_A = 3       # چند حلقه‌ی داخلی برایِ تخمینِ A


def logistic(r, A, B, r0, w):
    z = np.clip((r - r0) / max(w, 1e-6), -60.0, 60.0)
    return A + (B - A) / (1.0 + np.exp(-z))


def fit_logistic(r, y, B, r_disk):
    """
    برازشِ (r0, w) با شبکه + پالایش. A از حلقه‌هایِ داخلی، B از میدانِ دور (ثابت).
    خروجی: dict با A, B, r0, w, r2, depart_r
    """
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(r) < 6:
        return None
    A0 = float(np.median(y[:INNER_RINGS_FOR_A]))

    # شبکه‌ی اولیه: r0 در سرتاسرِ بازه، w به‌صورتِ لگاریتمی از خیلی تیز تا خیلی پهن
    r0_grid = np.linspace(r[0], r[-1], 60)
    w_grid = np.exp(np.linspace(math.log(0.05 * r_disk), math.log(3.0 * r_disk), 40))

    best = None
    for w in w_grid:
        for r0 in r0_grid:
            # A را به‌صورتِ خطی بهینه کن (با r0,w ثابت، مدل نسبت به A خطی است)
            s = 1.0 / (1.0 + np.exp(-np.clip((r - r0) / w, -60, 60)))
            # y ≈ A(1-s) + B s  ->  y - B s = A (1-s)
            denom = float(np.sum((1 - s) ** 2))
            if denom < 1e-9:
                continue
            A = float(np.sum((y - B * s) * (1 - s)) / denom)
            resid = y - (A + (B - A) * s)
            sse = float(np.sum(resid ** 2))
            if best is None or sse < best[0]:
                best = (sse, A, r0, w)
    if best is None:
        return None

    # پالایشِ محلی: تنصیفِ گامِ متوالی حولِ بهترین نقطه
    sse, A, r0, w = best
    step_r, step_w = (r[-1] - r[0]) / 60.0, w * 0.5
    for _ in range(40):
        improved = False
        for dr, dw in [(step_r, 0), (-step_r, 0), (0, step_w), (0, -step_w)]:
            r0n, wn = r0 + dr, max(w + dw, 1e-3)
            s = 1.0 / (1.0 + np.exp(-np.clip((r - r0n) / wn, -60, 60)))
            denom = float(np.sum((1 - s) ** 2))
            if denom < 1e-9:
                continue
            An = float(np.sum((y - B * s) * (1 - s)) / denom)
            e = float(np.sum((y - (An + (B - An) * s)) ** 2))
            if e < sse:
                sse, A, r0, w = e, An, r0n, wn
                improved = True
        if not improved:
            step_r *= 0.5
            step_w *= 0.5
            if step_r < 1e-3 and step_w < 1e-3:
                break

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / ss_tot if ss_tot > 1e-9 else 0.0

    # مرز = جدا شدن از مجانبِ هاله (نه نقطه‌ی عطف)
    # A + (B-A)*f = A + (B-A)/(1+exp(-(r-r0)/w))  ->  r = r0 + w*ln(f/(1-f))
    f = DEPART_FRAC
    depart_r = r0 + w * math.log(f / (1.0 - f))

    return {"A": A, "B": B, "r0": r0, "w": w, "r2": r2, "depart_r": depart_r,
            "sse": sse}


def px_per_mm_per_image(recs):
    """
    همان قاعده‌ی _estimate_px_per_mm_from_disks پایپلاین، آفلاین بازسازی‌شده:
    کوچک‌ترین خوشه‌ی قطرِ پیکسلی (تا ۳۰٪ بزرگ‌تر از کمینه) میانگین‌گیری و برابرِ ۶mm.
    لازم شد چون px_per_mm_est در ماژولِ ۱۶ محاسبه می‌شود که *بعد* از نقطه‌ی
    استخراجِ این پروفایل‌هاست -- ولی داده‌ی لازمش (شعاعِ همه‌ی دیسک‌ها) موجود است،
    پس اجرایِ دوباره‌ی پایپلاین لازم نیست.
    """
    from collections import defaultdict as _dd
    by_img = _dd(list)
    for x in recs:
        by_img[x["image"]].append(2.0 * x["r_disk"])
    out = {}
    for img, diam in by_img.items():
        mn = min(diam)
        cluster = [d for d in diam if d <= mn * 1.30]
        out[img] = float(np.mean(cluster)) / 6.0
    return out


def main():
    recs = json.loads(PROFILES.read_text())
    ppm_map = px_per_mm_per_image(recs)
    for x in recs:
        if not x.get("px_per_mm"):
            x["px_per_mm"] = ppm_map.get(x["image"], 0.0)
    matched = [x for x in recs if x.get("gt_num") is not None]
    print(f"پروفایل‌ها: {len(recs)} کل، {len(matched)} تطبیق‌یافته با مرجع\n")

    results = []
    n_dropped = 0
    for x in matched:
        if not x.get("lawn_mean") or not x.get("px_per_mm"):
            continue
        # باگِ کشف‌شده در اولین اجرا: پروفایل از 1.05*r_disk شروع می‌شود ولی ماسکِ
        # آگار دیسک‌ها را تا 1.35*r_disk کنار گذاشته -- پس چند حلقه‌ی نخست هیچ
        # پیکسلی ندارند و مقدارشان صفر ثبت شده. آن صفرها یک لبه‌ی مصنوعی می‌سازند
        # که برازش دقیقاً رویِ همان می‌نشیند (r0 ≈ 1.33*r_disk در همه‌ی موارد).
        # فقط حلقه‌هایی با پوششِ کافی نگه داشته می‌شوند.
        cnts = np.asarray(x["counts"], dtype=float)
        rc = np.asarray(x["ring_centers"], dtype=float)
        pr = np.asarray(x["profile"], dtype=float)
        ok = cnts >= 20
        if int(np.count_nonzero(ok)) < 8:
            n_dropped += 1
            continue
        first = int(np.argmax(ok))                       # اولین حلقه‌ی معتبر
        last = len(ok) - 1 - int(np.argmax(ok[::-1]))    # آخرین حلقه‌ی معتبر
        rc, pr, cn = rc[first:last + 1], pr[first:last + 1], cnts[first:last + 1]
        keep = cn >= 20
        rc, pr = rc[keep], pr[keep]
        if len(rc) < 8:
            n_dropped += 1
            continue
        fit = fit_logistic(rc, pr, x["lawn_mean"], x["r_disk"])
        if fit is None:
            continue
        ppm = x["px_per_mm"]
        fit.update({
            "image": x["image"], "gt_num": x["gt_num"], "gt_halo": x["gt_halo"],
            "r_disk": x["r_disk"], "ppm": ppm,
            "depart_mm": 2.0 * fit["depart_r"] / ppm,
            "r0_mm": 2.0 * fit["r0"] / ppm,
            "w_over_r0": fit["w"] / max(fit["r0"], 1e-6),
            "amp_over_lawnsd": abs(fit["B"] - fit["A"]) / max(x["lawn_std"], 1e-6),
        })
        results.append(fit)

    if n_dropped:
        print(f"(کنار گذاشته‌شده به‌دلیلِ پوششِ ناکافیِ حلقه‌ها: {n_dropped})\n")
    has = [r for r in results if r["gt_halo"] is not None]
    non = [r for r in results if r["gt_halo"] is None]

    print("=" * 72)
    print("پرسشِ ۱ — آیا سیگموئید اصلاً برازش می‌شود؟")
    print("=" * 72)
    for label, group in [("دارایِ هاله (مرجع)", has), ("بدونِ هاله (مرجع)", non)]:
        if not group:
            continue
        r2s = np.array([g["r2"] for g in group])
        print(f"  {label:<22} n={len(group):<4} R² میانه={np.median(r2s):.3f} "
              f"(صدکِ ۲۵={np.percentile(r2s,25):.3f}, صدکِ ۷۵={np.percentile(r2s,75):.3f})")

    print()
    print("=" * 72)
    print("پرسشِ ۲ — آیا مرزِ حاصل دقیق‌تر از MAE=۳.۸۵ فعلی است؟")
    print("=" * 72)
    for name, key in [("نقطه‌ی جدا شدن (تعریفِ کاربر)", "depart_mm"),
                      ("نقطه‌ی عطف r0 (برایِ مقایسه)", "r0_mm")]:
        errs = [g[key] - g["gt_halo"] for g in has]
        errs = np.array(errs)
        print(f"  {name:<32} MAE={np.mean(np.abs(errs)):6.2f} mm  "
              f"Bias={np.mean(errs):+6.2f} mm  (n={len(errs)})")

    print()
    print("=" * 72)
    print("پرسشِ ۳ — آیا w/r0 مثبت‌هایِ کاذب را جدا می‌کند؟")
    print("=" * 72)
    for name, key in [("w/r0", "w_over_r0"), ("دامنه/انحرافِ‌لَون", "amp_over_lawnsd"),
                      ("R² برازش", "r2")]:
        a = np.array([g[key] for g in has])
        b = np.array([g[key] for g in non])
        if len(b) == 0:
            continue
        pooled = math.sqrt((np.var(a) + np.var(b)) / 2.0)
        d = (np.mean(a) - np.mean(b)) / max(pooled, 1e-9)
        print(f"  {name:<22} هاله: میانه={np.median(a):7.3f}  "
              f"بدونِ‌هاله: میانه={np.median(b):7.3f}  جدایی(Cohen d)={d:+.2f}")

    json.dump(results, open("/tmp/sigmoid_fits.json", "w"))
    print("\nجزئیات: /tmp/sigmoid_fits.json")


if __name__ == "__main__":
    main()

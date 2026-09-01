"""
پ۲ — سنجشِ *پیش‌فرضِ* «تطبیقِ بافتِ خودِ هاله»، پیش از ساختنِ ماژول.

چرا اول پیش‌فرض و بعد ماژول: سه شاخه‌ی قبلی (آماری ۱۵.۸، کانالِ بافت، مدلِ رشد
۱۵.۹) هر سه ساخته شدند، درست کار کردند، و بعد معلوم شد **افزونه**‌اند — روی همان
زیرمجموعه‌ی آسان شلیک می‌کنند. پس این‌بار اول با داده‌ی موجود می‌سنجیم.

ایده‌ی پ۲ در یک جمله: به‌جای یک مرجع (لَون)، **دو** مرجع بسازیم — بافتِ داخلِ
هاله‌ی *همین دیسک* و بافتِ لَونِ میدانِ دور — و هر حلقه را به نزدیک‌ترینِ آن دو
نسبت دهیم. مرز جایی است که تخصیص از «هاله» به «لَون» می‌رود و برنمی‌گردد.

جذابیتِ اصلی: این قاعده **هیچ پارامترِ آزادی ندارد**. نقطه‌ی برش خودش از دو مرجعِ
همان تصویر بیرون می‌آید، نه از عددی که ما تنظیم کرده باشیم. با توجه به این‌که
گیتِ ۴σ در LOO شکست خورد، نبودِ پارامتر یک مزیتِ واقعی است نه یک ظرافتِ زیبایی‌شناختی.

هدفِ سنجش دقیقاً همان‌جایی است که خطا زندگی می‌کند: ۳۳ دیسکی که ادغام به شاخه‌ی
شعاعی سپرده (MAE ~۵.۵mm)، نه ۲۷ دیسکی که شاخه‌های ناحیه‌ای قبلاً خوب حل کرده‌اند.
"""
import csv
import json
import math
from collections import defaultdict

import numpy as np

PROFILES = "/tmp/radial_profiles.json"
BRANCH = "/tmp/branch_compare.csv"

MIN_COUNT = 20          # همان کفِ پوششِ حلقه که در پ۴ تثبیت شد
ZONE_REF_RINGS = 3      # کمینه‌ی ممکن برای یک میانگینِ پایدار


def ppm_map(recs):
    by = defaultdict(list)
    for x in recs:
        by[x["image"]].append(2.0 * x["r_disk"])
    return {k: float(np.mean([d for d in v if d <= min(v) * 1.30])) / 6.0
            for k, v in by.items()}


def clean(x):
    c = np.asarray(x["counts"], float)
    rc = np.asarray(x["ring_centers"], float)
    pr = np.asarray(x["profile"], float)
    ok = c >= MIN_COUNT
    if int(np.count_nonzero(ok)) < 8:
        return None
    f = int(np.argmax(ok))
    l = len(ok) - 1 - int(np.argmax(ok[::-1]))
    rc, pr, c = rc[f:l + 1], pr[f:l + 1], c[f:l + 1]
    k = c >= MIN_COUNT
    return (rc[k], pr[k]) if int(np.count_nonzero(k)) >= 8 else None


def two_reference_boundary(rc, pr, lawn_mean, lawn_sd):
    """مرز = اولین حلقه‌ای که به لَون تخصیص می‌یابد و دیگر برنمی‌گردد.

    مرجعِ هاله: میانگینِ چند حلقه‌ی نخست (نزدیک‌ترین آگارِ موجود به دیسک).
    مرجعِ لَون: میدانِ دور.
    تخصیص: نزدیک‌ترین مرجع بر حسبِ فاصله‌ی مطلق -- یعنی برشِ نقطه‌ی میانی، که
    خودش از داده می‌آید و پارامترِ آزاد نیست.
    """
    if len(pr) < ZONE_REF_RINGS + 2:
        return None
    zone = float(np.mean(pr[:ZONE_REF_RINGS]))
    sep = abs(lawn_mean - zone)
    # نگهبانِ دیسکِ مقاوم: اگر «هاله» و لَون از هم تفکیک‌پذیر نباشند، دو مرجع یکی‌اند
    # و قاعده بی‌معنی می‌شود. آستانه همان اندازه‌اثرِ ۱ (یک انحرافِ معیارِ لَون) است
    # که در ماژولِ ۱۵.۷ هم به‌کار می‌رود -- پارامترِ جدیدی معرفی نمی‌شود.
    if lawn_sd <= 1e-6 or sep < lawn_sd:
        return None
    mid = 0.5 * (zone + lawn_mean)
    lawnish = (pr > mid) if lawn_mean > zone else (pr < mid)
    # پایداری: از انتها به عقب بیا تا آخرین حلقه‌ای که «هاله» است
    idx = None
    for i in range(len(lawnish)):
        if lawnish[i:].all():
            idx = i
            break
    if idx is None or idx == 0:
        return None
    # درون‌یابیِ خطی بینِ دو حلقه‌ی طرفینِ برش -- زیرحلقه‌ای، بدونِ پارامتر
    y0, y1 = pr[idx - 1], pr[idx]
    if abs(y1 - y0) < 1e-9:
        return float(rc[idx])
    t = (mid - y0) / (y1 - y0)
    return float(rc[idx - 1] + t * (rc[idx] - rc[idx - 1]))


def main():
    recs = json.load(open(PROFILES))
    pm = ppm_map(recs)
    branch = {}
    for r in csv.DictReader(open(BRANCH)):
        if r["gt_num"]:
            branch[(r["image"], int(float(r["gt_num"])))] = r

    rows = []
    for x in recs:
        key = (x["image"], x["gt_num"]) if x.get("gt_num") is not None else None
        if key is None or key not in branch or not x.get("lawn_mean"):
            continue
        cl = clean(x)
        if cl is None:
            continue
        rc, pr = cl
        b = branch[key]
        r0 = two_reference_boundary(rc, pr, x["lawn_mean"], x.get("lawn_std") or 0.0)
        p = pm[x["image"]]
        rows.append({
            "image": x["image"], "gt": x["gt_halo"],
            "src": b["fusion_source"],
            "fused_mm": float(b["radial_mm"]),
            "p2_mm": (2.0 * r0 / p) if r0 else 0.0,
        })

    def report(title, sel):
        s = [r for r in rows if sel(r)]
        wgt = [r for r in s if r["gt"] is not None]
        fired = [r for r in wgt if r["p2_mm"] > 0.01]
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        print(f"  تعداد دیسک: {len(s)}   با هاله‌ی مرجع: {len(wgt)}   "
              f"پ۲ شلیک کرد: {len(fired)}")
        if not fired:
            print("  پ۲ روی هیچ‌کدام مرز نداد.")
            return
        ef = np.array([abs(r["fused_mm"] - r["gt"]) for r in fired])
        ep = np.array([abs(r["p2_mm"] - r["gt"]) for r in fired])
        n = len(ep)
        w = int(np.sum(ep < ef))
        z = (w - n / 2) / math.sqrt(n * 0.25)
        d = ef - ep
        t = d.mean() / (d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        print(f"  روی همان {n} دیسکی که پ۲ شلیک کرد:")
        print(f"    خروجیِ فعلی : MAE {ef.mean():6.2f}   میانه {np.median(ef):5.2f}")
        print(f"    پ۲          : MAE {ep.mean():6.2f}   میانه {np.median(ep):5.2f}")
        print(f"    پ۲ بهتر در {w} از {n}  ->  z={z:+.2f}   "
              f"بهبودِ میانگین={d.mean():+.2f} mm (t={t:+.2f})")

    # مثبتِ کاذب: دیسکِ بدونِ هاله‌ی مرجع که پ۲ برایش مرز می‌دهد
    nogt = [r for r in rows if r["gt"] is None]
    fp = [r for r in nogt if r["p2_mm"] > 0.01]

    report("همه‌ی دیسک‌ها", lambda r: True)
    report("زیرمجموعه‌ی بحرانی — ادغام به شاخه‌ی شعاعی سپرده (بیشترین خطا)",
           lambda r: r["src"] == "radial")
    report("زیرمجموعه‌ی آسان — شاخه‌ی ناحیه‌ای قبلاً حل کرده",
           lambda r: r["src"] in ("otsu", "watershed"))

    print("\n" + "=" * 72)
    print("رفتار روی دیسک‌هایِ بدونِ هاله (مقاوم)")
    print("=" * 72)
    print(f"  {len(nogt)} دیسکِ بدونِ هاله‌ی مرجع، پ۲ برای {len(fp)} تا مرز داد "
          f"(مثبتِ کاذبِ بالقوه)")


if __name__ == "__main__":
    main()

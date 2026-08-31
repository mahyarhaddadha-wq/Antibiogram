"""
آزمونِ صورت‌بندیِ LCCDE (پیشنهادِ کاربر).

دو فرضیه‌ی آزمایش‌پذیر که از این صورت‌بندی بیرون می‌آید:

(الف) در حالتِ دائمی، معادله‌ی نفوذِ شعاعی  r·C'' + C' = 0  ضریب‌ثابت نیست، ولی با
      u = ln(r) به  d²C/du² = 0  تبدیل می‌شود که هست. پس برازشِ سیگموئید در فضایِ
      ln(r) باید *بهتر* از فضایِ r باشد -- اگر فیزیکِ نفوذ واقعاً حاکم باشد.

(ب) فاصله‌ی «خوانشِ کارشناس» تا «حقیقتِ زیستی» نباید یک ثابت باشد، بلکه باید
      متناسب با پهنایِ گذار w باشد:  δ = k·w
      چون اگر کارشناس روی تیزترین نقطه (نقطه‌ی عطف) ببُرد و حقیقتِ زیستی نقطه‌ی
      جدا شدن باشد، آنگاه بنا به خودِ لجستیک  r_عطف − r_جداشدن = w·|ln(f/(1−f))|.
      این یعنی هرچه لبه تیزتر، اختلافِ خوانشِ انسانی با زیست‌شناسی کمتر -- که
      قابلِ‌سنجش است.
"""
import json
import math

import numpy as np

PROFILES = "/tmp/radial_profiles.json"


def fit_logistic(r, y, B, r_disk, log_space):
    """همان برازنده، ولی اختیاراً در فضایِ ln(r)."""
    r = np.asarray(r, dtype=float)
    y = np.asarray(y, dtype=float)
    x = np.log(r) if log_space else r
    if len(x) < 8:
        return None
    span = x[-1] - x[0]
    x0_grid = np.linspace(x[0], x[-1], 60)
    if log_space:
        w_grid = np.exp(np.linspace(math.log(span / 200.0), math.log(span), 40))
    else:
        w_grid = np.exp(np.linspace(math.log(0.05 * r_disk), math.log(3.0 * r_disk), 40))

    best = None
    for w in w_grid:
        for x0 in x0_grid:
            s = 1.0 / (1.0 + np.exp(-np.clip((x - x0) / w, -60, 60)))
            den = float(np.sum((1 - s) ** 2))
            if den < 1e-9:
                continue
            A = float(np.sum((y - B * s) * (1 - s)) / den)
            sse = float(np.sum((y - (A + (B - A) * s)) ** 2))
            if best is None or sse < best[0]:
                best = (sse, A, x0, w)
    if best is None:
        return None
    sse, A, x0, w = best
    sx, sw = span / 60.0, w * 0.5
    for _ in range(40):
        improved = False
        for dx, dw in [(sx, 0), (-sx, 0), (0, sw), (0, -sw)]:
            xn, wn = x0 + dx, max(w + dw, 1e-4)
            s = 1.0 / (1.0 + np.exp(-np.clip((x - xn) / wn, -60, 60)))
            den = float(np.sum((1 - s) ** 2))
            if den < 1e-9:
                continue
            An = float(np.sum((y - B * s) * (1 - s)) / den)
            e = float(np.sum((y - (An + (B - An) * s)) ** 2))
            if e < sse:
                sse, A, x0, w = e, An, xn, wn
                improved = True
        if not improved:
            sx *= 0.5
            sw *= 0.5
            if sx < 1e-5 and sw < 1e-5:
                break
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 1e-9 else 0.0
    # نقطه‌ی عطف در فضایِ اصلی
    r0 = math.exp(x0) if log_space else x0
    # پهنایِ گذار بر حسبِ پیکسل (در فضایِ لگاریتمی، w نسبی است -> به px تبدیل می‌شود)
    w_px = r0 * w if log_space else w
    return {"A": A, "B": B, "r0": r0, "w_px": w_px, "r2": r2}


def px_per_mm(recs):
    from collections import defaultdict
    by = defaultdict(list)
    for x in recs:
        by[x["image"]].append(2.0 * x["r_disk"])
    return {k: float(np.mean([d for d in v if d <= min(v) * 1.30])) / 6.0
            for k, v in by.items()}


def clean(x):
    c = np.asarray(x["counts"], float)
    rc = np.asarray(x["ring_centers"], float)
    pr = np.asarray(x["profile"], float)
    ok = c >= 20
    if int(np.count_nonzero(ok)) < 8:
        return None
    f = int(np.argmax(ok)); l = len(ok) - 1 - int(np.argmax(ok[::-1]))
    rc, pr, c = rc[f:l + 1], pr[f:l + 1], c[f:l + 1]
    k = c >= 20
    return (rc[k], pr[k]) if int(np.count_nonzero(k)) >= 8 else None


def main():
    recs = json.load(open(PROFILES))
    ppm = px_per_mm(recs)
    rows = []
    for x in recs:
        if x.get("gt_num") is None or not x.get("lawn_mean"):
            continue
        cl = clean(x)
        if cl is None:
            continue
        rc, pr = cl
        lin = fit_logistic(rc, pr, x["lawn_mean"], x["r_disk"], False)
        log = fit_logistic(rc, pr, x["lawn_mean"], x["r_disk"], True)
        if lin is None or log is None:
            continue
        p = ppm[x["image"]]
        rows.append({"image": x["image"], "gt_num": x["gt_num"], "gt": x["gt_halo"],
                     "lin_r2": lin["r2"], "log_r2": log["r2"],
                     "lin_mm": 2 * lin["r0"] / p, "log_mm": 2 * log["r0"] / p,
                     "lin_w": lin["w_px"], "log_w": log["w_px"], "ppm": p})

    has = [r for r in rows if r["gt"] is not None]
    print("=" * 74)
    print("فرضیه (الف): آیا برازش در فضایِ ln(r) بهتر است؟")
    print("=" * 74)
    a = np.array([r["lin_r2"] for r in rows]); b = np.array([r["log_r2"] for r in rows])
    print(f"  R² میانه — فضایِ r: {np.median(a):.3f}   فضایِ ln(r): {np.median(b):.3f}")
    w = int(np.sum(b > a)); n = len(rows)
    print(f"  ln(r) در {w} از {n} مورد برازشِ بهتر  ->  z={(w - n/2)/math.sqrt(n*0.25):+.2f}")
    ea = np.array([abs(r["lin_mm"] - r["gt"]) for r in has])
    eb = np.array([abs(r["log_mm"] - r["gt"]) for r in has])
    print(f"\n  MAE مرز — فضایِ r: {ea.mean():.2f} mm   فضایِ ln(r): {eb.mean():.2f} mm")
    print(f"  میانه — فضایِ r: {np.median(ea):.2f} mm   فضایِ ln(r): {np.median(eb):.2f} mm")
    wl = int(np.sum(eb < ea))
    print(f"  ln(r) در {wl} از {len(has)} مورد دقیق‌تر  ->  z={(wl - len(has)/2)/math.sqrt(len(has)*0.25):+.2f}")

    print()
    print("=" * 74)
    print("فرضیه (ب): آیا فاصله‌ی خوانشِ انسانی تا زیست‌شناسی متناسب با w است؟")
    print("=" * 74)
    # اگر کارشناس روی نقطه‌ی عطف ببرد و حقیقتِ زیستی نقطه‌ی ۵٪ باشد:
    K = abs(math.log(0.05 / 0.95))     # = 2.944
    wpx = np.array([r["lin_w"] for r in has])
    ppm_a = np.array([r["ppm"] for r in has])
    delta_mm = 2.0 * K * wpx / ppm_a   # اختلافِ قطر (mm)
    print(f"  δ = 2·k·w/ppm  با k=ln(0.95/0.05)={K:.3f}")
    print(f"  δ میانه={np.median(delta_mm):.2f} mm   میانگین={delta_mm.mean():.2f} mm   "
          f"بیشینه={delta_mm.max():.1f} mm")
    print(f"  δ < 2mm در {100*np.mean(delta_mm<2):.0f}% موارد،  "
          f"δ > 10mm در {100*np.mean(delta_mm>10):.0f}% موارد")
    sharp = delta_mm < np.median(delta_mm)
    print(f"\n  رویِ لبه‌هایِ تیز (δ زیرِ میانه):   MAE مرز = {ea[sharp].mean():.2f} mm")
    print(f"  رویِ لبه‌هایِ پهن (δ بالایِ میانه): MAE مرز = {ea[~sharp].mean():.2f} mm")
    json.dump(rows, open("/tmp/lccde_test.json", "w"))


if __name__ == "__main__":
    main()

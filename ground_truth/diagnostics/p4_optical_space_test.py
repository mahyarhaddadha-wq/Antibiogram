"""
آزمونِ تغییرِ فضایِ معادله (بندِ ۶ نقشه‌ی راه).

فرضیه: شدتِ خامِ پیکسل با چگالیِ سلولی رابطه‌ی خطی ندارد. اگر به فضایی برویم که
رابطه در آن خطی شود، برازشِ سیگموئید باید بهتر و مرز دقیق‌تر شود.

سه فضا مقایسه می‌شوند:
  raw : خودِ شدت (خطِ پایه‌ی فعلی)
  BL  : بیر–لامبرت،  OD = -ln(I / I_lawn)      -- فیزیکِ عبورِ نور
  KM  : کوبلکا–مونک، K/S = (1-R)^2 / (2R)      -- فیزیکِ بازتاب (حالتِ فعلیِ ما)

نکته: تصاویرِ ما بازتابی‌اند (لَون به‌خاطرِ پراکندگی روشن‌تر است)، پس از نظرِ فیزیکی
KM تبدیلِ درست‌تر است و BL فقط با نورِ پس‌زمینه‌ای دقیق می‌شود.
"""
import json
import math

import numpy as np

PROFILES = "/tmp/radial_profiles.json"
EPS = 1e-3


def fit(r, y, B, r_disk):
    """برازشِ لجستیک با B ثابت و A خطی -- همان برازنده‌ی ماژولِ ۱۵.۹."""
    r = np.asarray(r, float); y = np.asarray(y, float)
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
            sr *= 0.5; sw *= 0.5
            if sr < 1e-4 and sw < 1e-4:
                break
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return {"r0": r0, "w": w, "r2": 1.0 - sse / sst if sst > 1e-9 else 0.0}


def to_bl(y, lawn):
    """بیر–لامبرت: چگالیِ نوری نسبت به سطحِ لَون."""
    R = np.clip(np.asarray(y, float) / max(lawn, EPS), EPS, 10.0)
    return -np.log(R)


def to_km(y, lawn):
    """کوبلکا–مونک: (1-R)^2/(2R) با R نسبت به سطحِ لَون."""
    R = np.clip(np.asarray(y, float) / max(lawn, EPS), EPS, 1.0 - EPS)
    return (1.0 - R) ** 2 / (2.0 * R)


def ppm_map(recs):
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
    pm = ppm_map(recs)
    out = []
    for x in recs:
        if x.get("gt_num") is None or not x.get("lawn_mean"):
            continue
        cl = clean(x)
        if cl is None:
            continue
        rc, pr = cl
        lawn = x["lawn_mean"]
        rd, p = x["r_disk"], pm[x["image"]]

        f_raw = fit(rc, pr, lawn, rd)
        # در فضایِ تبدیل‌شده، سطحِ لَون به‌طورِ ساختاری صفر می‌شود
        f_bl = fit(rc, to_bl(pr, lawn), 0.0, rd)
        f_km = fit(rc, to_km(pr, lawn), 0.0, rd)
        if not (f_raw and f_bl and f_km):
            continue
        out.append({"image": x["image"], "gt": x["gt_halo"], "ppm": p,
                    "raw_mm": 2 * f_raw["r0"] / p, "raw_r2": f_raw["r2"],
                    "bl_mm": 2 * f_bl["r0"] / p, "bl_r2": f_bl["r2"],
                    "km_mm": 2 * f_km["r0"] / p, "km_r2": f_km["r2"]})

    has = [o for o in out if o["gt"] is not None]
    print(f"n={len(out)} پروفایل، {len(has)} با هاله‌ی مرجع\n")
    print("=" * 68)
    print("کیفیتِ برازش و دقتِ مرز در سه فضا")
    print("=" * 68)
    print(f"{'فضا':<28}{'R² میانه':>10}{'MAE':>9}{'میانه':>9}")
    for name, rk, mk in [("raw (شدتِ خام، خطِ پایه)", "raw_r2", "raw_mm"),
                         ("BL (بیر–لامبرت)", "bl_r2", "bl_mm"),
                         ("KM (کوبلکا–مونک)", "km_r2", "km_mm")]:
        r2 = np.median([o[rk] for o in out])
        e = np.array([abs(o[mk] - o["gt"]) for o in has])
        print(f"{name:<28}{r2:>10.3f}{e.mean():>9.2f}{np.median(e):>9.2f}")

    print()
    print("=" * 68)
    print("مقایسه‌ی جفتی با خطِ پایه (آزمونِ علامت)")
    print("=" * 68)
    er = np.array([abs(o["raw_mm"] - o["gt"]) for o in has])
    for name, mk in [("BL", "bl_mm"), ("KM", "km_mm")]:
        e = np.array([abs(o[mk] - o["gt"]) for o in has])
        w = int(np.sum(e < er)); n = len(e)
        z = (w - n / 2) / math.sqrt(n * 0.25)
        d = er - e
        t = d.mean() / (d.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        print(f"  {name}: بهتر در {w} از {n}  ->  z={z:+.2f}   "
              f"اختلافِ میانگین={d.mean():+.2f} mm (t={t:+.2f})")
    json.dump(out, open("/tmp/optical_space.json", "w"))


if __name__ == "__main__":
    main()

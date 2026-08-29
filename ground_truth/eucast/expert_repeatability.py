"""
اثرِ تکرارپذیریِ خودِ کارشناس (±۲ میلی‌متر) بر تفسیرِ همه‌ی اعدادِ پروژه.

کارشناسِ آزمایشگاه اعلام کرده اندازه‌گیریِ کارشناسانِ آن‌ها «از همه لحاظ ±۲ میلی‌متر
تغییر دارد». این عدد سه چیز را هم‌زمان عوض می‌کند و هر سه این‌جا حساب می‌شود:

  ۱) داده‌ی مرجعِ ما خودش نویز دارد، پس بخشی از MAE=۳.۸۵ ما مالِ ما نیست.
  ۲) هدفِ «MAE ≈ ۱.۰mm» که از سقف‌هایِ EUCAST درآمد، باید با تکرارپذیریِ انسانی
     مقایسه شود -- شاید اصلاً زیرِ سطحی باشد که با این مرجع قابلِ *اثبات* نیست.
  ۳) توافقِ دسته‌ایِ ۸۲.۵٪ ما در برابرِ چه چیزی باید سنجیده شود؟ سقفِ واقعی، توافقِ
     دو کارشناس با **یکدیگر** است، نه ۱۰۰٪.

## ابهامِ «±۲ میلی‌متر» -- و چرا سه تفسیر حساب می‌شود

«±۲» می‌تواند یعنی بازه‌ی کاملِ تغییرات، یا فاصله‌ی اطمینانِ ۹۵٪، یا یک انحرافِ
معیار. هر سه تفسیر گزارش می‌شود تا نتیجه به حدسِ ما وابسته نباشد:

    یکنواخت روی [−۲,+۲]        انحرافِ معیار ≈ ۱.۱۵ mm
    نرمال با بازه‌ی ۹۵٪ = ±۲   انحرافِ معیار ≈ ۱.۰۲ mm
    نرمال با انحرافِ معیار = ۲  انحرافِ معیار = ۲.۰۰ mm

## قطعیت

پروژه قطعی است، پس شبیه‌سازی با بذرِ ثابت اجرا می‌شود و هر بار عینِ همان عدد را
می‌دهد. تعدادِ تکرار به‌قدری بالاست که رقمِ اعشارِ گزارش‌شده پایدار باشد.
"""
import csv
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "ground_truth" / "eucast" / "eucast_v16_zone_breakpoints.csv"
PAIRS = REPO / "ground_truth" / "diagnostics" / "halo_branch_comparison.csv"
OUT = REPO / "ground_truth" / "eucast" / "expert_repeatability.csv"

NO_ZONE_MM = 6.0
SEED = 20260829
N_REP = 400
VME_LIMIT, ME_LIMIT = 1.5, 3.0

# سه تفسیر از «±۲ میلی‌متر»
MODELS = [
    ("یکنواخت روی [−۲,+۲]", "uniform", 2.0, 2.0 / math.sqrt(3.0)),
    ("نرمال، بازه‌ی ۹۵٪ = ±۲", "normal", 2.0 / 1.96, 2.0 / 1.96),
    ("نرمال، انحرافِ معیار = ۲", "normal", 2.0, 2.0),
]


def classify(d, s, r):
    return "S" if d >= s else ("R" if d < r else "I")


def err_type(ref, test):
    if ref == test:
        return "CA"
    if ref == "R" and test == "S":
        return "VME"
    if ref == "S" and test == "R":
        return "ME"
    return "mE"


def load_breakpoints():
    out = []
    for r in csv.DictReader(open(BP, encoding="utf-8")):
        try:
            s, v = float(r["zone_S_ge_mm"]), float(r["zone_R_lt_mm"])
        except ValueError:
            continue
        if s >= 50 or v >= 50 or s < v:
            continue
        out.append((s, v))
    return out


def load_pairs():
    out = []
    for r in csv.DictReader(open(PAIRS, encoding="utf-8")):
        if not r["gt_num"]:
            continue
        gt = r["gt_halo"].strip()
        sysv = float(r["radial_mm"])
        out.append({"ref": float(gt) if gt else NO_ZONE_MM,
                    "sys": sysv if sysv > 0.01 else NO_ZONE_MM,
                    "has_zone": bool(gt), "source": r["fusion_source"]})
    return out


def agreement(ref_arr, test_arr, bps):
    """نرخِ CA/VME/ME رویِ همه‌ی جفت‌هایِ (دیسک × نقطه‌ی شکست)."""
    c = {"CA": 0, "VME": 0, "ME": 0, "mE": 0}
    for s, v in bps:
        rc = np.where(ref_arr >= s, 2, np.where(ref_arr < v, 0, 1))
        tc = np.where(test_arr >= s, 2, np.where(test_arr < v, 0, 1))
        c["CA"] += int(np.sum(rc == tc))
        c["VME"] += int(np.sum((rc == 0) & (tc == 2)))
        c["ME"] += int(np.sum((rc == 2) & (tc == 0)))
        c["mE"] += int(np.sum((rc != tc) & ~(((rc == 0) & (tc == 2)) |
                                             ((rc == 2) & (tc == 0)))))
    n = sum(c.values())
    return {k: 100.0 * v / n for k, v in c.items()}


def draw(rng, kind, scale, n):
    return (rng.uniform(-scale, scale, n) if kind == "uniform"
            else rng.normal(0.0, scale, n))


def main():
    bps = load_breakpoints()
    pairs = load_pairs()
    ref = np.array([p["ref"] for p in pairs], dtype=float)
    sysv = np.array([p["sys"] for p in pairs], dtype=float)
    n = len(pairs)

    # ── ۱) خطایِ سیستم پس از کسرِ نویزِ مرجع ─────────────────────────────
    zone = np.array([p["has_zone"] and p["sys"] > NO_ZONE_MM for p in pairs])
    diff = (sysv - ref)[zone]
    mae_obs = float(np.mean(np.abs(diff)))
    sd_obs = float(np.std(diff, ddof=1))

    print("=" * 74)
    print("۱) چقدر از خطایِ مشاهده‌شده مالِ خودِ سیستم است؟")
    print("=" * 74)
    print("  اگر خطایِ سیستم و نویزِ مرجع مستقل باشند، واریانس‌ها جمع می‌شوند:")
    print("      sd_مشاهده² = sd_سیستم² + sd_مرجع²")
    print(f"\n  MAE مشاهده‌شده = {mae_obs:.2f} mm   (sd = {sd_obs:.2f} mm، n={int(zone.sum())})\n")
    print(f"  {'تفسیرِ ±۲mm':<26}{'sd مرجع':>9}{'sd سیستم':>10}{'MAE سیستم':>11}{'کاهش':>8}")
    for name, _, _, sd_ref in MODELS:
        v = sd_obs ** 2 - sd_ref ** 2
        sd_sys = math.sqrt(v) if v > 0 else 0.0
        mae_sys = mae_obs * (sd_sys / sd_obs) if sd_obs > 0 else 0.0
        print(f"  {name:<26}{sd_ref:>9.2f}{sd_sys:>10.2f}{mae_sys:>11.2f}"
              f"{mae_obs - mae_sys:>8.2f}")
    print("\n  یعنی حتی با بدبینانه‌ترین تفسیر، بیشترِ خطا مالِ خودِ سیستم است.")
    print("  ±۲mmیِ کارشناس، MAE=۳.۸۵ را توجیه نمی‌کند.")

    # ── ۲) دو کارشناس با هم چقدر توافق دارند؟ ────────────────────────────
    print()
    print("=" * 74)
    print("۲) سقفِ واقعی: دو کارشناس با **یکدیگر** چقدر توافق دارند؟")
    print("=" * 74)
    print("  کارشناسِ دوم شبیه‌سازی می‌شود: خوانشِ همان دیسک + نویزِ ±۲mm.")
    print("  این «سقفِ قابلِ‌دستیابی» است -- هیچ روشی نمی‌تواند از توافقِ خودِ")
    print("  انسان‌ها با هم بهتر *اثبات* شود، چون مرجعش همان انسان است.\n")
    print(f"  {'تفسیرِ ±۲mm':<26}{'MAE انسان-انسان':>17}{'CA':>8}{'VME':>8}{'ME':>8}")

    rows = []
    sys_ag = agreement(ref, sysv, bps)
    for name, kind, scale, sd_ref in MODELS:
        rng = np.random.default_rng(SEED)
        acc = {"CA": 0.0, "VME": 0.0, "ME": 0.0, "mE": 0.0}
        maes = []
        for _ in range(N_REP):
            a = ref + draw(rng, kind, scale, n)
            b = ref + draw(rng, kind, scale, n)
            # کارشناس تا نزدیک‌ترین میلی‌متر می‌خواند، و زیرِ قطرِ دیسک بی‌معنی است
            a = np.maximum(np.round(a), NO_ZONE_MM)
            b = np.maximum(np.round(b), NO_ZONE_MM)
            g = agreement(a, b, bps)
            for k in acc:
                acc[k] += g[k] / N_REP
            maes.append(float(np.mean(np.abs(a - b)[zone])))
        print(f"  {name:<26}{np.mean(maes):>17.2f}{acc['CA']:>7.1f}%"
              f"{acc['VME']:>7.2f}%{acc['ME']:>7.2f}%")
        rows.append({"model": name, "sd_ref_mm": round(sd_ref, 3),
                     "human_human_mae_mm": round(float(np.mean(maes)), 3),
                     "human_CA_pct": round(acc["CA"], 2),
                     "human_VME_pct": round(acc["VME"], 2),
                     "human_ME_pct": round(acc["ME"], 2),
                     "system_CA_pct": round(sys_ag["CA"], 2),
                     "system_VME_pct": round(sys_ag["VME"], 2),
                     "system_ME_pct": round(sys_ag["ME"], 2)})

    print(f"\n  {'سیستم در برابرِ کارشناس':<26}{mae_obs:>17.2f}"
          f"{sys_ag['CA']:>7.1f}%{sys_ag['VME']:>7.2f}%{sys_ag['ME']:>7.2f}%")

    # ── ۳) آیا خودِ کارشناس سقفِ EUCAST را برآورده می‌کند؟ ────────────────
    print()
    print("=" * 74)
    print("۳) نتیجه‌ی تعیین‌کننده")
    print("=" * 74)
    worst = max(rows, key=lambda r: r["human_VME_pct"])
    best = min(rows, key=lambda r: r["human_VME_pct"])
    print(f"  VMEیِ انسان-انسان بینِ {best['human_VME_pct']:.2f}٪ و "
          f"{worst['human_VME_pct']:.2f}٪ است (سقفِ مرسوم {VME_LIMIT}٪).")
    if worst["human_VME_pct"] > VME_LIMIT:
        print("  یعنی با ±۲mm، **حتی دو کارشناسِ انسانی هم** در بخشی از تفسیرها")
        print("  سقفِ VME را برآورده نمی‌کنند. پس آن سقف‌ها برایِ یک روشِ جدید در")
        print("  برابرِ *این* مرجع، معیارِ منصفانه‌ای نیستند.")
    print()
    print("  و درباره‌ی هدفِ MAE ≈ ۱.۰mm که از سقف‌هایِ EUCAST درآمد:")
    for r in rows:
        rel = "زیرِ" if 1.0 < r["human_human_mae_mm"] else "بالایِ"
        print(f"    {r['model']:<26} MAE انسان-انسان {r['human_human_mae_mm']:.2f} "
              f"-> هدفِ ۱.۰ {rel} تکرارپذیریِ انسانی است")
    print("\n  هدف دست‌نیافتنی نیست (دستگاه می‌تواند از انسان *باثبات‌تر* باشد)،")
    print("  ولی با این داده‌ی مرجع قابلِ **اثبات** نیست: نمی‌شود دقتی را نشان داد")
    print("  که از نویزِ خودِ خط‌کشِ سنجش کوچک‌تر است.")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nجزئیات: {OUT}")


if __name__ == "__main__":
    main()

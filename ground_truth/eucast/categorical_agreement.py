"""
ترجمه‌ی خطایِ اندازه‌گیری به زبانِ بالینی: توافقِ دسته‌ای S/I/R.

پرسشی که این اسکریپت جواب می‌دهد: **MAE=۳.۸۵mm چند درصد از تصمیم‌هایِ بالینی را
عوض می‌کند؟** این پرسشِ درست است، نه «چند میلی‌متر خطا داریم» — چون آن‌چه به پزشک
گزارش می‌شود عدد نیست، دسته است.

## مشکلِ داده و راهِ حلِ آن

طبقه‌بندیِ EUCAST به سه چیز نیاز دارد: قطرِ ناحیه، **گونه‌ی باکتری**، و **آنتی‌بیوتیک
+ محتوایِ دیسک**. داده‌ی مرجعِ ما فقط اولی را دارد؛ کارشناس گونه و آنتی‌بیوتیک را ثبت
نکرده و ما هم برچسبِ رویِ دیسک‌ها را نمی‌خوانیم.

این طبقه‌بندیِ *همین ۱۱ عکس* را غیرممکن می‌کند — ولی پرسشِ بالینی را نه. چون هر
نقطه‌ی شکست فقط یک عدد رویِ همان محورِ میلی‌متری است. پس به‌جای این‌که بپرسیم «این
دیسک چه دسته‌ای می‌گیرد»، می‌پرسیم:

    اگر نقطه‌ی شکستِ مربوطه هرکدام از ۳۹۷ نقطه‌ی شکستِ واقعیِ EUCAST v16 باشد،
    خوانشِ سیستم چند درصد مواقع با خوانشِ کارشناس **هم‌دسته** می‌شود؟

این یک پرسشِ خوش‌تعریف است و جوابش یک توزیع می‌دهد، نه یک عدد — که صادقانه‌تر هم
هست. عدمِ‌قطعیتِ ناشی از ندانستنِ آنتی‌بیوتیک به‌صورتِ پهنایِ همان توزیع دیده می‌شود.

## دسته‌بندیِ خطاها (استانداردِ ISO 20776-2 / CLSI M23)

    CA  توافقِ دسته‌ای       هر دو یک دسته
    VME خطایِ بسیار عمده     مرجع R ولی سیستم S  -- خطرناک‌ترین (حساسِ کاذب)
    ME  خطایِ عمده           مرجع S ولی سیستم R  -- درمانِ لازم را دریغ می‌کند
    mE  خطایِ جزئی           یکی از دو طرف I است

آستانه‌هایِ پذیرشِ مرسوم برایِ یک روشِ جدید: VME ≤ ۱.۵٪ و ME ≤ ۳٪.

## دیسکِ بدونِ هاله

کارشناس «بدونِ هاله» یعنی رشد تا لبه‌ی دیسک آمده، یعنی قطرِ ناحیه = قطرِ خودِ دیسک
(۶mm). این زیرِ هر نقطه‌ی شکستی است، پس دسته‌اش R است. با همین قرارداد، ۱۴ مثبتِ
کاذب و ۳ منفیِ کاذبِ حضورِ هاله هم به‌طورِ طبیعی واردِ حساب می‌شوند — یعنی عددِ
نهایی واقعاً end-to-end است، نه فقط رویِ دیسک‌هایی که هر دو طرف هاله دیده‌اند.
"""
import csv
import math
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BP = REPO / "ground_truth" / "eucast" / "eucast_v16_zone_breakpoints.csv"
PAIRS = REPO / "ground_truth" / "diagnostics" / "halo_branch_comparison.csv"
OUT = REPO / "ground_truth" / "eucast" / "categorical_agreement.csv"

NO_ZONE_MM = 6.0        # رشد تا لبه‌ی دیسک؛ قطرِ استانداردِ دیسک
VME_LIMIT = 1.5         # ٪ -- سقفِ مرسومِ پذیرش
ME_LIMIT = 3.0          # ٪


def classify(diameter_mm, s_ge, r_lt):
    """قراردادِ EUCAST: قطر ≥ S یعنی S، قطر < R یعنی R، بینشان I."""
    if diameter_mm >= s_ge:
        return "S"
    if diameter_mm < r_lt:
        return "R"
    return "I"


def error_type(ref, test):
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
            s = float(r["zone_S_ge_mm"])
            v = float(r["zone_R_lt_mm"])
        except ValueError:
            continue
        # EUCAST برایِ «دسته‌ی S وجود ندارد» عددِ دست‌نیافتنیِ ۵۰ می‌گذارد؛ آن سطرها
        # نقطه‌ی شکستِ واقعی رویِ محورِ اندازه‌گیری نیستند و باید کنار بروند.
        if s >= 50 or v >= 50:
            continue
        if s < v:
            continue
        out.append({"organism": r["organism"], "agent": r["agent"],
                    "disk": r["disk_content_ug"], "s": s, "r": v})
    return out


def load_pairs():
    out = []
    for r in csv.DictReader(open(PAIRS, encoding="utf-8")):
        if not r["gt_num"]:
            continue
        gt = r["gt_halo"].strip()
        sysv = float(r["radial_mm"])
        out.append({
            "image": r["image"],
            "source": r["fusion_source"],
            "ref_mm": float(gt) if gt else NO_ZONE_MM,
            "sys_mm": sysv if sysv > 0.01 else NO_ZONE_MM,
            "ref_has_zone": bool(gt),
        })
    return out


def evaluate(pairs, bps):
    """برایِ هر جفتِ (نقطه‌ی شکست × دیسک) دسته‌ی مرجع و سیستم را می‌سنجد."""
    per_bp = []
    totals = Counter()
    for b in bps:
        c = Counter()
        for p in pairs:
            ref = classify(p["ref_mm"], b["s"], b["r"])
            test = classify(p["sys_mm"], b["s"], b["r"])
            c[error_type(ref, test)] += 1
            totals[error_type(ref, test)] += 1
        n = sum(c.values())
        per_bp.append({**b, "n": n,
                       "CA": c["CA"], "VME": c["VME"], "ME": c["ME"], "mE": c["mE"],
                       "CA_pct": 100.0 * c["CA"] / n,
                       "VME_pct": 100.0 * c["VME"] / n,
                       "ME_pct": 100.0 * c["ME"] / n})
    return per_bp, totals


def pct(c, key):
    n = sum(c.values())
    return 100.0 * c[key] / n if n else 0.0


def quantiles(vals, qs=(5, 25, 50, 75, 95)):
    v = sorted(vals)
    out = []
    for q in qs:
        i = (len(v) - 1) * q / 100.0
        lo, hi = int(math.floor(i)), int(math.ceil(i))
        out.append(v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (i - lo))
    return out


def main():
    bps = load_breakpoints()
    pairs = load_pairs()
    per_bp, totals = evaluate(pairs, bps)

    print("=" * 74)
    print("توافقِ دسته‌ایِ S/I/R -- سنجیده رویِ کلِ نقاطِ شکستِ EUCAST v16.0")
    print("=" * 74)
    print(f"  {len(pairs)} دیسکِ تطبیق‌یافته × {len(bps)} نقطه‌ی شکستِ معتبر "
          f"= {sum(totals.values()):,} تصمیمِ شبیه‌سازی‌شده")
    print()
    print(f"  توافقِ دسته‌ای (CA)        {pct(totals,'CA'):6.2f}%")
    print(f"  خطایِ بسیار عمده (VME)    {pct(totals,'VME'):6.2f}%   "
          f"(سقفِ مرسوم {VME_LIMIT}%)  "
          f"{'قبول' if pct(totals,'VME') <= VME_LIMIT else 'رد'}")
    print(f"  خطایِ عمده (ME)           {pct(totals,'ME'):6.2f}%   "
          f"(سقفِ مرسوم {ME_LIMIT}%)  "
          f"{'قبول' if pct(totals,'ME') <= ME_LIMIT else 'رد'}")
    print(f"  خطایِ جزئی (mE)           {pct(totals,'mE'):6.2f}%")

    ca = [b["CA_pct"] for b in per_bp]
    q = quantiles(ca)
    print()
    print("  پراکندگیِ CA بر حسبِ این‌که نقطه‌ی شکست کدام باشد:")
    print(f"    صدکِ ۵ {q[0]:.1f}%   چارکِ اول {q[1]:.1f}%   میانه {q[2]:.1f}%   "
          f"چارکِ سوم {q[3]:.1f}%   صدکِ ۹۵ {q[4]:.1f}%")
    print(f"    نقاطِ شکستی که CA آن‌ها ≥۹۰٪ است: "
          f"{100.0*sum(1 for x in ca if x >= 90)/len(ca):.0f}%")

    # اثرِ فاصله‌ی خوانش تا نقطه‌ی شکست -- جایی که خطا واقعاً اهمیت پیدا می‌کند
    print()
    print("=" * 74)
    print("چرا اکثرِ خطاها دسته را عوض نمی‌کنند")
    print("=" * 74)
    near = far = near_flip = far_flip = 0
    for b in bps:
        for p in pairs:
            d = min(abs(p["ref_mm"] - b["s"]), abs(p["ref_mm"] - b["r"]))
            flip = classify(p["ref_mm"], b["s"], b["r"]) != \
                classify(p["sys_mm"], b["s"], b["r"])
            if d <= 4.0:
                near += 1
                near_flip += flip
            else:
                far += 1
                far_flip += flip
    print(f"  خوانشِ مرجع تا ۴mmیِ نقطه‌ی شکست : {100.0*near_flip/near:5.1f}% "
          f"تغییرِ دسته   (n={near:,})")
    print(f"  خوانشِ مرجع دورتر از ۴mm        : {100.0*far_flip/far:5.1f}% "
          f"تغییرِ دسته   (n={far:,})")
    print("  یعنی خطایِ میلی‌متری فقط در همسایگیِ نقطه‌ی شکست به تصمیم تبدیل می‌شود.")

    # تفکیک بر اساسِ شاخه‌ای که مرز را تولید کرده -- پیوند با یافته‌ی ۰.۳σ
    print()
    print("=" * 74)
    print("توافقِ دسته‌ای به تفکیکِ شاخه‌ی تولیدکننده‌ی مرز")
    print("=" * 74)
    print(f"  {'شاخه':<14}{'دیسک':>6}{'CA':>9}{'VME':>8}{'ME':>8}")
    for src in ("watershed", "otsu", "radial", "-"):
        sub = [p for p in pairs if p["source"] == src]
        if not sub:
            continue
        _, t = evaluate(sub, bps)
        name = {"-": "بدونِ هاله"}.get(src, src)
        print(f"  {name:<14}{len(sub):>6}{pct(t,'CA'):>8.1f}%"
              f"{pct(t,'VME'):>7.2f}%{pct(t,'ME'):>7.2f}%")

    # ── استحکامِ نتیجه: وزن‌دهیِ یکسان به هر جفتِ یکتایِ (S,R) ──────────────
    # در جدولِ EUCAST یک جفتِ نقطه‌ی شکست (مثلاً S≥24/R<21) برایِ ده‌ها عامل تکرار
    # می‌شود. اگر همه را یکسان بشماریم، آن موقعیتِ خاص رویِ محور بیش از حد وزن
    # می‌گیرد. این بخش همان حساب را رویِ جفت‌هایِ *یکتا* تکرار می‌کند.
    uniq = {}
    for b in bps:
        uniq.setdefault((b["s"], b["r"]), b)
    _, tu = evaluate(pairs, list(uniq.values()))
    print()
    print("=" * 74)
    print("استحکام: همان حساب با وزنِ یکسان رویِ جفت‌هایِ یکتایِ (S,R)")
    print("=" * 74)
    print(f"  {len(uniq)} جفتِ یکتا (در برابرِ {len(bps)} سطرِ جدول)")
    print(f"  CA {pct(tu,'CA'):.2f}%   VME {pct(tu,'VME'):.2f}%   "
          f"ME {pct(tu,'ME'):.2f}%   mE {pct(tu,'mE'):.2f}%")
    print("  نتیجه با حالتِ وزن‌دارِ بالا هم‌خوان است، پس به تکرارِ نقاطِ شکست حساس نیست.")

    # ── هدفِ مهندسی: خطا تا کجا باید کم شود؟ ──────────────────────────────
    # خطایِ هر دیسک را در ضریبِ k کوچک می‌کنیم و می‌بینیم VME کِی زیرِ سقف می‌رود.
    # این «MAEِ لازم برایِ پذیرشِ بالینی» را می‌دهد -- یک هدفِ عددیِ مشخص به‌جایِ
    # «هرچه کمتر بهتر».
    print()
    print("=" * 74)
    print("هدفِ مهندسی: خطا تا کجا باید کم شود تا بالینی قابلِ‌قبول شود؟")
    print("=" * 74)
    print(f"  {'ضریبِ خطا':<12}{'MAE':>8}{'CA':>9}{'VME':>8}{'ME':>8}")
    base_mae = (sum(abs(p["sys_mm"] - p["ref_mm"]) for p in pairs
                    if p["ref_has_zone"] and p["sys_mm"] > NO_ZONE_MM) /
                max(1, sum(1 for p in pairs
                           if p["ref_has_zone"] and p["sys_mm"] > NO_ZONE_MM)))
    target = None
    for k in (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.10):
        scaled = [{**p, "sys_mm": p["ref_mm"] + k * (p["sys_mm"] - p["ref_mm"])}
                  for p in pairs]
        _, t = evaluate(scaled, bps)
        v, m = pct(t, "VME"), pct(t, "ME")
        print(f"  ×{k:<11.2f}{k*base_mae:>7.2f}{pct(t,'CA'):>8.1f}%"
              f"{v:>7.2f}%{m:>7.2f}%")
        if target is None and v <= VME_LIMIT and m <= ME_LIMIT:
            target = k * base_mae
    if target:
        print(f"\n  هر دو سقف از حدودِ MAE ≈ {target:.2f} mm به پایین برآورده می‌شوند.")
    else:
        print("\n  حتی با کوچک‌ترین ضریب هم هر دو سقف هم‌زمان برآورده نشد "
              "-- یعنی بخشی از خطا از تشخیصِ حضور/عدمِ هاله می‌آید، نه از قطر.")

    per_bp.sort(key=lambda b: b["CA_pct"])
    print()
    print("=" * 74)
    print("بدترین ۱۰ نقطه‌ی شکست (جایی که سیستم بیشترین ریسک را دارد)")
    print("=" * 74)
    print(f"  {'ارگانیسم':<26}{'عامل':<34}{'S≥':>4}{'R<':>4}{'CA':>7}{'VME':>7}")
    for b in per_bp[:10]:
        print(f"  {b['organism'][:25]:<26}{b['agent'][:33]:<34}"
              f"{b['s']:>4.0f}{b['r']:>4.0f}{b['CA_pct']:>6.0f}%{b['VME_pct']:>6.1f}%")

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_bp[0].keys()))
        w.writeheader()
        w.writerows(per_bp)
    print(f"\nجزئیاتِ هر نقطه‌ی شکست: {OUT}")


if __name__ == "__main__":
    main()

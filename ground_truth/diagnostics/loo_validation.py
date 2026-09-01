"""
اعتبارسنجیِ leave-one-image-out برایِ تصمیم‌هایی که رویِ همین ۱۱ عکس انتخاب شده‌اند.

چرا لازم است: عددِ درون‌نمونه‌ای (in-sample) وقتی آستانه/آماره‌ی خودش هم از رویِ همان
داده انتخاب شده باشد، به‌طورِ سیستماتیک خوش‌بینانه است -- حتی اگر هیچ مدلی «آموزش»
نبیند. این اسکریپت همان انتخاب‌ها را ۱۱ بار تکرار می‌کند، هر بار رویِ ۱۰ عکس انتخاب
و رویِ عکسِ کنارگذاشته‌شده ارزیابی می‌کند، و نتیجه‌ی تجمیعی را کنارِ عددِ درون‌نمونه‌ای
می‌گذارد. اختلافِ این دو عدد، بزرگیِ واقعیِ overfitting است.
"""
import csv
from collections import defaultdict

CSV = "/tmp/branch_compare.csv"
GATES = [0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0]
OTSU_STATS = ["otsu_mean", "otsu_med", "otsu_p75", "otsu_p90"]
WS_STATS = ["ws_mean", "ws_med", "ws_p75", "ws_p90"]

rows = [r for r in csv.DictReader(open(CSV)) if r["gt_num"]]
for r in rows:
    for k in list(r):
        if k not in ("image", "gt_num", "fusion_source"):
            r[k] = float(r[k]) if r[k] not in ("", None) else 0.0
    r["gt"] = float(r["gt_halo"]) if r["gt_halo"] else None

IMAGES = sorted({r["image"] for r in rows})


def predict(r, gate, otsu_key, ws_key, use_stat):
    """همان زنجیره‌ی ادغامِ ماژولِ ۱۶.۷، با پارامترهایِ قابلِ‌انتخاب."""
    if r[ws_key] > 0.01:
        return r[ws_key]
    if r[otsu_key] > 0.01:
        return r[otsu_key]
    if use_stat and r["stat_mean"] > 0.01:
        return r["stat_mean"]
    if r["radial_pre_mm"] > 0.01 and abs(r["contrast_sigma"]) >= gate:
        return r["radial_pre_mm"]
    return 0.0


def score(subset, gate, otsu_key, ws_key, use_stat):
    tp = fn = fp = tn = 0
    errs = []
    for r in subset:
        g, v = r["gt"], predict(r, gate, otsu_key, ws_key, use_stat)
        if g is not None and v > 0.01:
            tp += 1
            errs.append(v - g)
        elif g is not None:
            fn += 1
        elif v > 0.01:
            fp += 1
        else:
            tn += 1
    acc = (tp + tn) / max(len(subset), 1)
    mae = sum(abs(e) for e in errs) / len(errs) if errs else 0.0
    return tp, fn, fp, tn, acc, mae, errs


def select(train, search_stats, use_stat):
    """انتخابِ پارامترها رویِ داده‌ی آموزش -- معیار: بیشترین دقت، و در تساوی کمترین MAE."""
    best, best_key = None, None
    otsu_opts = OTSU_STATS if search_stats else ["otsu_p90"]
    ws_opts = WS_STATS if search_stats else ["ws_mean"]
    for gate in GATES:
        for ok in otsu_opts:
            for wk in ws_opts:
                _, _, _, _, acc, mae, _ = score(train, gate, ok, wk, use_stat)
                key = (acc, -mae)
                if best is None or key > best:
                    best, best_key = key, (gate, ok, wk)
    return best_key


def run_loo(search_stats, use_stat, label):
    tp = fn = fp = tn = 0
    errs = []
    picks = defaultdict(int)
    for held in IMAGES:
        train = [r for r in rows if r["image"] != held]
        test = [r for r in rows if r["image"] == held]
        params = select(train, search_stats, use_stat)
        picks[params] += 1
        a, b, c, d, _, _, e = score(test, *params, use_stat)
        tp += a; fn += b; fp += c; tn += d; errs += e
    n = len(rows)
    acc = (tp + tn) / n
    mae = sum(abs(x) for x in errs) / len(errs) if errs else 0.0
    bias = sum(errs) / len(errs) if errs else 0.0
    print(f"{label:<34}{tp:>5}{fn:>5}{fp:>5}{tn:>5}{acc:>8.3f}{mae:>8.2f}{bias:>+8.2f}")
    return picks


print("=" * 82)
print("درون‌نمونه‌ای (in-sample) در برابرِ leave-one-image-out")
print("=" * 82)
print(f"{'':<34}{'TP':>5}{'FN':>5}{'FP':>5}{'TN':>5}{'Acc':>8}{'MAE':>8}{'Bias':>8}")

# خطِ پایه -- هیچ انتخابی رویِ داده انجام نشده، پس in-sample و LOO یکی است.
tp, fn, fp, tn, acc, mae, errs = score(rows, 0.0, "otsu_p90", "ws_mean", False)
b_tp = b_fn = b_fp = b_tn = 0
b_errs = []
for r in rows:
    g, v = r["gt"], (r["radial_pre_mm"] if r["radial_pre_mm"] > 0.01 else 0.0)
    if g is not None and v > 0.01:
        b_tp += 1; b_errs.append(v - g)
    elif g is not None: b_fn += 1
    elif v > 0.01: b_fp += 1
    else: b_tn += 1
print(f"{'شاخه‌ی شعاعی تنها (خطِ پایه)':<34}{b_tp:>5}{b_fn:>5}{b_fp:>5}{b_tn:>5}"
      f"{(b_tp+b_tn)/len(rows):>8.3f}{sum(abs(e) for e in b_errs)/len(b_errs):>8.2f}"
      f"{sum(b_errs)/len(b_errs):>+8.2f}")

# ادغامِ مستقرشده (gate=4.0, otsu_p90, ws_mean) -- درون‌نمونه‌ای
tp, fn, fp, tn, acc, mae, errs = score(rows, 4.0, "otsu_p90", "ws_mean", False)
print(f"{'ادغام (پارامترهایِ فعلی) -- in-sample':<34}{tp:>5}{fn:>5}{fp:>5}{tn:>5}"
      f"{acc:>8.3f}{mae:>8.2f}{sum(errs)/len(errs):>+8.2f}")

print()
p1 = run_loo(False, False, "LOO: فقط گیت انتخاب می‌شود")
p2 = run_loo(True, False, "LOO: گیت + آماره‌ها انتخاب می‌شوند")
p3 = run_loo(True, True, "LOO: + شاخه‌ی آماری در زنجیره")

print()
print("پارامترهایی که در تاهایِ LOO انتخاب شدند (گیت، آماره‌ی Otsu، آماره‌ی WS):")
for label, picks in [("فقط گیت", p1), ("گیت+آماره", p2)]:
    items = ", ".join(f"{k}×{v}" for k, v in sorted(picks.items(), key=lambda kv: -kv[1]))
    print(f"  {label}: {items}")

# آیا شاخه‌ی آماری چیزی اضافه می‌کند؟
print()
add = [r for r in rows
       if r["ws_mean"] <= 0.01 and r["otsu_p90"] <= 0.01 and r["stat_mean"] > 0.01]
with_gt = [r for r in add if r["gt"] is not None]
print(f"دیسک‌هایی که فقط شاخه‌ی آماری رویشان سیگنال دارد: {len(add)} "
      f"(از این تعداد {len(with_gt)} واقعاً هاله دارند، {len(add) - len(with_gt)} ندارند)")
if with_gt:
    e = [r["stat_mean"] - r["gt"] for r in with_gt]
    print(f"   MAE شاخه‌ی آماری رویِ همان‌ها: {sum(abs(x) for x in e)/len(e):.2f} mm")
    er = [r["radial_pre_mm"] - r["gt"] for r in with_gt if r["radial_pre_mm"] > 0.01]
    if er:
        print(f"   MAE شاخه‌ی شعاعی رویِ همان‌ها: {sum(abs(x) for x in er)/len(er):.2f} mm "
              f"(n={len(er)})")

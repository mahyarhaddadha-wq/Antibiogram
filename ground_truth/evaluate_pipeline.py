#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ارزیابیِ پایپلاینِ اصلی (disk_detection_pipeline_modular.ipynb) در برابرِ دیتابیسِ
Ground Truth ساخته‌شده از ۱۱ عکسِ اندازه‌گیری‌شده توسطِ متخصصِ آزمایشگاه
(ground_truth_expert_readings.csv).

خودِ نوت‌بوک هیچ تغییری نمی‌کند -- این اسکریپت مثلِ batch_process_antibiogram.py
فقط cfg.image_path را override می‌کند و نوت‌بوکِ واقعی را با nbclient اجرا می‌کند،
سپس سلولِ «گزارش نهایی» (خروجیِ متنیِ خودِ نوت‌بوک، نه هیچ منطقِ موازی/کپی‌شده) را
پارس می‌کند.

اصلِ کلیدیِ طراحی -- عدمِ Circularity: تطبیقِ دیسک‌های تشخیص‌داده‌شده با دیسک‌های
Ground Truth بر اساسِ **نزدیک‌ترین موقعیتِ پیکسلی** انجام می‌شود (نه بر اساسِ
شماره‌ی دیسک)، چون شماره‌گذاریِ Ground Truth از یک لوکیتورِ کاملاً مستقل
(annotate_disks.py) آمده و شماره‌گذاریِ پایپلاین ممکن است متفاوت باشد یا اصلاً
دیسکی را جا بیندازد/کاذب اضافه کند -- دقیقاً همان چیزی که این ارزیابی باید بسنجد.

اجرا:
    python3 ground_truth/evaluate_pipeline.py [--kernel antibiogram-test] [--no-overlays]
خروجی:
  - گزارشِ متنیِ کامل در stdout
  - جزئیاتِ خام: ground_truth/evaluation_results.csv
  - overlayِ خروجیِ خودِ پایپلاین (دایره‌های نهاییِ هاله، همان تصویرِ ماژولِ ۱۶.۶) برای
    هر عکس: ground_truth/pipeline_overlays/<نام‌عکس>_pipeline_halo.png -- برای مقایسه‌ی
    بصریِ مستقیم با عکسِ ماژیک‌دارِ متناظرش در ground_truth/marked_images/
"""
import argparse
import base64
import copy
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NOTEBOOK_PATH = REPO_ROOT / "disk_detection_pipeline_modular.ipynb"
RAW_DIR = HERE / "raw_images"
GT_CSV = HERE / "ground_truth_expert_readings.csv"
OUT_CSV = HERE / "evaluation_results.csv"
OVERLAY_DIR = HERE / "pipeline_overlays"

CFG_INIT_MARKER = "cfg = Phase2Config()"
REPORT_MARKER = "گزارش نهایی آنتی‌بایوگرام"
NO_DISK_MARKER = "هیچ دیسکی در این پتری تایید نشد"
# آخرین ماژولِ اصلاح‌کننده‌ی مرز هاله -- همان نشانه‌ای که batch_process_antibiogram.py
# هم برایِ استخراجِ تصویرِ نهاییِ هاله استفاده می‌کند.
HALO_MARKER = "# ── ماژول ۱۶.۶ (جدید)"

DISH_HEADER_RE = re.compile(r"پتری #(\d+) \(روش تشخیص: (\S+), confidence=([\d.]+)\)")
DISK_LINE_RE = re.compile(
    r"دیسک (\d+): مرکز=\(([\d.]+),([\d.]+)\)"
    r"(?: \| قطر دیسک≈([\d.]+) mm)?"
    r" \| (?:قطر هاله≈([\d.]+) mm|هاله تشکیل (نشد|شد))"
)

# نصفِ کمترین فاصله‌ی واقعیِ بین دو دیسکِ Ground Truth در همان عکس به‌عنوانِ
# شعاعِ تطبیق استفاده می‌شود (نه یک مقدارِ مطلق) -- یعنی این هم مثلِ بقیه‌ی
# پایپلاین نسبی/دیتاست‌محور است.
MATCH_RADIUS_SAFETY = 0.5


def find_cell_idx(nb, marker):
    for i, c in enumerate(nb["cells"]):
        if marker in c["source"]:
            return i
    raise RuntimeError("marker not found: " + marker)


def build_notebook_for_image(base_nb, image_path):
    nb = copy.deepcopy(base_nb)
    idx_cfg = find_cell_idx(nb, CFG_INIT_MARKER)
    override_cell = nbformat.v4.new_code_cell(f'cfg.image_path = r"{image_path}"')
    cells = list(nb["cells"])
    cells.insert(idx_cfg + 1, override_cell)
    nb["cells"] = cells
    return nb


def extract_stream_text(cell):
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append(out.get("text", ""))
    return "".join(parts)


def extract_png_outputs(cell):
    pngs = []
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if "image/png" in data:
            pngs.append(base64.b64decode(data["image/png"]))
    return pngs


def parse_report(text):
    """پارسِ خروجیِ متنیِ سلولِ گزارشِ نهایی -> لیستِ dish هرکدام با لیستِ disks."""
    dishes = []
    current = None
    for line in text.splitlines():
        m = DISH_HEADER_RE.search(line)
        if m:
            current = {"index": int(m.group(1)), "method": m.group(2),
                       "confidence": float(m.group(3)), "disks": [], "no_disks": False}
            dishes.append(current)
            continue
        if current is None:
            continue
        if NO_DISK_MARKER in line:
            current["no_disks"] = True
            continue
        m = DISK_LINE_RE.search(line)
        if m:
            disk_num, x, y, d_mm, h_mm, no_halo = m.groups()
            current["disks"].append({
                "disk_number": int(disk_num),
                "x": float(x), "y": float(y),
                "disk_diameter_mm": float(d_mm) if d_mm else None,
                "halo_diameter_mm": float(h_mm) if h_mm else None,
            })
    return dishes


def run_pipeline(image_path: Path, kernel_name: str, save_overlay: bool = True):
    base_nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nb = build_notebook_for_image(base_nb, image_path)
    client = NotebookClient(nb, kernel_name=kernel_name, timeout=1800)
    client.execute()

    if save_overlay:
        OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
        halo_pngs = extract_png_outputs(nb["cells"][find_cell_idx(nb, HALO_MARKER)])
        for i, png in enumerate(halo_pngs, start=1):
            suffix = "" if len(halo_pngs) == 1 else f"_dish{i}"
            out_path = OVERLAY_DIR / f"{image_path.stem}_pipeline_halo{suffix}.png"
            out_path.write_bytes(png)

    idx = find_cell_idx(nb, REPORT_MARKER)
    text = extract_stream_text(nb["cells"][idx])
    return parse_report(text)


def load_ground_truth():
    by_image = defaultdict(list)
    with open(GT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_image[row["image_file"]].append({
                "disk_number": int(row["disk_number"]),
                "x": float(row["x_px"]), "y": float(row["y_px"]),
                "disk_diameter_mm_expert":
                    float(row["disk_diameter_mm_expert"]) if row.get("disk_diameter_mm_expert") else None,
                "halo_diameter_mm_expert":
                    float(row["halo_diameter_mm_expert"]) if row["halo_diameter_mm_expert"] else None,
            })
    return by_image


def match_disks(gt_disks, sys_disks):
    """تطبیقِ حریصانه‌ی نزدیک‌ترین‌همسایه (greedy nearest-neighbor)، به‌ترتیبِ
    صریح/قطعی: جفت‌های کاندید به‌ترتیبِ فاصله‌ی صعودی مرتب می‌شوند، هر بار
    نزدیک‌ترین جفتِ هنوز-تطبیق‌نیافته پذیرفته می‌شود (اگر در آستانه بگنجد)."""
    if not gt_disks:
        return [], [], list(range(len(sys_disks)))
    if len(gt_disks) > 1:
        min_gt_dist = min(
            math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            for i, a in enumerate(gt_disks) for j, b in enumerate(gt_disks) if i < j
        )
    else:
        min_gt_dist = float("inf")
    match_radius = MATCH_RADIUS_SAFETY * min_gt_dist if min_gt_dist != float("inf") else float("inf")

    candidates = []
    for gi, g in enumerate(gt_disks):
        for si, s in enumerate(sys_disks):
            d = math.hypot(g["x"] - s["x"], g["y"] - s["y"])
            if d <= match_radius:
                candidates.append((d, gi, si))
    candidates.sort(key=lambda t: t[0])

    matched_gt, matched_sys = set(), set()
    pairs = []
    for d, gi, si in candidates:
        if gi in matched_gt or si in matched_sys:
            continue
        matched_gt.add(gi)
        matched_sys.add(si)
        pairs.append((gi, si, d))

    unmatched_gt = [i for i in range(len(gt_disks)) if i not in matched_gt]
    unmatched_sys = [i for i in range(len(sys_disks)) if i not in matched_sys]
    return pairs, unmatched_gt, unmatched_sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="antibiogram-test")
    ap.add_argument("--images", nargs="*", default=None,
                     help="فقط این عکس‌ها (مثلاً gt_01.jpg)؛ پیش‌فرض همه‌ی ۱۱ عکس")
    ap.add_argument("--no-overlays", action="store_true",
                     help="overlay خروجیِ پایپلاین را ذخیره نکن (فقط اجرای سریع‌تر)")
    args = ap.parse_args()

    gt_by_image = load_ground_truth()
    images = args.images or sorted(gt_by_image.keys())

    all_rows = []
    petri_ok, petri_total = 0, 0
    tp = fp = fn = 0
    zone_confusion = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}  # حضور/عدمِ هاله
    diffs_mm = []
    disk_diffs_mm = []

    for img in images:
        petri_total += 1
        image_path = RAW_DIR / img
        print(f"=== {img} ===")
        sys_dishes = run_pipeline(image_path, args.kernel, save_overlay=not args.no_overlays)

        n_real_dishes = sum(1 for d in sys_dishes if not d["no_disks"])
        if len(sys_dishes) == 1 and n_real_dishes == 1:
            petri_ok += 1
        else:
            print(f"  [هشدار] تشخیصِ پتری غیرمنتظره: {len(sys_dishes)} پتری "
                  f"({n_real_dishes} با دیسک) -- انتظار: دقیقاً ۱ پتری با دیسک")

        sys_disks_flat = []
        for d in sys_dishes:
            sys_disks_flat.extend(d["disks"])

        gt_disks = gt_by_image[img]
        pairs, unmatched_gt, unmatched_sys = match_disks(gt_disks, sys_disks_flat)

        tp += len(pairs)
        fn += len(unmatched_gt)
        fp += len(unmatched_sys)

        for gi, si, dist in pairs:
            g = gt_disks[gi]
            s = sys_disks_flat[si]
            gt_mm = g["halo_diameter_mm_expert"]
            sys_mm = s["halo_diameter_mm"]
            gt_disk_mm = g["disk_diameter_mm_expert"]
            sys_disk_mm = s["disk_diameter_mm"]

            if gt_mm is not None and sys_mm is not None:
                zone_confusion["tp"] += 1
                diff = sys_mm - gt_mm
                diffs_mm.append(diff)
            elif gt_mm is not None and sys_mm is None:
                zone_confusion["fn"] += 1
            elif gt_mm is None and sys_mm is not None:
                zone_confusion["fp"] += 1
            else:
                zone_confusion["tn"] += 1

            disk_diff = None
            if gt_disk_mm is not None and sys_disk_mm is not None:
                disk_diff = sys_disk_mm - gt_disk_mm
                disk_diffs_mm.append(disk_diff)

            all_rows.append({
                "image": img, "gt_disk_number": g["disk_number"],
                "match_dist_px": round(dist, 1),
                "gt_disk_mm": gt_disk_mm, "sys_disk_mm": sys_disk_mm, "disk_diff_mm": disk_diff,
                "gt_halo_mm": gt_mm, "sys_halo_mm": sys_mm,
                "diff_mm": (sys_mm - gt_mm) if (gt_mm is not None and sys_mm is not None) else None,
            })

        for gi in unmatched_gt:
            g = gt_disks[gi]
            print(f"  [FN دیسک] دیسکِ واقعی #{g['disk_number']} (x={g['x']:.0f},y={g['y']:.0f}) "
                  f"توسطِ پایپلاین تشخیص داده نشد.")
            all_rows.append({"image": img, "gt_disk_number": g["disk_number"],
                              "match_dist_px": None,
                              "gt_disk_mm": g["disk_diameter_mm_expert"], "sys_disk_mm": None, "disk_diff_mm": None,
                              "gt_halo_mm": g["halo_diameter_mm_expert"],
                              "sys_halo_mm": None, "diff_mm": None})
        for si in unmatched_sys:
            s = sys_disks_flat[si]
            print(f"  [FP دیسک] پایپلاین یک دیسکِ کاذب در (x={s['x']:.0f},y={s['y']:.0f}) گزارش داد.")
            all_rows.append({"image": img, "gt_disk_number": None,
                              "match_dist_px": None,
                              "gt_disk_mm": None, "sys_disk_mm": s["disk_diameter_mm"], "disk_diff_mm": None,
                              "gt_halo_mm": None,
                              "sys_halo_mm": s["halo_diameter_mm"], "diff_mm": None})

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt_disk_number", "match_dist_px",
                                               "gt_disk_mm", "sys_disk_mm", "disk_diff_mm",
                                               "gt_halo_mm", "sys_halo_mm", "diff_mm"])
        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 70)
    print("گزارشِ نهاییِ ارزیابی")
    print("=" * 70)

    print(f"\n۱) تشخیصِ پتری: {petri_ok}/{petri_total} عکس به‌درستی دقیقاً یک پتریِ "
          f"دارایِ دیسک تشخیص داده شد.")

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    print(f"\n۲) تشخیصِ دیسک (روی {petri_total} عکس، {tp + fn} دیسکِ واقعی):")
    print(f"   TP={tp}  FP={fp}  FN={fn}")
    print(f"   Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    print(f"\n۳) وضعیتِ تشخیصِ هاله (فقط روی دیسک‌هایِ درست‌تشخیص‌داده‌شده، n={tp}):")
    print(f"   هردو هاله دارند (TP)={zone_confusion['tp']}  "
          f"واقعی هاله دارد ولی پایپلاین ندید (FN)={zone_confusion['fn']}  "
          f"واقعی هاله ندارد ولی پایپلاین کاذب دید (FP)={zone_confusion['fp']}  "
          f"هردو بدونِ هاله (TN)={zone_confusion['tn']}")

    if diffs_mm:
        mae = statistics.mean(abs(d) for d in diffs_mm)
        bias = statistics.mean(diffs_mm)
        sd = statistics.stdev(diffs_mm) if len(diffs_mm) > 1 else 0.0
        loa_low, loa_high = bias - 1.96 * sd, bias + 1.96 * sd
        print(f"\n۴) دقتِ عددیِ قطرِ هاله (mm)، روی {len(diffs_mm)} دیسکِ هم‌دارایِ هاله:")
        print(f"   MAE={mae:.2f} mm   Bias (میانگینِ خطا)={bias:+.2f} mm   SD={sd:.2f} mm")
        print(f"   Bland–Altman حدودِ توافقِ ۹۵٪: [{loa_low:+.2f}, {loa_high:+.2f}] mm")
    else:
        print("\n۴) هیچ جفتِ دیسکِ هم‌دارایِ هاله‌ای برای مقایسه‌ی عددی پیدا نشد.")

    if disk_diffs_mm:
        d_mae = statistics.mean(abs(d) for d in disk_diffs_mm)
        d_bias = statistics.mean(disk_diffs_mm)
        d_sd = statistics.stdev(disk_diffs_mm) if len(disk_diffs_mm) > 1 else 0.0
        print(f"\n۵) دقتِ عددیِ قطرِ خودِ دیسک (mm، مرجع={gt_disks[0]['disk_diameter_mm_expert'] if gt_disks else '?'}mm "
              f"برایِ همه‌ی دیسک‌ها)، روی {len(disk_diffs_mm)} دیسک:")
        print(f"   MAE={d_mae:.2f} mm   Bias={d_bias:+.2f} mm   SD={d_sd:.2f} mm")

    print(f"\noverlayِ خروجیِ پایپلاین در: {OVERLAY_DIR}/ (برایِ مقایسه‌ی بصری با "
          f"ground_truth/marked_images/)")
    print(f"جزئیاتِ خام در: {OUT_CSV}")


if __name__ == "__main__":
    main()

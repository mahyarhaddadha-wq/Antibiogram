#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ساختِ پوشه‌هایِ بازرسیِ دستیِ ground truth -- یک پوشه به‌ازایِ هر ۱۱ عکسِ ارزیابی،
شاملِ عکسِ شماره‌گذاری‌شده (`overlays/gt_XX_numbered.jpg`)، عکسِ ماژیک‌دارِ کارشناسِ
متناظرش (`marked_images/marked_YY.jpg`)، و یک فایلِ متنی با مشخصاتِ هر دیسک/هاله
دقیقاً از رویِ `ground_truth_expert_readings.csv` -- تا کاربر بتواند مستقل از هر
کدِ دیگری بررسی کند که مکانیزمِ ارزیابی (evaluate_pipeline.py) از اطلاعاتِ درست
استفاده می‌کند.

**چرا تناظرِ marked_XX <-> gt_YY را نمی‌شود از رویِ نامِ فایل حدس زد:** نام‌گذاریِ
عکس‌هایِ ماژیک‌دار (marked_01 .. marked_11 + یک عکسِ اضافه‌یِ marked_12_gt02) با
ترتیبِ ارسال بوده، نه با شماره‌ی gt_XX. با مقایسه‌ی بصریِ مستقیم مشخص شد مثلاً
marked_04.jpg در واقع همان gt_01.jpg است (نه gt_04). تناظرِ زیر با تطبیقِ **مجموعه‌ی
دقیقِ اعدادِ دستی‌نویسِ آبیِ رویِ هر عکسِ ماژیک‌دار با ستونِ halo_diameter_mm_expert
همان gt_XX در CSV** به‌دست آمده (تطابقِ کاملِ چندمجموعه‌ای رویِ ۸-۹ عدد به‌ازایِ هر
عکس -- برخوردِ تصادفی عملاً غیرِممکن است)، نه با شماره‌ی فایل.
"""
import csv
import os
import shutil
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "ground_truth_expert_readings.csv")
OVERLAY_DIR = os.path.join(HERE, "overlays")
MARKED_DIR = os.path.join(HERE, "marked_images")
OUT_DIR = os.path.join(HERE, "verification_packets")

# تناظرِ gt_XX.jpg <-> marked_YY.jpg -- به‌دست‌آمده با تطبیقِ چندمجموعه‌ی
# halo_diameter_mm_expert (نه با شماره‌ی فایل؛ توضیح در docstring بالا).
GT_TO_MARKED = {
    "gt_01.jpg": "marked_04.jpg",
    "gt_02.jpg": "marked_12_gt02.jpg",
    "gt_03.jpg": "marked_01.jpg",
    "gt_04.jpg": "marked_02.jpg",
    "gt_05.jpg": "marked_03.jpg",
    "gt_06.jpg": "marked_06.jpg",
    "gt_07.jpg": "marked_07.jpg",
    "gt_08.jpg": "marked_08.jpg",
    "gt_09.jpg": "marked_09.jpg",
    "gt_10.jpg": "marked_10.jpg",
    "gt_11.jpg": "marked_11.jpg",
}


def load_rows():
    rows = defaultdict(list)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["image_file"]].append(row)
    return rows


def write_text_report(path, image_file, marked_file, rows):
    lines = []
    lines.append(f"داده‌ی مرجعِ (Ground Truth) دیسک/هاله برایِ: {image_file}")
    lines.append(f"عکسِ ماژیک‌دارِ کارشناسِ متناظر: {marked_file}")
    lines.append(f"منبعِ اعداد: ground_truth_expert_readings.csv (بدونِ هیچ تغییری)")
    lines.append("")
    lines.append("راهنما: شماره‌ی دیسک در ستونِ اول دقیقاً همان شماره‌ای است که رویِ")
    lines.append("عکسِ *_raw_numbered.jpg با رنگِ آبی/فیروزه‌ای کنارِ هر دیسک نوشته شده")
    lines.append("(برچسبِ آنتی‌بیوتیک هم رویِ خودِ همان عکس، کنارِ دایره، قابلِ خواندن است).")
    lines.append("برایِ تطبیق با عکسِ ماژیک‌دار (که ممکن است با زاویه/چرخشِ متفاوتی")
    lines.append("گرفته شده باشد)، از رویِ برچسبِ آنتی‌بیوتیکِ چاپ‌شده رویِ خودِ دیسک")
    lines.append("(که در هر دو عکس یکسان و قابلِ‌مشاهده است) دیسکِ متناظر را پیدا کنید،")
    lines.append("و عددی که کارشناس با ماژیکِ آبی کنارش نوشته را با ستونِ")
    lines.append("halo_diameter_mm_expert زیر مقایسه کنید.")
    lines.append("")
    header = f"{'شماره‌ی دیسک':<14}{'x_px':>10}{'y_px':>10}{'قطرِ دیسک (mm)':>18}{'قطرِ هاله (mm)':>18}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in sorted(rows, key=lambda r: int(r["disk_number"])):
        halo = r["halo_diameter_mm_expert"].strip()
        halo_disp = halo if halo else "بدونِ هاله"
        lines.append(
            f"{r['disk_number']:<14}{float(r['x_px']):>10.1f}{float(r['y_px']):>10.1f}"
            f"{r['disk_diameter_mm_expert']:>18}{halo_disp:>18}"
        )
    lines.append("")
    n_halo = sum(1 for r in rows if r["halo_diameter_mm_expert"].strip())
    lines.append(f"جمع: {len(rows)} دیسک، {n_halo} موردِ دارایِ هاله.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    rows_by_image = load_rows()
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    for image_file, marked_file in GT_TO_MARKED.items():
        stem = os.path.splitext(image_file)[0]  # gt_01
        pair_dir = os.path.join(OUT_DIR, stem)
        os.makedirs(pair_dir, exist_ok=True)

        numbered_src = os.path.join(OVERLAY_DIR, f"{stem}_numbered.jpg")
        marked_src = os.path.join(MARKED_DIR, marked_file)
        if not os.path.isfile(numbered_src):
            raise FileNotFoundError(numbered_src)
        if not os.path.isfile(marked_src):
            raise FileNotFoundError(marked_src)

        shutil.copy2(numbered_src, os.path.join(pair_dir, f"{stem}_raw_numbered.jpg"))
        marked_stem = os.path.splitext(marked_file)[0]
        shutil.copy2(marked_src, os.path.join(pair_dir, f"{stem}_expert_marked_src-{marked_stem}.jpg"))

        rows = rows_by_image.get(image_file, [])
        write_text_report(os.path.join(pair_dir, f"{stem}_disk_halo_data.txt"),
                          image_file, marked_file, rows)
        print(f"[ok] {stem}/  <-  {numbered_src.split('/')[-1]} + {marked_file}  ({len(rows)} دیسک)")

    print(f"\nهمه‌ی ۱۱ پوشه در {OUT_DIR} ساخته شد.")


if __name__ == "__main__":
    main()

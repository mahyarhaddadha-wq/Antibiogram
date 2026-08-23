#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
لوکیتورِ مستقلِ دیسک برای ساختِ Ground Truth (نه بخشی از پایپلاینِ اصلی).

چرا مستقل: قرار است این شماره‌گذاری مبنای «حقیقتِ زمینی» برای ارزیابیِ
disk_detection_pipeline_modular.ipynb باشد. اگر همان پایپلاین (شاخه‌ی Fusion،
ماژول ۱۴) برای پیدا کردنِ دیسک‌ها استفاده می‌شد، هر خطای آن پایپلاین (جاافتادنِ
یک دیسک، تفکیکِ غلط) مستقیماً وارد ground truth می‌شد و امکانِ دیدنِ آن خطا از
بین می‌رفت. پس اینجا عمداً از یک مسیرِ ساده و متفاوت استفاده می‌شود:
  ۱) پیدا کردنِ ظرف(های) پتری با Otsu ساده روی کل تصویر + بزرگ‌ترین کانتورها
     (نه Multi-Otsu + illumination_normalize + containment-filter + دو
     self-check که ماژول ۴ دارد).
  ۲) پیدا کردنِ دیسک‌ها با HoughCircles خامِ روی ROI هر ظرف، با بازه‌ی شعاعِ
     نسبی و سخاوتمندانه (نه پارامترهای دقیقِ cfg که خودِ Fusion از آن‌ها
     استفاده می‌کند).
  ۳) شماره‌گذاریِ دیسک‌ها به‌ترتیبِ ساعت‌گرد، شروع از ساعتِ ۱۲، نسبت به مرکزِ
     همان ظرف -- یک قاعده‌ی هندسیِ کاملاً قطعی و بدونِ وابستگی به کیفیتِ
     تشخیص.

خروجی هر عکس دستی بررسی می‌شود (overlay) قبل از این‌که کاربر مقادیرِ
میلی‌متریِ کاغذ را در CSV بنویسد -- اگر این لوکیتورِ ساده جایی دیسکی را از
دست بدهد یا یک شیِ کاذب را دیسک تشخیص دهد، باید دستی تصحیح شود (نه این‌که
دیتای ارزیابی را از قدیم خراب کند).

اجرا:
    python3 ground_truth/annotate_disks.py
ورودی: ground_truth/raw_images/*.jpg|*.jpeg|*.png
خروجی:
    ground_truth/overlays/<نام‌عکس>_numbered.jpg   -- برای بازرسیِ بصری
    ground_truth/ground_truth_template.csv          -- برای پرکردنِ دستیِ mm
"""
import csv
import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "raw_images")
OVERLAY_DIR = os.path.join(HERE, "overlays")
CSV_PATH = os.path.join(HERE, "ground_truth_template.csv")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# بازه‌ی نسبیِ سخاوتمندانه برای شعاعِ دیسک نسبت به قطرِ ظرف -- فقط برای پیدا
# کردنِ کاندید، نه یک ادعای دقیق؛ پهن‌تر از بازه‌ی cfg.disk_rel_diameter_min/max
# عمداً گرفته شده تا هیچ دیسکِ واقعی از قلم نیفتد (بازرسیِ بصریِ بعدی، کاذب‌ها
# را حذف می‌کند).
DISK_REL_RADIUS_MIN = 0.015
DISK_REL_RADIUS_MAX = 0.09

MIN_DISH_AREA_FRAC = 0.03  # نسبت به مساحتِ کلِ تصویر، برای رد کردنِ نویز


DISH_DOWNSCALE_TARGET_PX = 800


def find_dishes_hough(gray: np.ndarray):
    """تشخیصِ خودِ لبه‌ی گردِ ظرفِ پتری با یک HoughCircle درشت -- روی این ۱۱ عکس
    (هرکدام یک ظرف) این روش از Otsu ساده به‌مراتب مقاوم‌تر است: لبه‌ی ظرف همیشه
    یک دایره‌ی تمیز و پرکنتراست است، برخلاف Otsu که وقتی روشناییِ پس‌زمینه (مثلاً
    طرح/لوگوی چاپی روی پارچه‌ی زیرِ ظرف) به ظرف نزدیک باشد، آن‌ها را در یک بلابِ
    واحدِ نامنظم ادغام می‌کند."""
    h, w = gray.shape[:2]
    scale = DISH_DOWNSCALE_TARGET_PX / max(h, w)
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)
    sh, sw = small.shape[:2]
    r_min = int(0.25 * min(sh, sw))
    r_max = int(0.60 * min(sh, sw))
    circles = cv2.HoughCircles(
        small, cv2.HOUGH_GRADIENT, dp=1.5, minDist=sw,
        param1=80, param2=60, minRadius=r_min, maxRadius=r_max,
    )
    if circles is None:
        return []
    out = []
    for x, y, r in circles[0]:
        out.append({"cx": float(x / scale), "cy": float(y / scale), "r": float(r / scale)})
    return out


def find_dishes(gray: np.ndarray):
    """اول HoughCircle درشت (بند بالا)؛ اگر چیزی پیدا نکرد (مثلاً چند ظرف در یک
    قاب که با یک دایره‌ی تکی قابلِ توصیف نیست)، به روشِ قدیمیِ Otsu + کانتور
    برمی‌گردد."""
    hough_dishes = find_dishes_hough(gray)
    if hough_dishes:
        hough_dishes.sort(key=lambda d: (d["cx"], d["cy"]))
        return hough_dishes

    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=w / 300.0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ظرفِ پتری معمولاً روشن‌تر از پس‌زمینه نیست همیشه؛ هر دو قطبیت را امتحان کن.
    candidates = []
    for img_bin in (binary, cv2.bitwise_not(binary)):
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_DISH_AREA_FRAC * h * w:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            circularity = area / (np.pi * r * r + 1e-6)
            if circularity < 0.55:
                continue
            candidates.append((cx, cy, r, area))
    # حذفِ کاندیدهای هم‌پوشانِ تکراری (همان ظرف از دو قطبیت)، مرتب‌سازیِ قطعی
    # با x سپس y برای پایداریِ dish_id.
    candidates.sort(key=lambda t: t[3], reverse=True)
    kept = []
    for cx, cy, r, area in candidates:
        if any(np.hypot(cx - kx, cy - ky) < 0.5 * max(r, kr) for kx, ky, kr, _ in kept):
            continue
        kept.append((cx, cy, r, area))
    kept.sort(key=lambda t: (t[0], t[1]))
    return [{"cx": cx, "cy": cy, "r": r} for cx, cy, r, _ in kept]


DOWNSCALE_TARGET_PX = 900  # ROI به این عرض/ارتفاعِ تقریبی کوچک می‌شود، هم برای
# سرعت (Hough روی تصویرِ ~۳۰۰۰px بسیار کند است) و هم چون در این مقیاسِ
# استانداردشده Hough پایدارتر عمل می‌کند؛ نتیجه به مقیاسِ اصلی برگردانده می‌شود.


def find_disks_in_dish(gray: np.ndarray, dish: dict):
    cx, cy, r = dish["cx"], dish["cy"], dish["r"]
    x0, y0 = max(0, int(cx - r)), max(0, int(cy - r))
    x1, y1 = min(gray.shape[1], int(cx + r)), min(gray.shape[0], int(cy + r))
    roi = gray[y0:y1, x0:x1]

    scale = min(1.0, DOWNSCALE_TARGET_PX / max(roi.shape[:2]))
    roi_small = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else roi
    roi_small = cv2.medianBlur(roi_small, 5)

    r_min_px = max(4, int(DISK_REL_RADIUS_MIN * 2 * r * scale))
    r_max_px = max(r_min_px + 1, int(DISK_REL_RADIUS_MAX * 2 * r * scale))
    circles = cv2.HoughCircles(
        roi_small, cv2.HOUGH_GRADIENT, dp=1.2, minDist=r_min_px * 2.0,
        param1=100, param2=38, minRadius=r_min_px, maxRadius=r_max_px,
    )
    # سقفِ ۰.۷۸ (نه ۰.۹۸): روی هر ۱۱ عکس، دورترین دیسکِ واقعی از مرکز حداکثر
    # نسبتِ ۰.۶۸ دارد؛ اما بازتاب/لبه‌ی نورانیِ خودِ ظرف نزدیکِ لبه (نسبت‌های
    # ۰.۸+) گاهی به‌اشتباه به‌عنوانِ یک دایره‌ی کاذب شناسایی می‌شود -- این سقف
    # آن را حذف می‌کند بدون از دست‌دادنِ هیچ دیسکِ واقعی.
    # فیلترِ کنتراستِ روشنایی: دیسکِ کاغذیِ واقعی همیشه به‌وضوح از پس‌زمینه‌ی
    # محلی‌اش (آگار/هاله) روشن‌تر است -- دایره‌های کاذبِ نویزیِ آگارِ خالی (که با
    # پایین‌آوردنِ آستانه‌ی Hough برای گرفتنِ دیسک‌های کم‌کنتراست ظاهر می‌شوند)
    # حداکثر ۷ داشتند. آستانه‌ی ۱۰ (نه ۱۵) انتخاب شد چون دیسکِ زردرنگِ FM300 (که
    # کنتراستش نسبت به دیسک‌های سفید کمتر است) در دو عکس مقدارِ ۱۳-۱۵ داشت --
    # آستانه‌ی پایین‌تر آن را نگه می‌دارد و همچنان کاملاً بالاتر از کاذب‌هاست.
    MIN_BRIGHTNESS_CONTRAST = 10
    disks = []
    if circles is not None:
        for x, y, rad in circles[0]:
            gx, gy, grad = x / scale + x0, y / scale + y0, rad / scale
            if np.hypot(gx - cx, gy - cy) > r * 0.78:
                continue  # بیرونِ ظرف یا روی لبه/بازتابِ نورانیِ آن
            gxi, gyi = int(gx), int(gy)
            patch = gray[max(0, gyi - 25):gyi + 25, max(0, gxi - 25):gxi + 25]
            patch_bg = gray[max(0, gyi - 60):gyi + 60, max(0, gxi - 60):gxi + 60]
            if patch.size == 0 or patch_bg.size == 0:
                continue
            if float(patch.mean()) - float(patch_bg.mean()) < MIN_BRIGHTNESS_CONTRAST:
                continue  # نویزِ آگارِ خالی، نه دیسکِ واقعی
            disks.append((float(gx), float(gy), float(grad)))
    return disks


def clockwise_order(disks, cx, cy):
    def angle_from_12(x, y):
        ang = np.degrees(np.arctan2(x - cx, -(y - cy)))
        return ang % 360.0

    return sorted(disks, key=lambda d: angle_from_12(d[0], d[1]))


def annotate_image(path: str, csv_rows: list):
    name = os.path.basename(path)
    bgr = cv2.imread(path)
    if bgr is None:
        print(f"[رد شد] خواندنِ عکس ناموفق بود: {name}")
        return
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dishes = find_dishes(gray)
    if not dishes:
        print(f"[هشدار] هیچ ظرفِ پتری‌ای در {name} پیدا نشد -- نیازِ به بررسیِ دستی.")
        return

    overlay = bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, bgr.shape[1] / 1600.0)
    thick = max(2, int(bgr.shape[1] / 700))

    for dish_id, dish in enumerate(dishes, start=1):
        cx, cy, r = dish["cx"], dish["cy"], dish["r"]
        cv2.circle(overlay, (int(cx), int(cy)), int(r), (0, 200, 0), thick)
        disks = find_disks_in_dish(gray, dish)
        disks = clockwise_order(disks, cx, cy)
        # شعاعِ نمایشیِ ثابت (نسبت به قطرِ ظرف) به‌جای شعاعِ خامِ Hough -- شعاعِ
        # خام گاهی به‌جای لبه‌ی دیسک، لبه‌ی هاله را می‌گیرد (بی‌اثر روی x_px/y_px
        # که تنها مقادیرِ واقعاً استفاده‌شده در CSV هستند)، ولی برای بازرسیِ
        # بصریِ یکدست بهتر است اندازه‌ی ثابت رسم شود.
        marker_r = max(6, int(0.02 * 2 * r))
        for disk_number, (x, y, rad) in enumerate(disks, start=1):
            cv2.circle(overlay, (int(x), int(y)), marker_r, (0, 0, 255), thick)
            label = f"{disk_number}" if len(dishes) == 1 else f"D{dish_id}.{disk_number}"
            cv2.putText(overlay, label, (int(x) - 10, int(y) - marker_r - 8), font, scale,
                        (255, 255, 0), thick, cv2.LINE_AA)
            csv_rows.append({
                "image_file": name,
                "dish_id": dish_id,
                "disk_number": disk_number,
                "x_px": round(x, 1),
                "y_px": round(y, 1),
                "halo_diameter_mm_expert": "",
            })
        print(f"[{name}] ظرفِ #{dish_id}: {len(disks)} دیسک پیدا شد.")

    os.makedirs(OVERLAY_DIR, exist_ok=True)
    out_path = os.path.join(OVERLAY_DIR, os.path.splitext(name)[0] + "_numbered.jpg")
    cv2.imwrite(out_path, overlay)
    print(f"  -> overlay: {out_path}")


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    images = sorted(
        p for p in glob.glob(os.path.join(RAW_DIR, "*"))
        if p.lower().endswith(IMAGE_EXTENSIONS)
    )
    if not images:
        print(f"هیچ عکسی در {RAW_DIR} پیدا نشد. ابتدا ۱۱ عکس را آنجا کپی کنید.")
        return

    csv_rows = []
    for path in images:
        annotate_image(path, csv_rows)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_file", "dish_id", "disk_number", "x_px", "y_px",
                          "halo_diameter_mm_expert"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n{len(images)} عکس پردازش شد. CSV در: {CSV_PATH}")
    print("لطفاً هر overlay را در ground_truth/overlays/ بررسی کنید، سپس ستونِ "
          "halo_diameter_mm_expert را طبقِ کاغذِ متخصص پر کنید.")


if __name__ == "__main__":
    main()

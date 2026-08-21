"""
ابزار مشترک برای کش کردن خروجی مرحله‌ی تشخیص پتری (ماژول ۴ + ۴.۱).

چرا: ماژول‌های بعد از تشخیص دیسک (سگمنت هاله، رخدادها، شعاع) در این فاز از پروژه به‌طور
مکرر روی همه‌ی عکس‌ها تست می‌شوند، ولی خودِ تشخیص پتری (که پایدار و تایید‌شده است) هر بار
از نو اجرا می‌شود. این فایل دو تابع تولیدِ کدِ سلول (نه اجرای مستقیم) فراهم می‌کند که در
اسکریپت‌های تست (با همان الگوی nbformat+nbclient که در این نشست استفاده شده) به‌کار
می‌روند:

  - build_cache_cell_source(): برای *ساختن* کش -- بعد از اجرای سلول‌های ۰ تا ۱۲ نوت‌بوک
    (تنظیمات، توابع کمکی، بارگذاری عکس، ماژول ۴، ماژول ۴.۱)، این سلول `dishes` را به
    petri_cache/<نام‌عکس>/ سریالایز می‌کند.

  - load_cache_cell_source(image_path): برای *استفاده* از کش -- به‌جای اجرای سلول‌های
    بارگذاری عکس/ماژول ۴/۴.۱ (سلول‌های ۸ تا ۱۲)، این سلول مستقیماً `dishes`،
    `original_bgr`، `base_gray` را از کش بازسازی می‌کند تا اسکریپت‌های تست بتوانند
    بلافاصله سلول‌های ۱۳ به بعد (ماژول ۵ = تشخیص دیسک، به بعد) را روی آن‌ها اجرا کنند.

مهم: منطقِ خودِ تشخیص پتری اینجا تکرار/کپی نمی‌شود -- کش همیشه از اجرای واقعیِ نوت‌بوک
ساخته می‌شود (build_cache_cell_source صرفاً *بعد* از آن سلول‌ها اجرا می‌شود)، پس هیچ خطر
ناهماهنگی/drift بین کش و منطق واقعی نوت‌بوک وجود ندارد. اگر ماژول ۴/۴.۱ در نوت‌بوک تغییر
کند، فقط کافی است build_petri_cache.py دوباره اجرا شود.

فرمت ذخیره‌سازی (برای هر عکس، در petri_cache/<نام‌عکس بدون پسوند و کاراکترهای ناایمن>/):
  - meta.json: لیستی از دیکشنری‌های متادیتای هر پتری (تمام کلیدهای اسکالر/تاپل dish،
    به‌علاوه نام فایل‌های PNG مربوط به آرایه‌های همان پتری).
  - dish_{i}_roi_gray_masked.png, dish_{i}_processing_mask_roi.png,
    dish_{i}_precise_dish_mask.png: سه آرایه‌ی مستقلِ per-dish (فشرده‌سازی بدون افت با
    PNG). سایر فیلدهای آرایه‌ای (mask_full، roi_gray_precise_faded) عمداً ذخیره
    نمی‌شوند چون مستقیماً از همین سه + متادیتا با همان توابع کمکیِ خودِ نوت‌بوک (که در
    اسکریپتِ لودکننده هم از قبل تعریف شده‌اند) قابل بازسازیِ کاملاً یکسان هستند --
    ذخیره‌ی آن‌ها فقط حجم را بی‌دلیل ۲-۳ برابر می‌کرد.
"""

import os
import re

# مسیر مطلق (نه نسبی) -- چون kernel نوت‌بوک‌های تستی ممکن است با CWD متفاوتی اجرا شود؛
# لنگر گرفتن به محل خودِ این فایل (ریشه‌ی ریپازیتوری) قابل‌اتکاترین راه است.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "petri_cache")

# فیلدهای اسکالر/تاپلِ ساده‌ی dish که مستقیماً قابل JSON-serialize هستند (بدون آرایه).
_SCALAR_KEYS = [
    "index", "bbox", "roi_offset_xy", "center_roi_xy", "center_global_xy",
    "radius_px", "diameter_px", "confidence", "method", "status",
    "precise_dish_result",
]


def _safe_stem(image_path: str) -> str:
    stem = os.path.splitext(os.path.basename(image_path))[0]
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem)


def build_cache_cell_source(cache_root: str = CACHE_DIR) -> str:
    """کدی که -- بعد از سلول‌های ۰ تا ۱۲ نوت‌بوک -- `dishes` را به دیسک سریالایز می‌کند."""
    return f'''
import os, json, re, cv2

_cache_root = {cache_root!r}
_stem = re.sub(r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(cfg.image_path))[0])
_out_dir = os.path.join(_cache_root, _stem)
os.makedirs(_out_dir, exist_ok=True)

_meta = []
for d in dishes:
    _entry = {{k: d.get(k) for k in {_SCALAR_KEYS!r}}}
    # هر سه آرایه با PNG بدون افت -- امتحان شد که roi_gray_masked با JPEG (حتی کیفیت
    # ۹۵) ذخیره شود، ولی مقایسه‌ی مستقیمِ اجرای fresh در برابر cached (اسکریپت
    # validate_petri_cache.py) نشان داد حتی افتِ ناچیزِ JPEG روی این تصویر، از طریق
    # roi_gray_precise_faded به ماژول‌های ۵/۱۰/۱۰.۵ نشت می‌کند و امتیازهای Hough/Blob
    # را به‌طور سیستماتیک (نه فقط نویز جزئی) جابه‌جا می‌کند -- یعنی این مرحله دقیقاً
    # همان مرحله‌ای است که قرار است با این کش تست/تنظیم شود، پس هر افتی اینجا غیرقابل‌
    # قبول است. PNG بدون افت را نگه می‌داریم، با وجود حجم بیشتر.
    for arr_name, ext, params in (
        ("roi_gray_masked", "png", []),
        ("processing_mask_roi", "png", []),
        ("precise_dish_mask", "png", []),
    ):
        fname = f"dish_{{d['index']}}_{{arr_name}}.{{ext}}"
        cv2.imwrite(os.path.join(_out_dir, fname), d[arr_name], params)
        _entry[arr_name + "_file"] = fname
    _meta.append(_entry)

with open(os.path.join(_out_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(_meta, f, ensure_ascii=False, indent=1)

print(f"[petri_cache] {{len(_meta)}} پتری برای {{cfg.image_path}} در {{_out_dir}} ذخیره شد.")
'''


def load_cache_cell_source(image_path: str, cache_root: str = CACHE_DIR) -> str:
    """
    کدی که به‌جای سلول‌های ۸ تا ۱۲ (بارگذاری عکس + ماژول ۴ + ۴.۱)، مستقیماً `dishes`،
    `original_bgr`، `base_gray` را از کش می‌سازد. باید *بعد* از سلول‌های ۰ تا ۶ (تنظیمات
    + توابع کمکی، برای دسترسی به apply_border_fade_from_mask و cfg) اجرا شود.
    """
    stem = _safe_stem(image_path)
    return f'''
import os, json, cv2

cfg.image_path = r"{image_path}"
original_bgr = cv2.imread(cfg.image_path, cv2.IMREAD_COLOR)
if original_bgr is None:
    raise FileNotFoundError(f"Could not read image: {{cfg.image_path}}")
base_gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)

_in_dir = os.path.join({cache_root!r}, {stem!r})
with open(os.path.join(_in_dir, "meta.json"), "r", encoding="utf-8") as f:
    _meta = json.load(f)

dishes = []
for _entry in _meta:
    d = dict(_entry)
    for arr_name in ("roi_gray_masked", "processing_mask_roi", "precise_dish_mask"):
        fname = d.pop(arr_name + "_file")
        d[arr_name] = cv2.imread(os.path.join(_in_dir, fname), cv2.IMREAD_GRAYSCALE)
    d["bbox"] = tuple(d["bbox"])
    d["roi_offset_xy"] = tuple(d["roi_offset_xy"])
    d["center_roi_xy"] = tuple(d["center_roi_xy"])
    d["center_global_xy"] = tuple(d["center_global_xy"])
    # بازسازیِ فیلدهای مشتق‌شده (عمداً ذخیره نشده‌اند) -- دقیقاً با همان تابع کمکیِ
    # خودِ نوت‌بوک، پس با اجرای تازه‌ی ماژول ۴.۱ کاملاً یکسان است.
    d["roi_gray_precise_faded"] = apply_border_fade_from_mask(
        d["roi_gray_masked"], d["precise_dish_mask"], cfg.precise_dish_mask_fade_px
    )
    h, w = base_gray.shape[:2]
    _mask_full = np.zeros((h, w), dtype=np.uint8)
    x0, y0 = d["roi_offset_xy"]
    mh, mw = d["processing_mask_roi"].shape[:2]
    _mask_full[y0:y0 + mh, x0:x0 + mw] = d["processing_mask_roi"]
    d["mask_full"] = _mask_full
    d["roi_gray"] = base_gray[y0:y0 + mh, x0:x0 + mw]
    dishes.append(d)

print(f"[petri_cache] {{len(dishes)}} پتری برای {{cfg.image_path}} از کش بارگذاری شد.")
'''

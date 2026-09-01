"""
تستِ رگرسیون برایِ باگِ «Maximum allowed size exceeded» در ماژول‌هایِ ۱۵.۶ تا ۱۵.۹.

شرطی که باگ را فعال می‌کند: ماسکِ ظرف **هیچ پیکسلِ صفری** نداشته باشد -- که وقتی
ظرفِ پتری از لبه‌ی کادرِ عکس بیرون زده باشد رخ می‌دهد، چون آن‌وقت داخلِ ROI همه‌جا
ظرف است.

در آن حالت `cv2.distanceTransform` مقدارِ FLT_MAX (۳٫۴e38) برمی‌گرداند، چون فاصله تا
«نزدیک‌ترین پیکسلِ صفر» را می‌سنجد و پیکسلِ صفری وجود ندارد. آن عدد به‌عنوانِ شعاعِ
بیرونیِ جست‌وجو استفاده می‌شد و `np.linspace` را با ~۴e37 حلقه صدا می‌زد.

اجرا:  python3 ground_truth/diagnostics/test_dt_no_zero_pixel.py
انتظار: PASS. پیش از اصلاح، همین تست FAIL با همان ValueError می‌داد.
"""
import nbformat, numpy as np, cv2, sys
nb=nbformat.read("/home/user/Antibiogram/disk_detection_pipeline_modular.ipynb",as_version=4)
ns={"np":np,"cv2":cv2}
for c in nb.cells:
    if c["cell_type"]!="code": continue
    s="".join(c["source"])
    try: exec(compile(s,"<cell>","exec"), ns)
    except Exception: pass
    if "segment_halos_statistical" in ns and "cfg" in ns: break
if "segment_halos_statistical" not in ns or "cfg" not in ns:
    print("SKIP — تابع/کانفیگ در دسترس نشد"); sys.exit(0)
H=W=400
canvas=np.full((H,W),150,np.uint8)
mask=np.full((H,W),255,np.uint8)      # ← بدونِ هیچ پیکسلِ صفر: شرطِ عکسِ کاربر
try:
    r=ns["segment_halos_statistical"](canvas, mask, mask,
        [{"x":200.0,"y":200.0,"r":40.0}], {"mean":150.0,"std":5.0}, ns["cfg"])
    print("PASS — اجرا شد، status =", r["status"])
except Exception as e:
    print(f"FAIL — {type(e).__name__}: {e}")

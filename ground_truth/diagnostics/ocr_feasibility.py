"""
امکان‌سنجیِ خواندنِ کدِ آنتی‌بیوتیک رویِ دیسک با OCR.

پرسش: آیا افزودنِ OCR می‌تواند به «مرجعِ بهتر» برساند؟

## جوابِ کوتاه: نه، دستِ‌کم نه به آن معنایی که مسئله‌ی ما دارد

نویزِ ±۲ میلی‌متر مالِ **خوانشِ کولیسیِ کارشناس از لبه‌ی هاله** است. برچسبِ رویِ
دیسک هیچ ربطی به آن اندازه‌گیری ندارد. خواندنِ کد، آنتی‌بیوتیک را می‌دهد نه قطرِ
دقیق‌ترِ هاله، پس تکرارپذیریِ مرجع را عوض نمی‌کند.

## ولی چیزهایِ دیگری می‌دهد که ارزشمندند

۱. ماژولِ ۱۸ (طبقه‌بندیِ EUCAST) خودکار می‌شود؛ الان آنتی‌بیوتیکِ هر دیسک باید
   دستی در `cfg.eucast_disk_agents` وارد شود.
۲. ارزیابیِ دسته‌ای **دقیق** می‌شود به‌جایِ میانگین‌گیری رویِ ۳۴۹ نقطه‌ی شکست.
   این بهبودِ واقعی در کیفیتِ *ارزیابی* است (نه در کیفیتِ مرجع).
۳. تنها مسیرِ غیرمستقیم به مرجعِ بهتر: با دانستنِ آنتی‌بیوتیک + گونه، بازه‌هایِ
   کنترلِ کیفیِ EUCAST و سازگاریِ بینِ پلیت‌ها اجازه می‌دهند خوانش‌هایِ پرتِ *خودِ
   کارشناس* علامت بخورند. مسیرِ واقعی ولی ضعیفی است.

## نتیجه‌ی امکان‌سنجی

کدها برایِ چشمِ انسان در همین رزولوشن (~۱۳.۳ پیکسل بر میلی‌متر) **خوانا هستند** --
از ۹ دیسکِ عکسِ gt_01، ۸ تا دستی خوانده شد: RA 5، TE 30، FOX 30، GM 10، SXT،
CP 5، E 15، CC 2 (تصویر: disk_labels_gt01.png).

ولی OCRِ آماده کاملاً شکست می‌خورد: **۰ از ۸** در دو تلاشِ جدی، شاملِ ماسکِ دایره‌ای
دیسک، آستانه‌ی Otsu معکوس، بزرگ‌نماییِ ۶ برابر، CLAHE، و راست‌سازی با
`minAreaRect` در چهار جهت با psm ۷ و ۸. بدتر از آن، Tesseract با **اطمینانِ بالا
اشتباه** می‌کند (اطمینانِ ۸۵ برایِ «G» رویِ دیسکی که «TE 30» است).

علتش ساختاری است، نه تنظیمات: OCRِ عمومی برایِ متنِ چاپیِ افقیِ سند آموزش دیده،
درحالی‌که این‌ها مهرِ دستیِ منحنی، با چرخشِ کاملاً تصادفی، ارتفاعِ نویسه ~۲۰ پیکسل،
و گاهی بسیار کم‌رنگ‌اند (SXT).

**ابزارِ درست OCR نیست، طبقه‌بندیِ واژگانِ بسته است:** مجموعه‌ی کدها بسته و کوچک
است (~۳۰ تا ۴۰ کد)، پس مسئله «تشخیصِ متنِ آزاد» نیست بلکه «انتخاب از یک فهرستِ
معلوم» است -- که به‌مراتب ساده‌تر است.

**ولی یک مشکلِ مرغ‌و‌تخم‌مرغ دارد:** آموزش یا اعتبارسنجیِ چنین طبقه‌بندی به برچسبِ
مرجعِ کدها نیاز دارد، که در دیتاستِ ما وجود ندارد. کسی باید اول آن‌ها را برچسب
بزند -- و اگر بزند، برایِ همین ۱۱ پلیت دیگر به OCR نیازی نیست.

اجرا:  python3 ground_truth/diagnostics/ocr_feasibility.py
نیازمند: tesseract-ocr و pytesseract (هیچ‌کدام وابستگیِ خودِ پایپلاین نیستند).
"""
import cv2, numpy as np, pytesseract, csv, math
from collections import defaultdict
gt=defaultdict(list)
for r in csv.DictReader(open('/home/user/Antibiogram/ground_truth/ground_truth_expert_readings.csv')):
    gt[r['image_file']].append((float(r['x_px']),float(r['y_px'])))
img=cv2.imread('/home/user/Antibiogram/ground_truth/raw_images/gt_01.jpg',cv2.IMREAD_GRAYSCALE)
truth=['RA 5','TE 30','FOX 30','GM 10','SXT','CP 5','E 15','CC 2','?']
CFG='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

def prep(x,y,r=46):
    c=img[max(0,y-r):y+r, max(0,x-r):x+r].copy()
    # ماسکِ دایره‌ایِ دیسک تا بافتِ آگارِ اطراف واردِ آستانه‌گذاری نشود
    m=np.zeros(c.shape,np.uint8); cv2.circle(m,(c.shape[1]//2,c.shape[0]//2),int(r*0.92),255,-1)
    c=np.where(m>0,c,255).astype(np.uint8)
    c=cv2.resize(c,(0,0),fx=6,fy=6,interpolation=cv2.INTER_CUBIC)
    # متنِ تیره رویِ کاغذِ روشن -> آستانه‌ی Otsu معکوس
    _,b=cv2.threshold(cv2.GaussianBlur(c,(5,5),0),0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    b=cv2.morphologyEx(b,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    return c,b

def deskew_angles(b):
    """زاویه‌ی اصلیِ توده‌ی متن از رویِ جعبه‌ی چرخیده‌ی همه‌ی پیکسل‌هایِ تیره."""
    pts=cv2.findNonZero(b)
    if pts is None or len(pts)<50: return [0]
    (_,_),(w,h),ang=cv2.minAreaRect(pts)
    if w<h: ang+=90
    return [ang, ang+90, ang+180, ang+270]

ok=0
for k,(x,y) in enumerate(gt['gt_01.jpg'][:9]):
    c,b=prep(int(x),int(y))
    best=('',0,0)
    cands=deskew_angles(b)
    for ang in cands:
        M=cv2.getRotationMatrix2D((c.shape[1]/2,c.shape[0]/2),ang,1.0)
        rot=cv2.warpAffine(c,M,(c.shape[1],c.shape[0]),borderValue=255,flags=cv2.INTER_CUBIC)
        for psm in (7,8):
            d=pytesseract.image_to_data(rot,config=f'--psm {psm} '+CFG,output_type=pytesseract.Output.DICT)
            for t,cf in zip(d['text'],d['conf']):
                t=t.strip()
                if len(t)>=1 and float(cf)>best[1]: best=(t,float(cf),ang)
    hit = best[0] and best[0] in truth[k].replace(' ','')
    ok+=bool(hit)
    print('disk %d  truth=%-8s  best_ocr=%-8s conf=%2.0f  ang=%3.0f  %s'%(k+1,truth[k],best[0] or '-',best[1],best[2],'OK' if hit else ''))
print('\nتعدادِ درست (زیررشته‌ی کدِ واقعی): %d از ۸ دیسکِ خوانا'%ok)

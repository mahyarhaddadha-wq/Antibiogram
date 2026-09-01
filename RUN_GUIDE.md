# راهنمای اجرای آفلاین

سه کار می‌توانید انجام دهید، و برای هرکدام یک اسکریپت جدا هست:

| می‌خواهم... | اسکریپت |
|---|---|
| یک عکس را پردازش کنم و خروجی‌هایش را ببینم | `run_single_image.py` |
| همه‌ی عکس‌های یک پوشه را پردازش کنم | `batch_process_antibiogram.py` |
| دقت سیستم را روی ۱۱ عکس مرجع بسنجم | `ground_truth/evaluate_pipeline.py` |

هر سه دقیقاً همان نوت‌بوک `disk_detection_pipeline_modular.ipynb` را اجرا می‌کنند.
**نوت‌بوک تنها منبع حقیقت الگوریتم است** — این اسکریپت‌ها آن را تغییر نمی‌دهند،
فقط مسیر عکس را جایگزین می‌کنند و کل نوت‌بوک را با یک کرنل تازه اجرا می‌کنند. پس
هر تغییری در نوت‌بوک بدهید، بلافاصله در هر سه اثر می‌کند.

---

## گام ۱ — نصب پیش‌نیازها (فقط یک‌بار)

پنجره‌ی PowerShell را باز کنید و بروید داخل پوشه‌ی پروژه:

```powershell
cd "D:\مسیر\پوشه\Antibiogram"
```

بعد:

```powershell
pip install nbformat nbclient opencv-python numpy matplotlib pymupdf
```

## گام ۲ — پیدا کردن نام کرنل

اسکریپت‌ها نوت‌بوک را با یک «کرنل» Jupyter اجرا می‌کنند و باید نامش را بدانید:

```powershell
jupyter kernelspec list
```

خروجی چیزی شبیه این است:

```
Available kernels:
  python3      C:\Users\ASA\AppData\Roaming\jupyter\kernels\python3
```

آن کلمه‌ی اول (`python3` یا هر چیز دیگر) نام کرنل شماست. در همه‌ی دستورهای بعدی
جای `python3` همان را بگذارید.

> **اگر `jupyter` شناخته نشد:** یعنی در PATH نیست. به‌جایش این را بزنید:
> `python -m jupyter kernelspec list`
> و اگر باز هم نشد، اول `pip install jupyter` و سپس این را اجرا کنید تا کرنل ساخته شود:
> `python -m ipykernel install --user --name python3`

---

## کار ۱: پردازش یک عکس

```powershell
python run_single_image.py --image "D:\antibiogram\input\plate1.jpg" --kernel python3
```

خروجی‌ها در `output\plate1\` ساخته می‌شوند:

| فایل | چیست |
|---|---|
| `01_fusion_disks.png` | دیسک‌های تشخیص‌داده‌شده (بعد از اعتبارسنجی شکل، ماژول ۱۱.۵) |
| `02_halo_dish1.png` | مرز نهایی هاله‌ها (ماژول ۱۶.۶) |
| `03_bubbles_dish1.png` | حباب/رخداد داخل هاله (ماژول ۱۷) |
| `04_final_report.txt` | گزارش متنی: قطر دیسک، قطر هاله، حباب، دسته‌ی EUCAST |
| `05_eucast.txt` | جزئیات طبقه‌بندی بالینی S/I/R |
| `06_full_log.txt` | کل خروجی متنی اجرا (برای وقتی چیزی خطا داد) |

اگر پوشه‌ی خروجی را جای دیگری می‌خواهید:

```powershell
python run_single_image.py --image "D:\antibiogram\input\plate1.jpg" `
                           --output-dir "D:\antibiogram\output" `
                           --kernel python3
```

> در PowerShell علامت ادامه‌ی خط ` ` ` ` (backtick) است. در Command Prompt قدیمی `^`.
> اگر گیج‌کننده است، همه را در **یک خط** بنویسید — کاملاً درست کار می‌کند.

### طبقه‌بندی EUCAST (اختیاری)

برای این‌که سیستم به‌جای عدد، دسته‌ی `S` / `I` / `R` بدهد، باید دو چیز را به آن
بگویید که **از روی عکس قابل تشخیص نیستند**: گونه‌ی باکتری، و آنتی‌بیوتیک هر دیسک.

```powershell
python run_single_image.py --image plate1.jpg --kernel python3 --organism "Enterobacterales" --agents "1=Ciprofloxacin,3=Gentamicin,5=Meropenem"
```

- `--organism` یکی از نام‌های ستون اول فایل
  `ground_truth\eucast\eucast_v16_zone_breakpoints.csv` است
  (مثلاً `Enterobacterales`، `Staphylococcus spp.`، `Pseudomonas spp.`).
- `--agents` شماره‌ی دیسک (همان شماره‌ای که در گزارش می‌بینید) به نام آنتی‌بیوتیک.
  لازم نیست همه‌ی دیسک‌ها را بدهید.

**اگر ندهید هیچ خطایی نمی‌دهد** — فقط به‌جای دسته می‌نویسد «دسته اعلام نشد» و
درصد دسته‌های ممکن را نشان می‌دهد، که صادقانه‌تر از حدس زدن آنتی‌بیوتیک است.

---

## کار ۲: پردازش دسته‌ای یک پوشه

```powershell
python batch_process_antibiogram.py --input-dir "D:\antibiogram\input" --output-dir "D:\antibiogram\output" --kernel python3
```

برای هر عکس یک زیرپوشه با همان فایل‌های بالا ساخته می‌شود، به‌علاوه‌ی یک
`summary.txt` در ریشه که وضعیت همه‌ی عکس‌ها را می‌گوید.

اگر همه‌ی پلیت‌های آن پوشه یک گونه‌ی باکتری دارند:

```powershell
python batch_process_antibiogram.py --input-dir "D:\antibiogram\input" --output-dir "D:\antibiogram\output" --kernel python3 --organism "Enterobacterales"
```

> **زمان اجرا:** هر عکس حدود **۱ تا ۳ دقیقه** طول می‌کشد (کل نوت‌بوک برای هر عکس
> از صفر اجرا می‌شود). برای ۷۸ عکس یعنی حدود دو تا سه ساعت. اگر شبانه اجرایش
> کنید راحت‌تر است.

---

## کار ۳: سنجش دقت روی ۱۱ عکس مرجع

این همان کدی است که همه‌ی اعداد گزارش‌شده در `EVALUATION.md` از آن آمده‌اند.

```powershell
python ground_truth\evaluate_pipeline.py --kernel python3
```

یا فقط چند عکس خاص (سریع‌تر، برای تست):

```powershell
python ground_truth\evaluate_pipeline.py --kernel python3 --images gt_01.jpg gt_07.jpg
```

**خروجی:**

- گزارش کامل در ترمینال: تشخیص پتری، TP/FP/FN و Precision/Recall/F1 برای دیسک،
  حضور/عدم هاله، و MAE/Bias/Bland–Altman برای قطر هاله.
- `ground_truth\evaluation_results.csv` — جزئیات خام هر دیسک (در اکسل باز کنید).
- `ground_truth\pipeline_overlays\<نام>_pipeline_halo.png` — تصویر خروجی پایپلاین.
  **برای مقایسه‌ی چشمی، این را کنار `ground_truth\marked_images\marked_XX.jpg`
  (همان عکس ماژیک‌خورده‌ی خودتان) بگذارید.**

اگر نمی‌خواهید منتظر ذخیره‌ی تصاویر بمانید: `--no-overlays` را اضافه کنید.

> ⚠️ **این اسکریپت را وقتی نوت‌بوک را باز و در حال ویرایش دارید اجرا نکنید.** برای
> هر عکس نوت‌بوک را از روی دیسک می‌خواند، پس اگر وسط اجرا ذخیره‌اش کنید نتیجه
> بی‌اعتبار می‌شود. (این اشتباه یک بار در خود پروژه اتفاق افتاد.)

---

## تحلیل‌های EUCAST (اختیاری، بدون نیاز به کرنل)

این دو مستقل از نوت‌بوک‌اند و در چند ثانیه اجرا می‌شوند:

```powershell
python ground_truth\eucast\parse_breakpoints.py
python ground_truth\eucast\categorical_agreement.py
python ground_truth\eucast\expert_repeatability.py
```

اولی جدول نقاط شکست را از PDF رسمی EUCAST می‌سازد، دومی خطای میلی‌متری را به
توافق دسته‌ای (CA/VME/ME) ترجمه می‌کند، سومی اثر تکرارپذیری ±۲ میلی‌متری کارشناس
را حساب می‌کند.

---

## عیب‌یابی

| پیام | علت و راه‌حل |
|---|---|
| `NoSuchKernel: python3` | نام کرنل اشتباه است. `jupyter kernelspec list` را بزنید و نام درست را با `--kernel` بدهید. |
| `ModuleNotFoundError: cv2` | opencv در **کرنل** نصب نیست (نه در پایتون اصلی). داخل همان محیطی که کرنل از آن ساخته شده: `pip install opencv-python`. |
| `jupyter is not recognized` | به‌جایش `python -m jupyter ...` را بزنید. |
| `نوت‌بوک پیدا نشد` | اسکریپت را از داخل پوشه‌ی ریپازیتوری اجرا کنید، یا مطمئن شوید فایل `.ipynb` کنار اسکریپت است. |
| اجرا خیلی طول کشید | طبیعی است؛ هر عکس ۱ تا ۳ دقیقه. با `--images` فقط چند عکس را تست کنید. |
| خروجی خالی/ناقص | `06_full_log.txt` را ببینید؛ خطای واقعی آن‌جاست. |

---

## چند نکته که موقع خواندن نتایج بدانید

اینها در `EVALUATION.md` و `ROADMAP.md` کامل توضیح داده شده‌اند، ولی خلاصه‌شان:

- **تشخیص دیسک خوب است:** Precision=۱.۰۰۰، Recall=۰.۹۷۸، F1=۰.۹۸۹.
- **اندازه‌گیری هاله متوسط است:** MAE=۳.۸۵mm، ولی **میانه**=۲.۰۸mm — یعنی نیمی از
  اندازه‌گیری‌ها در باند ±۲ میلی‌متری خود کارشناس هستند و میانگین را اقلیتی از
  خطاهای بزرگ بالا می‌کشد.
- **۱۴ مثبت کاذب هاله** هنوز حل‌نشده است.
- **دسته‌ی بالینی:** CA=۸۲.۵٪ در برابر سقف انسانی ۹۲–۹۶٪.
- سیستم در وضع فعلی **ابزار غربالگری** است، نه ابزار تصمیم بالینی.

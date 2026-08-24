# Evaluation Methodology and Results

*(For the Persian version, see [بخش فارسی](#روش‌ارزیابی-و-نتایج-فارسی) below.)*

## 1. Ground-truth dataset

Eleven photographs of real antibiotic-susceptibility (Kirby–Bauer / disk-diffusion) plates were used to construct an independent reference dataset, so that the accuracy of the automated pipeline could be assessed against measurements the pipeline itself played no part in producing.

**Disk localization (non-circular by construction).** Disk positions in the ground-truth set were located with a minimal, standalone Hough-circle detector (`ground_truth/annotate_disks.py`), not with the Fusion pipeline under evaluation. This is a deliberate methodological choice: had the same system that is being evaluated also been used to define which disks exist and where, any systematic detection error of that system would have been silently absorbed into the ground truth rather than exposed by it. Each disk was numbered clockwise from the 12 o'clock position relative to the plate centre, and every numbered overlay was visually inspected before being used, with any missed or spurious detection corrected by hand.

**Expert reading.** For each numbered disk, a laboratory expert measured the inhibition-zone (halo) diameter directly on a printed copy of the photograph (hand-marked in pen) and the value was transcribed into `ground_truth/ground_truth_expert_readings.csv`. Disk diameter was fixed at the manufacturer-standard 6.0 mm for every disk (per the documented physical assumption of this project — antibiotic disks used here have only two possible physical sizes, and the disks photographed were all of the 6 mm class).

**Resulting dataset.** 93 disks across 11 plate photographs; 64 disks carry an expert-measured halo diameter, 29 were confirmed by the expert to show no inhibition zone.

## 2. Evaluation procedure

The evaluation script (`ground_truth/evaluate_pipeline.py`) executes the **unmodified** production notebook (`disk_detection_pipeline_modular.ipynb`) programmatically, once per image, via `nbclient.NotebookClient` — the identical code path used in normal operation, with only the input image path overridden. No evaluation-specific branch of the pipeline exists; this guarantees that the numbers below describe the system as actually deployed, not a specially tuned variant.

**Disk matching.** System-reported disks are matched to ground-truth disks by nearest **position** only — never by internal disk index or detection order, since the pipeline's internal ordering is unrelated to the ground truth's clockwise numbering. The match radius is not a fixed pixel constant; it is set per image to one half of that image's minimum inter-disk distance in the ground truth, keeping the criterion scale-relative rather than tied to a specific camera or resolution.

**Disk-detection accuracy.** Matched/unmatched status yields true positives (TP), false positives (FP, a system disk matching no ground-truth disk) and false negatives (FN, a ground-truth disk matched by no system disk); precision, recall and F1 are computed over all 93 ground-truth disks pooled across the 11 images.

**Halo-presence accuracy.** Restricted to the subset of disks correctly matched by position (so that a missed *disk* is not conflated with a missed *halo*). A 2×2 confusion matrix (TP/FN/FP/TN) records whether the system reports an inhibition zone where the expert did or did not observe one.

**Halo-diameter accuracy.** Computed only on the subset where both the expert and the system report a halo (the presence-TP subset), as mean absolute error (MAE), mean signed error (bias), standard deviation, and the 95% Bland–Altman limits of agreement.

**Disk-diameter accuracy.** Computed analogously over all correctly matched disks, against the fixed 6.0 mm reference. Note: the pipeline's own pixel→millimetre calibration is derived from the very same disks (the tightest cluster of disk diameters in pixels, within a 30% tolerance band, averaged and mapped onto the known 6.0 mm standard — see `_estimate_px_per_mm_from_disks` in the notebook). Disk-diameter accuracy is therefore a self-consistency check on that calibration step, not an independent validation of it; the low residual error mainly confirms the *disks within one plate are of consistent apparent size*, which the calibration step already assumes.

## 3. Results

**Table 1 — Petri dish detection.** 11/11 photographs (100%) had exactly one disk-bearing petri dish correctly localized.

**Table 2 — Disk detection** (93 ground-truth disks, 11 images)

| TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 91 | 1 | 2 | 0.989 | 0.978 | 0.984 |

**Table 3 — Halo-presence confusion matrix** (n = 91 correctly matched disks)

| | Expert: halo present | Expert: no halo |
|---|---:|---:|
| **System: halo present** | 57 (TP) | 14 (FP) |
| **System: no halo** | 6 (FN) | 14 (TN) |

Derived: accuracy = 0.780, precision = 0.803, recall = 0.905.

**Table 4 — Halo-diameter accuracy** (n = 57, both expert and system report a halo)

| MAE | Bias (mean signed error) | SD | 95% limits of agreement |
|---:|---:|---:|---:|
| 5.50 mm | −0.53 mm | 7.95 mm | [−16.12, +15.05] mm |

**Table 5 — Disk-diameter accuracy** (n = 91, reference = 6.0 mm)

| MAE | Bias | SD |
|---:|---:|---:|
| 0.17 mm | −0.00 mm | 0.23 mm |

## 4. Independent reproduction

These results were independently reproduced, unchanged, on a second, unrelated machine (a Windows workstation, fresh standalone Python 3.11.9 installation, no shared environment or configuration with the development machine) by re-running the identical, unmodified `evaluate_pipeline.py` script end to end. The reproduction returned numerically identical summary statistics (same TP/FP/FN counts, same MAE/bias/SD to two decimal places), which is consistent with the pipeline's stated design goal of full determinism — the same input image always yields the same output, with no randomized or non-reproducible step in the detection or measurement code path.

## 5. Limitations (stated for transparency)

- **Sample size.** 11 images and 93 disks is a modest first evaluation set; the numbers above should be read as an initial accuracy characterization, not a definitive validation, pending assessment on a larger and more varied image set.
- **Halo-diameter error is moderate (MAE 5.50 mm)** and is concentrated in specific, identified failure modes — most notably signal leakage from a genuinely large inhibition zone into an adjacent disk's background estimate, and a residual pure-illumination-gradient (vignetting) effect in some plates — both under active investigation and documented in `PROJECT_HISTORY.md`.
- **The presence-classification threshold trades recall for precision.** The statistical reliability gate used to decide "zone present vs. absent" (a minimum contrast-to-noise ratio) was tightened in the most recent revision to remove a large class of false positives; this measurably improved specificity (14 true negatives, up from 0) but cost 6 new false negatives — real, low-contrast zones whose signal did not clear the same threshold. This trade-off is explicit and not yet fully resolved.
- **Disk-diameter accuracy reflects a self-consistency check**, not an independently calibrated measurement, for the reason given in §2.

---

# روش‌ارزیابی و نتایج (فارسی)

## ۱) دیتاست مرجع (Ground Truth)

یازده عکس از پلیت‌های واقعیِ آنتی‌بایوگرام (روش انتشار دیسک/Kirby–Bauer) برای ساختِ یک دیتاستِ مرجعِ مستقل استفاده شد — مستقل به این معنا که خودِ سیستمِ موردِ ارزیابی هیچ نقشی در تعیینِ این‌که «حقیقتِ زمینی» چیست نداشته باشد.

**مکان‌یابیِ دیسک‌ها (به‌طورِ ساختاری غیرِدایره‌ای/مستقل).** موقعیتِ دیسک‌ها در دیتاستِ مرجع با یک تشخیص‌دهنده‌ی سادهِ و مستقلِ Hough Circle (`ground_truth/annotate_disks.py`) پیدا شد، نه با پایپلاینِ Fusion که قرار است ارزیابی شود. این یک انتخابِ روش‌شناختیِ آگاهانه است: اگر همان سیستمی که ارزیابی می‌شود در تعریفِ این‌که «چه دیسکی کجاست» هم دخالت داشت، هر خطایِ سیستماتیکِ آن سیستم به‌جایِ آشکارشدن، در خودِ حقیقتِ زمینی جذب می‌شد. هر دیسک از ساعتِ ۱۲ نسبت به مرکزِ پلیت، به‌ترتیبِ ساعت‌گرد شماره‌گذاری شد و هر overlایِ شماره‌گذاری‌شده پیش از استفاده به‌صورتِ بصری بازبینی شد؛ هر موردِ ازدست‌رفته یا کاذب دستی تصحیح گردید.

**اندازه‌گیریِ کارشناسی.** برایِ هر دیسکِ شماره‌گذاری‌شده، یک کارشناسِ آزمایشگاه قطرِ هالهِ مهار را مستقیماً رویِ نسخه‌ی چاپ‌شده‌ی عکس (با خودکار/ماژیک) اندازه گرفت و مقدار در `ground_truth/ground_truth_expert_readings.csv` ثبت شد. قطرِ خودِ دیسک برایِ همه‌ی دیسک‌ها برابرِ استانداردِ کارخانه‌ای ۶.۰ میلی‌متر در نظر گرفته شد (طبقِ مفروضِ مستندِ این پروژه: دیسک‌هایِ آنتی‌بیوتیکِ استفاده‌شده در این مجموعه فقط از خانواده‌ی ۶ میلی‌متری بودند).

**دیتاستِ نهایی.** ۹۳ دیسک در ۱۱ عکس؛ ۶۴ دیسک دارایِ قطرِ هالهِ اندازه‌گیری‌شده‌یِ کارشناسی، و ۲۹ دیسک که کارشناس تاییدِ «بدونِ هاله» بودنشان را کرده.

## ۲) روشِ ارزیابی

اسکریپتِ ارزیابی (`ground_truth/evaluate_pipeline.py`) خودِ نوت‌بوکِ نهاییِ پروژه (`disk_detection_pipeline_modular.ipynb`) را **بدونِ هیچ تغییری** و به‌ازایِ هر عکس یک‌بار، از طریقِ `nbclient.NotebookClient` اجرا می‌کند — دقیقاً همان مسیرِ کدی که در استفاده‌ی معمولی اجرا می‌شود، فقط با override‌کردنِ مسیرِ فایلِ تصویرِ ورودی. هیچ شاخه‌ی مخصوصِ ارزیابی در خودِ پایپلاین وجود ندارد؛ این تضمین می‌کند اعدادِ زیر توصیف‌کننده‌یِ سیستمی است که واقعاً استفاده می‌شود، نه یک نسخه‌ی مخصوصِ تنظیم‌شده.

**تطبیقِ دیسک‌ها.** دیسک‌هایِ گزارش‌شده توسطِ سیستم فقط بر اساسِ نزدیک‌ترین **موقعیت** با دیسک‌هایِ مرجع تطبیق داده می‌شوند — هرگز بر اساسِ شماره‌یِ داخلی یا ترتیبِ تشخیص، چون ترتیبِ داخلیِ پایپلاین ربطی به شماره‌گذاریِ ساعت‌گردِ مرجع ندارد. شعاعِ تطبیق یک مقدارِ پیکسلیِ ثابت نیست؛ برایِ هر عکس برابرِ نصفِ کمترین فاصله‌ی بینِ دو دیسکِ مرجع در همان عکس تعیین می‌شود تا معیار نسبی/مقیاس‌آگاه بماند، نه وابسته به یک دوربین یا رزولوشنِ خاص.

**دقتِ تشخیصِ دیسک.** وضعیتِ تطبیق/عدم‌تطبیق، TP/FP/FN را می‌دهد؛ Precision، Recall و F1 رویِ مجموعِ ۹۳ دیسکِ مرجعِ هر ۱۱ عکس محاسبه می‌شود.

**دقتِ حضورِ هاله.** فقط رویِ زیرمجموعه‌یِ دیسک‌هایی که به‌درستی از نظرِ موقعیت تطبیق یافته‌اند محاسبه می‌شود (تا یک دیسکِ ازدست‌رفته با یک هاله‌یِ ازدست‌رفته قاطی نشود). یک ماتریسِ درهم‌ریختگیِ ۲×۲ (TP/FN/FP/TN) ثبت می‌کند که آیا سیستم دقیقاً همان‌جایی هاله گزارش داده که کارشناس دیده یا ندیده.

**دقتِ عددیِ قطرِ هاله.** فقط رویِ زیرمجموعه‌ای محاسبه می‌شود که هم کارشناس و هم سیستم هاله گزارش داده‌اند (همان زیرمجموعه‌یِ TP حضور)، به‌صورتِ MAE، بایاسِ میانگین (خطایِ علامت‌دار)، انحرافِ‌معیار، و حدودِ توافقِ ۹۵٪ِ Bland–Altman.

**دقتِ عددیِ قطرِ دیسک.** به همین شکل رویِ همه‌یِ دیسک‌هایِ به‌درستی‌تطبیق‌یافته، در برابرِ مرجعِ ثابتِ ۶.۰ میلی‌متر محاسبه می‌شود. نکته: کالیبراسیونِ px→mm خودِ پایپلاین از رویِ همین دیسک‌ها به دست می‌آید (فشرده‌ترین خوشه‌یِ قطرِ دیسک‌ها به پیکسل، در بازه‌یِ تحملِ ۳۰٪، میانگین‌گیری‌شده و معادلِ استانداردِ ۶.۰mm گذاشته می‌شود — تابعِ `_estimate_px_per_mm_from_disks` در نوت‌بوک). پس دقتِ قطرِ دیسک بیشتر یک بررسیِ خودسازگاریِ همین گامِ کالیبراسیون است، نه یک اعتبارسنجیِ کاملاً مستقل از آن؛ خطایِ باقی‌مانده‌یِ کم عمدتاً تاییدِ این است که «دیسک‌هایِ داخلِ یک پلیتِ واحد اندازه‌یِ ظاهریِ یکسانی دارند» — که خودِ گامِ کالیبراسیون هم دقیقاً همین فرض را دارد.

## ۳) نتایج

**جدولِ ۱ — تشخیصِ پتری.** ۱۱ از ۱۱ عکس (۱۰۰٪) دقیقاً یک پتریِ دارایِ دیسک به‌درستی مکان‌یابی شد.

**جدولِ ۲ — تشخیصِ دیسک** (۹۳ دیسکِ مرجع، ۱۱ عکس)

| TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| ۹۱ | ۱ | ۲ | ۰.۹۸۹ | ۰.۹۷۸ | ۰.۹۸۴ |

**جدولِ ۳ — ماتریسِ درهم‌ریختگیِ حضورِ هاله** (n = ۹۱ دیسکِ به‌درستی‌تطبیق‌یافته)

| | کارشناس: هاله دارد | کارشناس: بدونِ هاله |
|---|---:|---:|
| **سیستم: هاله دارد** | ۵۷ (TP) | ۱۴ (FP) |
| **سیستم: بدونِ هاله** | ۶ (FN) | ۱۴ (TN) |

مقادیرِ مشتق‌شده: دقتِ کلی (accuracy) = ۰.۷۸۰، Precision = ۰.۸۰۳، Recall = ۰.۹۰۵.

**جدولِ ۴ — دقتِ عددیِ قطرِ هاله** (n = ۵۷، هم کارشناس هم سیستم هاله گزارش داده‌اند)

| MAE | Bias (خطایِ میانگینِ علامت‌دار) | SD | حدودِ توافقِ ۹۵٪ |
|---:|---:|---:|---:|
| ۵.۵۰ mm | −۰.۵۳ mm | ۷.۹۵ mm | [−۱۶.۱۲, +۱۵.۰۵] mm |

**جدولِ ۵ — دقتِ عددیِ قطرِ دیسک** (n = ۹۱، مرجع = ۶.۰mm)

| MAE | Bias | SD |
|---:|---:|---:|
| ۰.۱۷ mm | −۰.۰۰ mm | ۰.۲۳ mm |

## ۴) بازتولیدپذیریِ مستقل

این نتایج به‌طورِ مستقل و بدونِ هیچ تغییری رویِ یک سیستمِ دوم و کاملاً بی‌ربط (یک لپ‌تاپِ ویندوزی، نصبِ تازه و مستقلِ Python 3.11.9، بدونِ هیچ محیط یا تنظیمِ مشترک با ماشینِ توسعه) با اجرایِ مجددِ همان اسکریپتِ بدون‌تغییرِ `evaluate_pipeline.py` بازتولید شد. این بازتولید دقیقاً همان آمارِ خلاصه را داد (همان تعدادِ TP/FP/FN، همان MAE/Bias/SD تا دو رقمِ اعشار) — سازگار با هدفِ طراحیِ مستندِ پایپلاین مبنی‌بر قطعیتِ کامل (determinism): یک تصویرِ ورودیِ یکسان همیشه خروجیِ یکسان می‌دهد، بدونِ هیچ گامِ تصادفی یا غیرِقابل‌بازتولید در مسیرِ کدِ تشخیص یا اندازه‌گیری.

## ۵) محدودیت‌ها (برایِ شفافیت بیان می‌شود)

- **حجمِ نمونه.** ۱۱ عکس و ۹۳ دیسک یک مجموعه‌یِ ارزیابیِ اولیه و نسبتاً محدود است؛ اعدادِ بالا باید به‌عنوانِ یک توصیفِ اولیه‌یِ دقت خوانده شوند، نه یک اعتبارسنجیِ قطعی، تا زمانی‌که رویِ مجموعه‌ای بزرگ‌تر و متنوع‌تر از تصاویر سنجیده شود.
- **خطایِ قطرِ هاله متوسط است (MAE=۵.۵۰mm)** و عمدتاً در الگوهایِ شکستِ مشخص و شناسایی‌شده متمرکز است — به‌ویژه نشتِ سیگنال از یک هاله‌یِ واقعاً بزرگ به برآوردِ پس‌زمینه‌یِ دیسکِ همسایه، و یک اثرِ باقی‌مانده‌یِ شیبِ نورِ ناهموار (vignetting) در برخی پلیت‌ها — که هردو در حالِ بررسیِ فعال‌اند و در `PROJECT_HISTORY.md` مستند شده‌اند.
- **آستانه‌یِ طبقه‌بندیِ حضور، Recall را به‌نفعِ Precision معامله می‌کند.** گیتِ آماریِ اطمینان که برایِ تصمیمِ «هاله هست یا نیست» استفاده می‌شود (حداقلِ نسبتِ کنتراست‌به‌نویز)، در آخرین بازبینی سخت‌تر شد تا یک دسته‌یِ بزرگ از FPها حذف شود؛ این کار specificity را محسوس بهبود داد (۱۴ True Negative، از صفر) اما هزینه‌اش ۶ False Negativeِ تازه بود — هاله‌هایِ واقعیِ کم‌کنتراستی که سیگنالشان از همان آستانه عبور نکرد. این مصالحه صریح بیان می‌شود و هنوز به‌طورِ کامل حل نشده.
- **دقتِ قطرِ دیسک بیانگرِ یک بررسیِ خودسازگاری است**، نه یک اندازه‌گیریِ کالیبره‌شده‌یِ کاملاً مستقل، به‌دلیلِ توضیحِ بخشِ ۲.

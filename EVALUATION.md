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

**Multi-branch halo segmentation.** Zone boundaries are produced by three independent branches that share one preprocessed substrate and are then combined:

- *Agar canvas* (shared input): the dish interior in grayscale with uneven illumination removed. The illumination field is estimated by normalized convolution over agar pixels only — antibiotic disks and the plastic rim are excluded first, and their field values interpolated from surrounding agar. Estimating the field over the whole image instead lets each bright disk inflate its own neighbourhood, and subtracting that contaminated field carves an artificial dark ring around every disk, manufacturing the very artefact the correction is meant to remove.
- *Radial branch*: ring-averaged intensity profile with a background-convergence criterion.
- *Region branch (Otsu)*: global threshold over the agar canvas, with the zone/lawn polarity decided from the far field rather than assumed.
- *Region branch (watershed)*: marker-controlled flooding along the canvas's real edges, each disk seeding its own label and the far field seeding the lawn label, so adjacent zones divide on the edge between them rather than on a geometric bisector. Regions are validated against the far-field lawn distribution by standardized effect size.

*Far-field reference*: agar pixels furthest from every disk cannot physically lie inside any zone, so their intensity distribution serves as the operational definition of bacterial lawn — supplying both the watershed background marker and the validation reference.

**Fusion rule.** The branches are complementary in a measured, asymmetric way (Table 6), so they are combined by division of labour rather than by averaging: whichever region branch fires supplies the measurement (watershed first, then Otsu), and the radial branch supplies both the presence decision and the measurement wherever no region branch fired. The rule contains no threshold fitted to the evaluation set.

**Disk-diameter accuracy.** Computed analogously over all correctly matched disks, against the fixed 6.0 mm reference. Note: the pipeline's own pixel→millimetre calibration is derived from the very same disks (the tightest cluster of disk diameters in pixels, within a 30% tolerance band, averaged and mapped onto the known 6.0 mm standard — see `_estimate_px_per_mm_from_disks` in the notebook). Disk-diameter accuracy is therefore a self-consistency check on that calibration step, not an independent validation of it; the low residual error mainly confirms the *disks within one plate are of consistent apparent size*, which the calibration step already assumes.

## 3. Results

**Table 1 — Petri dish detection.** 11/11 photographs (100%) had exactly one disk-bearing petri dish correctly localized.

**Table 2 — Disk detection** (93 ground-truth disks, 11 images)

| TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 91 | 1 | 2 | 0.989 | 0.978 | 0.984 |

With the shape-verification stage (module 11.5, arc continuity plus a label- and rotation-invariant disk template) the single false positive is removed at no cost in recall:

| TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 91 | 0 | 2 | 1.000 | 0.978 | 0.989 |

**Table 3 — Halo-presence confusion matrix** (n = 91 correctly matched disks)

| | Expert: halo present | Expert: no halo |
|---|---:|---:|
| **System: halo present** | 60 (TP) | 13 (FP) |
| **System: no halo** | 3 (FN) | 15 (TN) |

Derived: accuracy = 0.824, precision = 0.822, recall = 0.952.

**Table 4 — Halo-diameter accuracy** (n = 60, both expert and system report a halo)

| MAE | Bias (mean signed error) | SD | 95% limits of agreement |
|---:|---:|---:|---:|
| 3.85 mm | −0.61 mm | 5.98 mm | [−12.33, +11.11] mm |

**Table 5 — Disk-diameter accuracy** (n = 91, reference = 6.0 mm)

| MAE | Bias | SD |
|---:|---:|---:|
| 0.18 mm | −0.00 mm | 0.23 mm |

**Table 6 — Individual branch performance** (n = 91; the basis for the fusion rule)

| branch | TP | FN | FP | TN | Accuracy | MAE |
|---|---:|---:|---:|---:|---:|---:|
| radial | 57 | 6 | 14 | 14 | 0.780 | 5.49 mm |
| Otsu (90th percentile) | 26 | 37 | 1 | 27 | 0.582 | 3.66 mm |
| watershed | 11 | 52 | 2 | 26 | 0.407 | **1.08 mm** |
| statistical | 16 | 47 | 6 | 22 | 0.418 | 5.02 mm |

The asymmetry is the whole basis of the design: the radial branch has by far the best recall but the worst precision and the worst measurement accuracy, while the region branches almost never fire falsely (1 and 2 false positives against 28 no-zone disks) and are three to five times more accurate when they do.

**Table 7 — Effect of fusion** (n = 91)

| | Accuracy | MAE | 95% limits of agreement |
|---|---:|---:|---:|
| radial branch alone | 0.780 | 5.50 mm | [−16.12, +15.05] mm |
| fused | 0.813 | **3.85 mm** | **[−12.33, +11.11]** mm |
| fused + shape verification (module 11.5, deployed) | **0.824** | **3.85 mm** | **[−12.33, +11.11]** mm |

Fusion is better than or equal to the single-branch baseline on every axis: three real zones recovered (TP 57→60, FN 6→3), false positives unchanged at 14, and a 30% reduction in measurement error. The shape-verification stage added afterwards removes one further false halo (14→13) as a side effect of no longer admitting a spurious disk, lifting accuracy to 0.824.

## 3.1 Guarding against overfitting

With 11 images and 91 disks, and a design in which several branches and statistics were selected by comparing them on that same set, apparent performance can be inflated even though no model is trained. Two safeguards were applied.

First, every candidate was measured on the whole set rather than tuned on individual images. This is what rejected the texture/speckle channel: local-variance distributions for zone and lawn separated by only +0.29 to +0.64 sd on one plate and were statistically indistinguishable on another (−0.07 to +0.06 sd, 99–100% overlap) across three window sizes, so the channel was removed rather than tuned into apparent usefulness.

Second, leave-one-image-out cross-validation was applied to every decision that involved selecting a value. This caught a real instance of overfitting. An earlier version of the fusion rule required weak radial detections (below 4σ contrast) to be confirmed by a region branch; in-sample this appeared to cut false positives from 14 to 9 and raise accuracy to 0.835. Under leave-one-out — threshold chosen on ten images, evaluated on the eleventh — accuracy fell to 0.758, *worse than applying no rule at all* (0.780), because the gate cost five true positives to remove three false ones. The gate was removed.

What survives cross-validation is the part with no fitted parameter: MAE improves from 5.49 mm to 3.85 mm in-sample and to ≈3.75 mm under leave-one-out, because each region branch decides from its own image's far-field reference rather than from any dataset-wide constant. The choice of the 90th-percentile statistic for the Otsu branch was also stable, selected in all 11 folds. Since the deployed configuration contains no threshold fitted on the evaluation set, the reported figures require no optimism correction.

The statistical branch is implemented but deliberately **not** part of the deployed fusion chain. Adding it moves MAE from 3.85 to 3.73 mm while leaving every other metric unchanged — a 0.12 mm difference roughly six times smaller than the standard error of the MAE at this sample size, and therefore indistinguishable from noise. It is markedly accurate on the five disks only it detects (0.51 mm against 4.23 mm for the radial branch on the two of them that are real zones), but two disks cannot support a design decision. It remains available for re-evaluation on a larger annotated set.

## 3.2 Reference repeatability, and what the error figures can mean

Every figure above is measured against a single expert reading, so the reading's own repeatability bounds what any of them can establish. The reference laboratory states that its experts' zone measurements vary by **±2 mm in all respects**. Because "±2 mm" is ambiguous (a full range, a 95% interval, or one standard deviation), all three readings are carried through rather than one being assumed. The simulation uses a fixed seed and is therefore deterministic and reproducible (`ground_truth/eucast/expert_repeatability.py`).

**The reference noise does not account for the system's error.** If the system error and the reference noise are independent their variances add, so subtracting the reference contribution from the observed spread gives:

| Interpretation of ±2 mm | Reference SD | Corrected system MAE |
|---|---:|---:|
| Uniform on [−2, +2] | 1.15 | 3.78 |
| Normal, 95% interval = ±2 | 1.02 | 3.80 |
| Normal, SD = 2 | 2.00 | 3.63 |

Even on the most generous reading, MAE falls only from 3.85 to 3.63 mm. The residual error is predominantly the system's own.

**But the attainable ceiling is not 100% agreement.** Simulating a second expert (the same reading plus ±2 mm noise, rounded to the nearest millimetre) and repeating the categorical analysis of §3 gives a human-versus-human benchmark:

| Comparison | MAE (mm) | CA | VME | ME |
|---|---:|---:|---:|---:|
| Expert vs expert (uniform) | 1.37 | 95.0% | 2.08% | 2.00% |
| Expert vs expert (95% = ±2) | 1.16 | 95.8% | 1.70% | 1.77% |
| Expert vs expert (SD = 2) | 2.26 | 92.1% | 3.29% | 3.39% |
| **System vs expert** | **3.85** | **82.5%** | **6.86%** | **9.24%** |

Two consequences follow, and both matter for how the §3 numbers should be read.

First, the realistic ceiling for categorical agreement against this kind of reference is roughly **92–96%, not 100%**. The system's shortfall is therefore about 10 percentage points rather than 18.

Second, at ±2 mm **two human experts do not themselves satisfy the conventional VME limit of 1.5%** (they reach 1.70–3.29%). Those ISO/CLSI limits were defined for comparison against a reference MIC obtained by broth dilution, not against a caliper reading, so applying them unmodified to an expert-read reference overstates what any method could demonstrate.

**Consequence for the accuracy target.** The target derived in §3 from the EUCAST limits (MAE ≈ 1.0 mm) lies *below* human repeatability under all three interpretations. It is not unattainable — an automated reader can be more repeatable than a human — but it is **not demonstrable against this reference**: an accuracy finer than the noise of the measuring instrument cannot be shown with that instrument. Demonstrating it would require a stronger reference, such as the mean of several independent readings of the same plate (whose noise falls as 1/√n) or an MIC reference from broth dilution.

**Where the system currently sits relative to the ±2 mm band.** 48% of measurements (29 of 60) fall within ±2 mm of the expert — that is, they are as close as a second expert would be expected to be — and the median error, 2.08 mm, sits essentially on the edge of that band. The mean is pulled up by a minority of large errors, concentrated in one branch:

| Branch | n | Within ±2 mm | Median error (mm) |
|---|---:|---:|---:|
| Watershed | 11 | 91% | 1.17 |
| Otsu | 16 | 50% | 1.94 |
| Radial | 33 | 33% | 3.11 |

**The false positives are not explained by reading ambiguity either.** Were the 14 halo false positives a product of the ±2 mm uncertainty about whether a zone exists at all, their reported diameters would cluster near the 6 mm disk. They do not: the median reported diameter is 10.5 mm and only 2 of the 14 fall below 8 mm. These are genuine detection errors.

## 4. Independent reproduction

These results were independently reproduced, unchanged, on two further unrelated machines (a Windows workstation, fresh standalone Python 3.11.9 installation, no shared environment or configuration with the development machine) by re-running the identical, unmodified `evaluate_pipeline.py` script end to end. The reproduction returned numerically identical summary statistics (same TP/FP/FN counts, same MAE/bias/SD to two decimal places), and a third reproduction on a Linux container returned the same figures again, which is consistent with the pipeline's stated design goal of full determinism — the same input image always yields the same output, with no randomized or non-reproducible step in the detection or measurement code path.

## 5. Limitations (stated for transparency)

- **The reference itself carries ±2 mm of noise.** The reporting laboratory states its experts' readings vary by ±2 mm in all respects. Variance subtraction shows this accounts for very little of the system's error (§3.2), but it sets a hard ceiling on what these results can establish: agreement finer than about 1.2 mm, or categorical agreement above roughly 92–96%, cannot be demonstrated against a single expert reading. Conclusions about accuracy below that level would require a stronger reference.
- **Sample size.** 11 images and 93 disks is a modest first evaluation set; the numbers above should be read as an initial accuracy characterization, not a definitive validation, pending assessment on a larger and more varied image set.
- **Halo-diameter error remains moderate (MAE 3.85 mm).** An empirical failure catalogue (`ground_truth/diagnostics/halo_failure_catalogue.csv`) attributes the residual error mainly to cases where a one-dimensional radial profile cannot distinguish "reached the lawn" from "not there yet" or "now inside a neighbour's zone". Adding region-based branches addressed a substantial part of this; it is not fully solved.
- **Disk detection reaches F1 0.989** (precision 1.000) after the shape-verification stage; the two remaining false negatives have distinct causes and are documented in `ROADMAP.md`.
- **Fourteen halo false positives remain unresolved.** The only mechanism tested for reducing them — a contrast gate on weak radial detections — was rejected by cross-validation (§3.1) because it did not generalise. No validated replacement has been found.
- **Selection pressure on a small set.** Several branches and statistics were compared on the same 11 images. Cross-validation was applied to every value-selecting decision, and the deployed configuration contains no fitted threshold, but the number of design choices evaluated against 91 disks remains high relative to the evidence. Additional expert-annotated images would do more for confidence than any further methodological refinement.
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

**سگمنت‌کردنِ چندشاخه‌ایِ هاله.** مرزِ هاله توسطِ سه شاخه‌ی مستقل تولید می‌شود که یک زیرساختِ پیش‌پردازش‌شده‌ی مشترک دارند و سپس ادغام می‌شوند:

- *بومِ آگار* (ورودیِ مشترک): داخلِ ظرف به‌صورتِ خاکستری با نورِ ناهموارِ حذف‌شده. میدانِ روشنایی با کانولوشنِ نرمال‌شده و **فقط از رویِ پیکسل‌هایِ آگار** تخمین زده می‌شود — دیسک‌ها و دیواره‌ی پلاستیکیِ ظرف پیش از تخمین کنار گذاشته و مقدارشان از آگارِ اطراف درون‌یابی می‌شود. تخمینِ میدان از رویِ کلِ تصویر باعث می‌شود هر دیسکِ روشن همسایگیِ خودش را بالا ببرد و تفریقِ آن تخمینِ آلوده دورِ هر دیسک یک حلقه‌ی تاریکِ ساختگی بسازد — یعنی دقیقاً همان آرتیفکتی که قرار است حذف شود.
- *شاخه‌ی شعاعی*: پروفایلِ میانگینِ حلقه‌ای با معیارِ همگرایی به پس‌زمینه.
- *شاخه‌ی ناحیه‌ای (Otsu)*: آستانه‌ی سراسری رویِ بومِ آگار، با تعیینِ پلاریته‌ی هاله/لَون از رویِ میدانِ دور به‌جایِ فرض‌کردنِ آن.
- *شاخه‌ی ناحیه‌ای (Watershed)*: سیلابِ کنترل‌شده با نشانگر رویِ لبه‌هایِ واقعیِ بوم — هر دیسک برچسبِ خودش را می‌کارد و میدانِ دور برچسبِ لَون را — پس دو هاله‌ی مجاور رویِ لبه‌ی واقعیِ بینشان تقسیم می‌شوند نه رویِ یک نیمسازِ هندسیِ کور. هر ناحیه با اندازه‌ی اثرِ استانداردشده در برابرِ توزیعِ لَونِ میدانِ دور اعتبارسنجی می‌شود.

*مرجعِ میدانِ دور*: پیکسل‌هایِ آگاری که از هر دیسکی دورترند به‌لحاظِ فیزیکی نمی‌توانند داخلِ هیچ هاله‌ای باشند، پس توزیعِ شدتشان تعریفِ عملیاتیِ «لَونِ باکتری» است — و هم نشانگرِ پس‌زمینه‌ی Watershed را می‌دهد هم مرجعِ اعتبارسنجی را.

**قاعده‌ی ادغام.** شاخه‌ها به‌شکلی *اندازه‌گیری‌شده و نامتقارن* مکملِ هم‌اند (جدولِ ۶)، پس به‌جایِ میانگین‌گیری با تقسیمِ کار ترکیب می‌شوند: هر شاخه‌ی ناحیه‌ای که فعال شود اندازه‌گیری را تامین می‌کند (اول Watershed، سپس Otsu)، و شاخه‌ی شعاعی هم تصمیمِ حضور را می‌دهد و هم اندازه‌گیری را هرجا هیچ شاخه‌ی ناحیه‌ای فعال نشده باشد. این قاعده هیچ آستانه‌ای که رویِ داده‌ی ارزیابی برازش شده باشد ندارد.

**دقتِ عددیِ قطرِ دیسک.** به همین شکل رویِ همه‌یِ دیسک‌هایِ به‌درستی‌تطبیق‌یافته، در برابرِ مرجعِ ثابتِ ۶.۰ میلی‌متر محاسبه می‌شود. نکته: کالیبراسیونِ px→mm خودِ پایپلاین از رویِ همین دیسک‌ها به دست می‌آید (فشرده‌ترین خوشه‌یِ قطرِ دیسک‌ها به پیکسل، در بازه‌یِ تحملِ ۳۰٪، میانگین‌گیری‌شده و معادلِ استانداردِ ۶.۰mm گذاشته می‌شود — تابعِ `_estimate_px_per_mm_from_disks` در نوت‌بوک). پس دقتِ قطرِ دیسک بیشتر یک بررسیِ خودسازگاریِ همین گامِ کالیبراسیون است، نه یک اعتبارسنجیِ کاملاً مستقل از آن؛ خطایِ باقی‌مانده‌یِ کم عمدتاً تاییدِ این است که «دیسک‌هایِ داخلِ یک پلیتِ واحد اندازه‌یِ ظاهریِ یکسانی دارند» — که خودِ گامِ کالیبراسیون هم دقیقاً همین فرض را دارد.

## ۳) نتایج

**جدولِ ۱ — تشخیصِ پتری.** ۱۱ از ۱۱ عکس (۱۰۰٪) دقیقاً یک پتریِ دارایِ دیسک به‌درستی مکان‌یابی شد.

**جدولِ ۲ — تشخیصِ دیسک** (۹۳ دیسکِ مرجع، ۱۱ عکس)

| TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| ۹۱ | ۱ | ۲ | ۰.۹۸۹ | ۰.۹۷۸ | ۰.۹۸۴ |

با افزودنِ مرحله‌ی اعتبارسنجیِ شکل (ماژولِ ۱۱.۵ — پیوستگیِ کمانی به‌علاوه‌ی الگویِ دیسکِ مستقل از برچسب و چرخش)، تنها مثبتِ کاذب بدونِ هیچ هزینه‌ای در recall حذف می‌شود:

| TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| ۹۱ | ۰ | ۲ | ۱.۰۰۰ | ۰.۹۷۸ | ۰.۹۸۹ |

**جدولِ ۳ — ماتریسِ درهم‌ریختگیِ حضورِ هاله** (n = ۹۱ دیسکِ به‌درستی‌تطبیق‌یافته)

| | کارشناس: هاله دارد | کارشناس: بدونِ هاله |
|---|---:|---:|
| **سیستم: هاله دارد** | ۶۰ (TP) | ۱۳ (FP) |
| **سیستم: بدونِ هاله** | ۳ (FN) | ۱۵ (TN) |

مقادیرِ مشتق‌شده: دقتِ کلی (accuracy) = ۰.۸۲۴، Precision = ۰.۸۲۲، Recall = ۰.۹۵۲.

**جدولِ ۴ — دقتِ عددیِ قطرِ هاله** (n = ۶۰، هم کارشناس هم سیستم هاله گزارش داده‌اند)

| MAE | Bias (خطایِ میانگینِ علامت‌دار) | SD | حدودِ توافقِ ۹۵٪ |
|---:|---:|---:|---:|
| ۳.۸۵ mm | −۰.۶۱ mm | ۵.۹۸ mm | [−۱۲.۳۳, +۱۱.۱۱] mm |

**جدولِ ۵ — دقتِ عددیِ قطرِ دیسک** (n = ۹۱، مرجع = ۶.۰mm)

| MAE | Bias | SD |
|---:|---:|---:|
| ۰.۱۸ mm | −۰.۰۰ mm | ۰.۲۳ mm |

**جدولِ ۶ — عملکردِ تک‌تکِ شاخه‌ها** (n = ۹۱؛ مبنایِ قاعده‌ی ادغام)

| شاخه | TP | FN | FP | TN | دقت | MAE |
|---|---:|---:|---:|---:|---:|---:|
| شعاعی | ۵۷ | ۶ | ۱۴ | ۱۴ | ۰.۷۸۰ | ۵.۴۹ mm |
| Otsu (صدکِ ۹۰) | ۲۶ | ۳۷ | ۱ | ۲۷ | ۰.۵۸۲ | ۳.۶۶ mm |
| Watershed | ۱۱ | ۵۲ | ۲ | ۲۶ | ۰.۴۰۷ | **۱.۰۸ mm** |
| آماری | ۱۶ | ۴۷ | ۶ | ۲۲ | ۰.۴۱۸ | ۵.۰۲ mm |

همین نامتقارنی، کلِ مبنایِ طراحی است: شاخه‌ی شعاعی به‌مراتب بهترین پوشش را دارد ولی بدترین precision و بدترین دقتِ اندازه‌گیری؛ شاخه‌هایِ ناحیه‌ای تقریباً هرگز کاذب نمی‌زنند (۱ و ۲ کاذب در برابرِ ۲۸ دیسکِ بدونِ هاله) و وقتی فعال می‌شوند سه تا پنج برابر دقیق‌ترند.

**جدولِ ۷ — اثرِ ادغام** (n = ۹۱)

| | دقت | MAE | حدودِ توافقِ ۹۵٪ |
|---|---:|---:|---:|
| فقط شاخه‌ی شعاعی | ۰.۷۸۰ | ۵.۵۰ mm | [−۱۶.۱۲, +۱۵.۰۵] mm |
| ادغام‌شده | ۰.۸۱۳ | **۳.۸۵ mm** | **[−۱۲.۳۳, +۱۱.۱۱]** mm |
| ادغام + اعتبارسنجیِ شکل (ماژولِ ۱۱.۵، مستقر) | **۰.۸۲۴** | **۳.۸۵ mm** | **[−۱۲.۳۳, +۱۱.۱۱]** mm |

ادغام رویِ هر محور بهتر یا مساویِ خطِ پایه است: سه هاله‌ی واقعی بازیابی شد (TP از ۵۷ به ۶۰، FN از ۶ به ۳)، مثبتِ کاذب بدونِ تغییر در ۱۴ ماند، و خطایِ اندازه‌گیری ۳۰٪ کاهش یافت. معیارهایِ تشخیصِ دیسک و قطرِ دیسک دست‌نخورده ماندند.

## ۳.۱) محافظت در برابرِ Overfitting

با ۱۱ عکس و ۹۱ دیسک، و طراحی‌ای که در آن چند شاخه و چند آماره با مقایسه رویِ همان مجموعه انتخاب شده‌اند، عملکردِ ظاهری می‌تواند بیش از واقع باشد — حتی وقتی هیچ مدلی آموزش نمی‌بیند. دو محافظ به‌کار رفت.

**اول**، هر گزینه رویِ کلِ مجموعه سنجیده شد نه رویِ تک‌عکس تنظیم. همین باعثِ ردِ کانالِ بافت/اسپکل شد: جدایی توزیعِ واریانسِ محلیِ هاله و لَون رویِ یک پلیت فقط ۰.۲۹+ تا ۰.۶۴+ انحرافِ‌معیار و رویِ پلیتِ دیگر عملاً صفر بود (۰.۰۷− تا ۰.۰۶+، هم‌پوشانیِ ۹۹–۱۰۰٪) در سه اندازه‌ی پنجره؛ کانال حذف شد نه این‌که تا مفیدشدنِ ظاهری تنظیم شود.

**دوم**، اعتبارسنجیِ leave-one-image-out رویِ هر تصمیمی که شاملِ انتخابِ یک مقدار بود اجرا شد — و یک موردِ واقعیِ overfitting را گرفت. نسخه‌ی قبلیِ قاعده‌ی ادغام لازم می‌دانست تشخیصِ شعاعیِ ضعیف (زیرِ ۴σ) با یک شاخه‌ی ناحیه‌ای تایید شود؛ درون‌نمونه‌ای به‌نظر می‌رسید مثبتِ کاذب را از ۱۴ به ۹ می‌رساند و دقت را به ۰.۸۳۵. زیرِ leave-one-out — آستانه رویِ ده عکس انتخاب، رویِ یازدهمی سنجیده — دقت به ۰.۷۵۸ افتاد، یعنی *بدتر از اعمال‌نکردنِ هیچ قاعده‌ای* (۰.۷۸۰)، چون آن گیت برایِ حذفِ سه کاذب، پنج تشخیصِ درست را قربانی می‌کرد. گیت حذف شد.

آن‌چه از اعتبارسنجی جانِ سالم به‌در برد همان بخشی است که هیچ پارامترِ برازش‌شده ندارد: MAE از ۵.۴۹ به ۳.۸۵ میلی‌متر درون‌نمونه‌ای و به ≈۳.۷۵ زیرِ leave-one-out بهبود می‌یابد، چون هر شاخه‌ی ناحیه‌ای از رویِ مرجعِ میدانِ دورِ *خودِ همان تصویر* تصمیم می‌گیرد نه از رویِ ثابتی در سطحِ دیتاست. انتخابِ آماره‌ی صدکِ ۹۰ برایِ شاخه‌ی Otsu هم پایدار بود و در هر ۱۱ تا انتخاب شد. چون پیکربندیِ مستقرشده هیچ آستانه‌ی برازش‌شده رویِ داده‌ی ارزیابی ندارد، اعدادِ گزارش‌شده نیازی به تصحیحِ خوش‌بینی ندارند.

شاخه‌ی آماری پیاده‌سازی شده ولی **عمداً** جزوِ زنجیره‌ی مستقرشده نیست. افزودنش MAE را از ۳.۸۵ به ۳.۷۳ می‌برد و بقیه‌ی معیارها را دست‌نخورده می‌گذارد — اختلافی ۰.۱۲ میلی‌متری که حدودِ شش برابر کوچک‌تر از خطایِ استانداردِ MAE در این حجمِ نمونه است و بنابراین از نویز قابلِ‌تفکیک نیست. رویِ پنج دیسکی که تنها همین شاخه تشخیصشان می‌دهد به‌طورِ محسوسی دقیق است (۰.۵۱ در برابرِ ۴.۲۳ میلی‌متر شاخه‌ی شعاعی، رویِ آن دو موردی که واقعاً هاله‌اند)، ولی دو دیسک نمی‌تواند پایه‌ی یک تصمیمِ طراحی باشد. برایِ ارزیابیِ مجدد رویِ مجموعه‌ی بزرگ‌ترِ حاشیه‌نویسی‌شده باقی می‌ماند.

## ۳.۲) تکرارپذیریِ مرجع، و این‌که اعدادِ خطا چه می‌توانند بگویند

همه‌ی اعدادِ بالا در برابرِ خوانشِ **یک** کارشناس سنجیده شده‌اند، پس تکرارپذیریِ خودِ آن خوانش تعیین می‌کند این اعداد اصلاً چه چیزی را می‌توانند اثبات کنند. آزمایشگاهِ مرجع اعلام کرده اندازه‌گیریِ کارشناسانِ آن‌ها **«از همه لحاظ ±۲ میلی‌متر تغییر دارد»**. چون «±۲» مبهم است (بازه‌ی کامل؟ فاصله‌ی اطمینانِ ۹۵٪؟ یک انحرافِ معیار؟)، هر سه تفسیر حساب می‌شود به‌جایِ فرضِ یکی. شبیه‌سازی با بذرِ ثابت اجرا می‌شود، پس قطعی و بازتولیدپذیر است (`ground_truth/eucast/expert_repeatability.py`).

**نویزِ مرجع، خطایِ سیستم را توجیه نمی‌کند.** اگر خطایِ سیستم و نویزِ مرجع مستقل باشند واریانس‌هایشان جمع می‌شود، پس با کسرِ سهمِ مرجع از پراکندگیِ مشاهده‌شده:

| تفسیرِ ±۲ میلی‌متر | انحرافِ معیارِ مرجع | MAE سیستم پس از تصحیح |
|---|---:|---:|
| یکنواخت روی [−۲, +۲] | ۱.۱۵ | ۳.۷۸ |
| نرمال، بازه‌ی ۹۵٪ = ±۲ | ۱.۰۲ | ۳.۸۰ |
| نرمال، انحرافِ معیار = ۲ | ۲.۰۰ | ۳.۶۳ |

حتی با سخاوتمندانه‌ترین تفسیر، MAE از ۳.۸۵ فقط به ۳.۶۳ میلی‌متر می‌رسد. خطایِ باقی‌مانده عمدتاً مالِ خودِ سیستم است.

**ولی سقفِ قابلِ‌دستیابی، توافقِ ۱۰۰٪ نیست.** با شبیه‌سازیِ کارشناسِ دوم (همان خوانش به‌علاوه‌ی نویزِ ±۲ میلی‌متر، گردشده به نزدیک‌ترین میلی‌متر) و تکرارِ تحلیلِ دسته‌ایِ بخشِ ۳:

| مقایسه | MAE (mm) | CA | VME | ME |
|---|---:|---:|---:|---:|
| کارشناس در برابرِ کارشناس (یکنواخت) | ۱.۳۷ | ۹۵.۰٪ | ۲.۰۸٪ | ۲.۰۰٪ |
| کارشناس در برابرِ کارشناس (۹۵٪ = ±۲) | ۱.۱۶ | ۹۵.۸٪ | ۱.۷۰٪ | ۱.۷۷٪ |
| کارشناس در برابرِ کارشناس (sd = ۲) | ۲.۲۶ | ۹۲.۱٪ | ۳.۲۹٪ | ۳.۳۹٪ |
| **سیستم در برابرِ کارشناس** | **۳.۸۵** | **۸۲.۵٪** | **۶.۸۶٪** | **۹.۲۴٪** |

دو پیامد دارد و هر دو در نحوه‌ی خواندنِ اعدادِ بخشِ ۳ اثر می‌گذارند.

نخست، سقفِ واقع‌بینانه‌ی توافقِ دسته‌ای در برابرِ چنین مرجعی حدودِ **۹۲ تا ۹۶ درصد است، نه ۱۰۰ درصد**. پس فاصله‌ی سیستم تا سقف حدودِ ۱۰ واحدِ درصد است، نه ۱۸.

دوم، با ±۲ میلی‌متر **خودِ دو کارشناسِ انسانی هم سقفِ مرسومِ VME=۱.۵٪ را برآورده نمی‌کنند** (به ۱.۷۰ تا ۳.۲۹ درصد می‌رسند). آن سقف‌هایِ ISO/CLSI برایِ مقایسه با MICِ مرجع از روشِ رقتِ براث تعریف شده‌اند، نه با خوانشِ کولیس؛ پس اعمالِ بی‌تغییرِ آن‌ها به یک مرجعِ کارشناس‌خوانده، بیش از آن‌چه هر روشی بتواند نشان دهد سخت‌گیری می‌کند.

**پیامد برایِ هدفِ دقت.** هدفی که در بخشِ ۳ از سقف‌هایِ EUCAST به‌دست آمد (MAE ≈ ۱.۰ میلی‌متر) در هر سه تفسیر **زیرِ** تکرارپذیریِ انسانی قرار می‌گیرد. این هدف دست‌نیافتنی نیست — یک خوانشگرِ خودکار می‌تواند از انسان باثبات‌تر باشد — ولی **در برابرِ این مرجع قابلِ نمایش نیست**: دقتی ظریف‌تر از نویزِ خودِ ابزارِ سنجش را نمی‌شود با همان ابزار نشان داد. اثباتش به مرجعِ قوی‌تری نیاز دارد: میانگینِ چند خوانشِ مستقل از همان پلیت (که نویزش با ۱/√n کم می‌شود)، یا مرجعِ MIC از رقتِ براث.

**وضعیتِ فعلیِ سیستم نسبت به باندِ ±۲ میلی‌متر.** ۴۸ درصدِ اندازه‌گیری‌ها (۲۹ از ۶۰) در فاصله‌ی ±۲ میلی‌متریِ کارشناس قرار می‌گیرند — یعنی به همان اندازه نزدیک‌اند که از یک کارشناسِ دوم انتظار می‌رود — و میانه‌ی خطا با ۲.۰۸ میلی‌متر عملاً رویِ لبه‌ی همان باند است. میانگین را اقلیتی از خطاهایِ بزرگ بالا می‌کشد که در یک شاخه متمرکزند:

| شاخه | n | داخلِ ±۲ میلی‌متر | میانه‌ی خطا (mm) |
|---|---:|---:|---:|
| Watershed | ۱۱ | ۹۱٪ | ۱.۱۷ |
| Otsu | ۱۶ | ۵۰٪ | ۱.۹۴ |
| شعاعی | ۳۳ | ۳۳٪ | ۳.۱۱ |

**مثبت‌هایِ کاذب هم با ابهامِ خوانش توضیح داده نمی‌شوند.** اگر ۱۴ مثبتِ کاذبِ حضورِ هاله محصولِ همان عدمِ‌قطعیتِ ±۲ میلی‌متری درباره‌ی وجود یا نبودِ هاله بودند، قطرِ گزارش‌شده‌شان باید حولِ دیسکِ ۶ میلی‌متری جمع می‌شد. چنین نیست: میانه‌ی قطرِ گزارش‌شده ۱۰.۵ میلی‌متر است و فقط ۲ مورد از ۱۴ زیرِ ۸ میلی‌متر قرار می‌گیرند. این‌ها خطاهایِ واقعیِ تشخیص‌اند.

## ۴) بازتولیدپذیریِ مستقل

این نتایج به‌طورِ مستقل و بدونِ هیچ تغییری رویِ یک سیستمِ دوم و کاملاً بی‌ربط (یک لپ‌تاپِ ویندوزی، نصبِ تازه و مستقلِ Python 3.11.9، بدونِ هیچ محیط یا تنظیمِ مشترک با ماشینِ توسعه) با اجرایِ مجددِ همان اسکریپتِ بدون‌تغییرِ `evaluate_pipeline.py` بازتولید شد. این بازتولید دقیقاً همان آمارِ خلاصه را داد (همان تعدادِ TP/FP/FN، همان MAE/Bias/SD تا دو رقمِ اعشار) — سازگار با هدفِ طراحیِ مستندِ پایپلاین مبنی‌بر قطعیتِ کامل (determinism): یک تصویرِ ورودیِ یکسان همیشه خروجیِ یکسان می‌دهد، بدونِ هیچ گامِ تصادفی یا غیرِقابل‌بازتولید در مسیرِ کدِ تشخیص یا اندازه‌گیری.

## ۵) محدودیت‌ها (برایِ شفافیت بیان می‌شود)

- **خودِ مرجع ±۲ میلی‌متر نویز دارد.** آزمایشگاهِ گزارش‌دهنده اعلام کرده خوانشِ کارشناسانش از همه لحاظ ±۲ میلی‌متر تغییر دارد. کسرِ واریانس نشان می‌دهد این سهمِ بسیار کمی از خطایِ سیستم را توضیح می‌دهد (بخشِ ۳.۲)، ولی سقفِ سختی رویِ آن‌چه این نتایج می‌توانند اثبات کنند می‌گذارد: توافقی ظریف‌تر از حدودِ ۱.۲ میلی‌متر، یا توافقِ دسته‌ایِ بالاتر از حدودِ ۹۲ تا ۹۶ درصد، در برابرِ خوانشِ یک کارشناسِ تنها قابلِ نمایش نیست. هر نتیجه‌گیری درباره‌ی دقتِ زیرِ آن سطح، به مرجعِ قوی‌تری نیاز دارد.
- **حجمِ نمونه.** ۱۱ عکس و ۹۳ دیسک یک مجموعه‌یِ ارزیابیِ اولیه و نسبتاً محدود است؛ اعدادِ بالا باید به‌عنوانِ یک توصیفِ اولیه‌یِ دقت خوانده شوند، نه یک اعتبارسنجیِ قطعی، تا زمانی‌که رویِ مجموعه‌ای بزرگ‌تر و متنوع‌تر از تصاویر سنجیده شود.
- **خطایِ قطرِ هاله همچنان متوسط است (MAE=۳.۸۵mm).** کاتالوگِ تجربیِ حالت‌هایِ شکست (`ground_truth/diagnostics/halo_failure_catalogue.csv`) خطایِ باقی‌مانده را عمدتاً به مواردی نسبت می‌دهد که پروفایلِ یک‌بعدیِ شعاعی نمی‌تواند «به لَون رسیدم» را از «هنوز نرسیدم» یا «واردِ قلمروِ همسایه شدم» تشخیص دهد. افزودنِ شاخه‌هایِ ناحیه‌ای بخشِ قابلِ‌توجهی از این را برطرف کرد؛ کاملاً حل‌نشده باقی است.
- **تشخیصِ دیسک با مرحله‌ی اعتبارسنجیِ شکل به F1=۰.۹۸۹ می‌رسد** (Precision=۱.۰۰۰)؛ دو منفیِ کاذبِ باقی‌مانده علت‌هایِ متفاوتی دارند و در `ROADMAP.md` مستند شده‌اند.
- **۱۴ مثبتِ کاذبِ حضورِ هاله رفع‌نشده مانده.** تنها سازوکاری که برایِ کاهششان آزموده شد — یک گیتِ کنتراست رویِ تشخیص‌هایِ شعاعیِ ضعیف — در اعتبارسنجیِ متقاطع (بخشِ ۳.۱) رد شد چون تعمیم نمی‌یافت. جایگزینِ اعتبارسنجی‌شده‌ای پیدا نشده است.
- **فشارِ انتخاب رویِ یک مجموعه‌ی کوچک.** چند شاخه و چند آماره رویِ همان ۱۱ عکس مقایسه شدند. اعتبارسنجیِ متقاطع رویِ هر تصمیمِ مقدارگزین اعمال شد و پیکربندیِ مستقرشده هیچ آستانه‌ی برازش‌شده ندارد، ولی تعدادِ انتخاب‌هایِ طراحی که در برابرِ ۹۱ دیسک سنجیده شده‌اند نسبت به حجمِ شواهد بالاست. عکس‌هایِ حاشیه‌نویسی‌شده‌ی بیشتر، بیش از هر پالایشِ روش‌شناختیِ دیگری به اطمینان کمک می‌کند.
- **دقتِ قطرِ دیسک بیانگرِ یک بررسیِ خودسازگاری است**، نه یک اندازه‌گیریِ کالیبره‌شده‌یِ کاملاً مستقل، به‌دلیلِ توضیحِ بخشِ ۲.

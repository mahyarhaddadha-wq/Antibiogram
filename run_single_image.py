#!/usr/bin/env python3
"""
اجرای پایپلاینِ آنتی‌بایوگرام رویِ **یک** عکس، و ذخیره‌ی همه‌ی خروجی‌ها در یک پوشه.

نمونه‌ی اجرا (ویندوز، از داخلِ پوشه‌ی ریپازیتوری):

    python run_single_image.py --image "D:\\antibiogram\\input\\plate1.jpg"

    python run_single_image.py --image "D:\\antibiogram\\input\\plate1.jpg" ^
                               --output-dir "D:\\antibiogram\\output" ^
                               --kernel python3

اگر گونه‌ی باکتری و آنتی‌بیوتیکِ هر دیسک را می‌دانید، طبقه‌بندیِ بالینیِ EUCAST هم
انجام می‌شود (ماژولِ ۱۸):

    python run_single_image.py --image plate1.jpg ^
        --organism "Enterobacterales" ^
        --agents "1=Ciprofloxacin,3=Gentamicin,5=Meropenem"

بدونِ این دو، ماژولِ ۱۸ دسته اعلام نمی‌کند و فقط بازه‌ی دسته‌هایِ ممکن را نشان
می‌دهد -- که صادقانه‌تر از حدس زدنِ آنتی‌بیوتیک است.

خروجی‌ها در پوشه‌ی <output-dir>/<نامِ عکس>/ :
    01_fusion_disks.png     تشخیصِ نهاییِ دیسک‌ها (بعد از اعتبارسنجیِ شکلِ ماژولِ ۱۱.۵)
    02_halo_dish<N>.png     مرزِ نهاییِ هاله (ماژولِ ۱۶.۶)، یک فایل برایِ هر پتری
    03_bubbles_dish<N>.png  رخدادهایِ حباب داخلِ هاله (ماژولِ ۱۷)
    04_final_report.txt     گزارشِ متنی: قطرِ دیسک، قطرِ هاله، حباب، و دسته‌ی EUCAST
    05_eucast.txt           جزئیاتِ طبقه‌بندیِ بالینی (ماژولِ ۱۸)
    06_full_log.txt         کلِ خروجیِ متنیِ اجرا (برایِ عیب‌یابی)

پیش‌نیاز:
    pip install nbformat nbclient opencv-python numpy matplotlib
    و یک کرنلِ Jupyter (نامش را با `jupyter kernelspec list` ببینید).

اصلِ طراحی: خودِ نوت‌بوک تنها منبعِ حقیقتِ الگوریتم است. این اسکریپت آن را تغییر
نمی‌دهد؛ فقط یک سلولِ override برایِ مسیرِ عکس (و اختیاراً تنظیماتِ EUCAST) اضافه
می‌کند و کلِ نوت‌بوک را با کرنلِ تازه اجرا می‌کند.
"""
import argparse
import base64
import copy
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = SCRIPT_DIR / "disk_detection_pipeline_modular.ipynb"

# نشانه‌هایِ متنیِ سلول‌ها -- به شماره‌ی سلول وابسته نیستیم تا افزودنِ ماژولِ جدید
# به نوت‌بوک این اسکریپت را نشکند.
CFG_MARKER = "cfg = Phase2Config()"
CFG_EXT_MARKER = "cfg.halo_fusion_otsu_percentile"
FUSION_MARKER = "Fusion Result —"
HALO_MARKER = "# ── ماژول ۱۶.۶ (جدید)"
BUBBLE_MARKER = "Halo Bubble Events —"
EUCAST_MARKER = "# ── ماژول ۱۸ (جدید)"
REPORT_MARKER = "گزارش نهایی آنتی‌بایوگرام"


def find_cell(nb, marker):
    for i, c in enumerate(nb["cells"]):
        if marker in "".join(c["source"]):
            return i
    return None


def parse_agents(text):
    """'1=Ciprofloxacin,3=Gentamicin' -> {1: 'Ciprofloxacin', 3: 'Gentamicin'}"""
    out = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"قالبِ --agents اشتباه است: {part!r} (باید مثلِ 1=نامِ عامل باشد)")
        k, v = part.split("=", 1)
        out[int(k.strip())] = v.strip()
    return out


def build_notebook(base_nb, image_path, organism, agents):
    nb = copy.deepcopy(base_nb)
    cells = list(nb["cells"])

    lines = [f'cfg.image_path = r"{image_path}"']
    if organism:
        lines.append(f'cfg.eucast_organism = {organism!r}')
    if agents:
        lines.append(f'cfg.eucast_disk_agents = {agents!r}')

    # تنظیماتِ EUCAST در سلولِ Config Extension تعریف می‌شوند، پس override باید
    # *بعد* از آن بیاید وگرنه بازنویسی می‌شود. اگر آن سلول نبود، بعد از ساختِ cfg.
    idx = find_cell(nb, CFG_EXT_MARKER)
    if idx is None:
        idx = find_cell(nb, CFG_MARKER)
    if idx is None:
        raise SystemExit("سلولِ پیکربندی در نوت‌بوک پیدا نشد -- نوت‌بوک درست نیست؟")
    cells.insert(idx + 1, nbformat.v4.new_code_cell("\n".join(lines)))
    nb["cells"] = cells
    return nb


def pngs_of(nb, marker):
    i = find_cell(nb, marker)
    if i is None:
        return []
    return [base64.b64decode(o["data"]["image/png"])
            for o in nb["cells"][i].get("outputs", [])
            if "image/png" in o.get("data", {})]


def text_of(nb, marker):
    i = find_cell(nb, marker)
    if i is None:
        return ""
    return "".join(o.get("text", "") for o in nb["cells"][i].get("outputs", [])
                   if o.get("output_type") == "stream")


def all_text(nb):
    return "".join(o.get("text", "") for c in nb["cells"]
                   for o in c.get("outputs", []) if o.get("output_type") == "stream")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True, help="مسیرِ عکسِ ورودی")
    ap.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "output",
                    help="پوشه‌ی خروجی (پیش‌فرض: ./output کنارِ همین اسکریپت)")
    ap.add_argument("--kernel", default="python3",
                    help="نامِ کرنلِ Jupyter (با `jupyter kernelspec list` ببینید)")
    ap.add_argument("--organism", default=None,
                    help='گونه/گروهِ ارگانیسم برایِ EUCAST، مثلاً "Enterobacterales"')
    ap.add_argument("--agents", default=None,
                    help='آنتی‌بیوتیکِ هر دیسک، مثلاً "1=Ciprofloxacin,3=Gentamicin"')
    ap.add_argument("--timeout", type=int, default=1800,
                    help="سقفِ زمانِ اجرا برایِ هر سلول (ثانیه)")
    args = ap.parse_args()

    if not NOTEBOOK_PATH.exists():
        raise SystemExit(f"نوت‌بوک پیدا نشد: {NOTEBOOK_PATH}\n"
                         "این اسکریپت باید کنارِ فایلِ .ipynb باشد.")
    if not args.image.is_file():
        raise SystemExit(f"عکس پیدا نشد: {args.image}")

    agents = parse_agents(args.agents)
    print(f"[{args.image.name}] در حالِ پردازش ... (کرنل: {args.kernel})")

    base = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nb = build_notebook(base, args.image.resolve(), args.organism, agents)
    try:
        NotebookClient(nb, kernel_name=args.kernel, timeout=args.timeout).execute()
    except Exception as e:
        print(f"\nاجرا با خطا متوقف شد: {type(e).__name__}: {e}", file=sys.stderr)
        print("اگر خطا مربوط به کرنل است، نامِ درست را با "
              "`jupyter kernelspec list` ببینید و با --kernel بدهید.", file=sys.stderr)
        raise SystemExit(1)

    out = args.output_dir / args.image.stem
    out.mkdir(parents=True, exist_ok=True)

    fusion = pngs_of(nb, FUSION_MARKER)
    if fusion:
        (out / "01_fusion_disks.png").write_bytes(fusion[0])
    for i, png in enumerate(pngs_of(nb, HALO_MARKER), start=1):
        (out / f"02_halo_dish{i}.png").write_bytes(png)
    for i, png in enumerate(pngs_of(nb, BUBBLE_MARKER), start=1):
        (out / f"03_bubbles_dish{i}.png").write_bytes(png)

    report = text_of(nb, REPORT_MARKER)
    (out / "04_final_report.txt").write_text(report, encoding="utf-8")
    (out / "05_eucast.txt").write_text(text_of(nb, EUCAST_MARKER), encoding="utf-8")
    (out / "06_full_log.txt").write_text(all_text(nb), encoding="utf-8")

    print(f"[{args.image.name}] تمام شد -> {out}\n")
    if report.strip():
        print(report)


if __name__ == "__main__":
    main()

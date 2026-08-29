#!/usr/bin/env python3
"""
اجرای دسته‌ای (batch) پایپلاین آنتی‌بایوگرام روی همه‌ی عکس‌های یک پوشه.

اجرا (روی ویندوز، از داخل پوشه‌ی ریپازیتوری):
    python batch_process_antibiogram.py
    python batch_process_antibiogram.py --input-dir "D:\antibiogram_engine-version 2\input" ^
                                        --output-dir "D:\antibiogram_engine-version 2\output"

هر عکس با اجرای کامل نوت‌بوک disk_detection_pipeline_modular.ipynb پردازش می‌شود --
خودِ نوت‌بوک هیچ تغییری نمی‌کند (طبق اصل «نوت‌بوک تنها منبع حقیقت الگوریتم است»)، فقط
cfg.image_path برای هر عکس override می‌شود و کل نوت‌بوک با یک کرنل تازه اجرا می‌شود
(دقیقاً همان‌طور که این نوت‌بوک همیشه به‌صورت تعاملی برای یک عکس اجرا می‌شود).

برای هر عکس، در پوشه‌ی خروجی/<نام عکس>/ چهار خروجی ذخیره می‌شود:
  01_fusion_disks.png     -- نتیجه‌ی نهایی تشخیص دیسک‌ها (ماژول ۱۵، همه‌ی پتری‌های عکس)
  02_halo_dish<N>.png     -- نتیجه‌ی نهایی و اصلاح‌شده‌ی هاله (ماژول ۱۶.۶، بعد از رشد
                             نامتقارن ۱۶.۵ و رفع رخدادهای زاویه‌ای)، یک فایل به‌ازای هر پتری
  03_bubbles_dish<N>.png  -- رخدادهای حباب/توده‌ی داخل هاله (ماژول ۱۷)، یک فایل به‌ازای هر پتری
  04_final_report.txt     -- گزارش نهایی متنی: قطر دیسک/هاله/حباب/دسته‌ی EUCAST (ماژول ۱۹)
  05_eucast.txt           -- جزئیات طبقه‌بندی بالینی S/I/R طبق EUCAST v16.0 (ماژول ۱۸)
و یک summary.txt در ریشه‌ی پوشه‌ی خروجی با وضعیت کلی همه‌ی عکس‌ها.

پیش‌نیاز: nbformat و nbclient (`pip install nbformat nbclient`) و یک کرنل Jupyter نصب‌شده
(پیش‌فرض این اسکریپت نام کرنل "python3" را فرض می‌کند -- با `jupyter kernelspec list` نام
واقعی کرنل خودتان را ببینید و در صورت تفاوت با `--kernel <نام>` مشخص کنید).
"""
import argparse
import base64
import copy
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = SCRIPT_DIR / "disk_detection_pipeline_modular.ipynb"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_OUTPUT_DIR = Path(r"D:\antibiogram_engine-version 2\output")

FUSION_MARKER = "Fusion Result —"
HALO_MARKER = "# ── ماژول ۱۶.۶ (جدید)"  # آخرین ماژولِ اصلاح‌کننده‌ی مرز هاله (بعد از ۱۶.۵) -- تصویرِ خروجی‌اش دایره‌های نهایی/تصحیح‌شده را نشان می‌دهد
BUBBLE_MARKER = "Halo Bubble Events —"
EUCAST_MARKER = "# ── ماژول ۱۸ (جدید)"   # طبقه‌بندیِ بالینیِ S/I/R
REPORT_MARKER = "گزارش نهایی آنتی‌بایوگرام"
CFG_INIT_MARKER = "cfg = Phase2Config()"


def _default_input_dir() -> Path:
    """
    پوشه‌ی ورودی پیش‌فرض مستقیماً از روی مقدار پیش‌فرض فعلی cfg.image_path در خودِ
    نوت‌بوک استخراج می‌شود -- تا این اسکریپت همیشه با آدرسی که همین الان در کد تنظیم
    شده هماهنگ بماند، بدون این‌که مسیر به‌صورت دستی و جدا اینجا هم تکرار/کپی شود.
    """
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        m = re.search(r'image_path:\s*str\s*=\s*r?"([^"]+)"', cell["source"])
        if m:
            return Path(m.group(1)).parent
    raise RuntimeError("مقدار پیش‌فرض cfg.image_path در نوت‌بوک پیدا نشد.")


def _find_cell_index(nb, marker: str) -> int:
    for i, cell in enumerate(nb["cells"]):
        if marker in cell["source"]:
            return i
    raise RuntimeError(f"سلولی حاوی این نشانه در نوت‌بوک پیدا نشد: {marker!r}")


def _build_notebook_for_image(base_nb, image_path: Path, organism: str = None):
    """کپی مستقل از نوت‌بوک پایه با یک سلول override برای cfg.image_path، بلافاصله بعد
    از سلولی که cfg ساخته می‌شود -- بدون هیچ تغییر دیگری در نوت‌بوک اصلی."""
    nb = copy.deepcopy(base_nb)
    lines = [f'cfg.image_path = r"{image_path}"']
    if organism:
        # تنظیماتِ EUCAST در سلولِ Config Extension تعریف می‌شوند، پس override باید
        # بعد از آن بیاید وگرنه بازنویسی می‌شود.
        lines.append(f'cfg.eucast_organism = {organism!r}')
        idx_cfg = _find_cell_index(nb, "cfg.halo_fusion_otsu_percentile")
    else:
        idx_cfg = _find_cell_index(nb, CFG_INIT_MARKER)
    override_cell = nbformat.v4.new_code_cell("\n".join(lines))
    cells = list(nb["cells"])
    cells.insert(idx_cfg + 1, override_cell)
    nb["cells"] = cells
    return nb


def _extract_png_outputs(cell) -> list:
    pngs = []
    for out in cell.get("outputs", []):
        data = out.get("data", {})
        if "image/png" in data:
            pngs.append(base64.b64decode(data["image/png"]))
    return pngs


def _extract_stream_text(cell) -> str:
    parts = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            parts.append(out.get("text", ""))
    return "".join(parts)


def process_one_image(base_nb, image_path: Path, output_dir: Path, kernel_name: str,
                      organism: str = None) -> bool:
    print(f"[{image_path.name}] در حال پردازش ...")
    try:
        nb = _build_notebook_for_image(base_nb, image_path, organism)
        client = NotebookClient(nb, kernel_name=kernel_name, timeout=1800)
        client.execute()

        img_out_dir = output_dir / image_path.stem
        img_out_dir.mkdir(parents=True, exist_ok=True)

        fusion_pngs = _extract_png_outputs(nb["cells"][_find_cell_index(nb, FUSION_MARKER)])
        if fusion_pngs:
            (img_out_dir / "01_fusion_disks.png").write_bytes(fusion_pngs[0])

        halo_pngs = _extract_png_outputs(nb["cells"][_find_cell_index(nb, HALO_MARKER)])
        for i, png in enumerate(halo_pngs, start=1):
            (img_out_dir / f"02_halo_dish{i}.png").write_bytes(png)

        bubble_pngs = _extract_png_outputs(nb["cells"][_find_cell_index(nb, BUBBLE_MARKER)])
        for i, png in enumerate(bubble_pngs, start=1):
            (img_out_dir / f"03_bubbles_dish{i}.png").write_bytes(png)

        report_text = _extract_stream_text(nb["cells"][_find_cell_index(nb, REPORT_MARKER)])
        (img_out_dir / "04_final_report.txt").write_text(report_text, encoding="utf-8")

        # ماژولِ ۱۸ -- طبقه‌بندیِ بالینی. بدونِ اعلامِ ارگانیسم/آنتی‌بیوتیک، به‌جایِ
        # دسته، بازه‌ی دسته‌هایِ ممکن را چاپ می‌کند؛ در هر دو حالت ذخیره می‌شود.
        eucast_text = _extract_stream_text(nb["cells"][_find_cell_index(nb, EUCAST_MARKER)])
        (img_out_dir / "05_eucast.txt").write_text(eucast_text, encoding="utf-8")

        print(f"[{image_path.name}] تمام شد -> {img_out_dir}")
        return True
    except Exception as e:
        print(f"[{image_path.name}] خطا: {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=None,
                       help="پوشه‌ی عکس‌های ورودی (پیش‌فرض: پوشه‌ی cfg.image_path فعلی در نوت‌بوک)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                       help=f"پوشه‌ی خروجی (پیش‌فرض: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--kernel", type=str, default="python3",
                       help="نام کرنل Jupyter برای اجرای نوت‌بوک (پیش‌فرض: python3؛ با "
                            "`jupyter kernelspec list` نام واقعی خودتان را چک کنید)")
    parser.add_argument("--organism", type=str, default=None,
                       help='گونه/گروهِ ارگانیسم برایِ طبقه‌بندیِ EUCAST -- برایِ کلِ پوشه '
                            'یکسان فرض می‌شود، مثلاً "Enterobacterales". اگر ندهید، '
                            'ماژولِ ۱۸ فقط بازه‌ی دسته‌هایِ ممکن را گزارش می‌کند.')
    args = parser.parse_args()

    input_dir = args.input_dir or _default_input_dir()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"خطا: پوشه‌ی ورودی پیدا نشد: {input_dir}")
        sys.exit(1)

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"هیچ عکسی در {input_dir} پیدا نشد.")
        sys.exit(1)

    print(f"{len(images)} عکس در «{input_dir}» پیدا شد. خروجی در: «{output_dir}»\n")

    base_nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

    ok_count = 0
    summary_lines = []
    for image_path in images:
        success = process_one_image(base_nb, image_path, output_dir, args.kernel)
        ok_count += int(success)
        summary_lines.append(f"{image_path.name}: {'OK' if success else 'FAILED'}")

    (output_dir / "summary.txt").write_text(
        f"{ok_count}/{len(images)} عکس با موفقیت پردازش شد.\n\n" + "\n".join(summary_lines),
        encoding="utf-8")
    print(f"\nتمام شد: {ok_count}/{len(images)} عکس با موفقیت پردازش شد. "
         f"خلاصه در {output_dir / 'summary.txt'}")


if __name__ == "__main__":
    main()

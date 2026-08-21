"""
اسکریپت ساخت/به‌روزرسانیِ کش تشخیص پتری برای همه‌ی عکس‌های ریپازیتوری (یا لیست دلخواه).

اجرا: python3 build_petri_cache.py [--images img1.jpg img2.jpg ...] [--kernel NAME]
بدون --images: تمام فایل‌های *.jpg/*.jpeg موجود در همین پوشه (به‌جز موارد داخل خودِ
petri_cache/) پردازش می‌شوند.

هر عکس با نوت‌بوکِ واقعی (سلول‌های ۰ تا ۱۲: تنظیمات، توابع کمکی، بارگذاری عکس، ماژول ۴،
ماژول ۴.۱) از طریق nbformat+nbclient اجرا می‌شود -- یعنی کش همیشه دقیقاً همان چیزی است که
خودِ نوت‌بوک تولید می‌کند، بدون هیچ منطقِ کپی‌شده/جداگانه.
"""
import argparse
import glob
import os
import shutil

import nbformat
from nbclient import NotebookClient

import petri_cache_utils as pcu

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_NOTEBOOK = os.path.join(HERE, "disk_detection_pipeline_modular.ipynb")


def find_cell_idx(nb, marker):
    for i, c in enumerate(nb["cells"]):
        if marker in c["source"]:
            return i
    raise RuntimeError("marker not found: " + marker)


def build_for_image(image_path: str, kernel_name: str, tmp_dir: str):
    nb = nbformat.read(SRC_NOTEBOOK, as_version=4)
    idx_module41_end = find_cell_idx(nb, "استخراج ماسک دقیق و فید لبه‌های هر پتری")

    base_cells = list(nb["cells"][: idx_module41_end + 1])
    set_path_src = f'cfg.image_path = r"{image_path}"\nprint("=== IMAGE:", {image_path!r}, "===")'

    # cfg باید قبل از بارگذاری عکس مقداردهی شده باشد -- پیدا کردن سلول ساخت cfg و درج
    # override مسیر عکس بلافاصله بعدش (همان الگوی همه‌ی اسکریپت‌های تست این نشست).
    idx_cfg = find_cell_idx(nb, "cfg = Phase2Config()")
    new_cells = list(base_cells[: idx_cfg + 1])
    new_cells.append(nbformat.v4.new_code_cell(set_path_src))
    new_cells += list(base_cells[idx_cfg + 1:])
    new_cells.append(nbformat.v4.new_code_cell(pcu.build_cache_cell_source()))

    nb["cells"] = new_cells
    test_path = os.path.join(tmp_dir, "build_cache_run.ipynb")
    nbformat.write(nb, test_path)

    nb2 = nbformat.read(test_path, as_version=4)
    client = NotebookClient(nb2, kernel_name=kernel_name, timeout=900)
    client.execute()

    for c in nb2["cells"]:
        if c["cell_type"] == "code" and "[petri_cache]" in c["source"]:
            for out in c.get("outputs", []):
                if out.get("output_type") == "stream":
                    print(out["text"].strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--kernel", default="antibiogram-test")
    args = ap.parse_args()

    if args.images:
        images = args.images
    else:
        images = sorted(
            glob.glob(os.path.join(HERE, "*.jpg")) + glob.glob(os.path.join(HERE, "*.jpeg"))
        )

    tmp_dir = os.path.join(HERE, ".petri_cache_build_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        ok, failed = 0, []
        for i, img in enumerate(images):
            print(f"--- [{i + 1}/{len(images)}] {os.path.basename(img)} ---")
            try:
                build_for_image(img, args.kernel, tmp_dir)
                ok += 1
            except Exception as e:
                print(f"FAILED: {type(e).__name__}: {str(e)[:500]}")
                failed.append(os.path.basename(img))
        print(f"\nDONE: {ok}/{len(images)} succeeded.")
        if failed:
            print("Failed images:", failed)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

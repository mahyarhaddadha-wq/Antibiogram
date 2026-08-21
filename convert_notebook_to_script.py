#!/usr/bin/env python3
"""
تبدیلِ disk_detection_pipeline_modular.ipynb به یک اسکریپتِ پایتونِ ساده با
نشانه‌گذاریِ سلولیِ استانداردِ percent (`# %%`، همان فرمتِ jupytext) --
disk_detection_pipeline_modular.py.

چرا: نوت‌بوک تنها منبعِ حقیقتِ الگوریتم است (طبقِ اصلِ همیشگیِ این پروژه)؛ این
اسکریپت هیچ منطقی را دستی کپی نمی‌کند، فقط سلول‌های خودِ نوت‌بوک را عیناً به بلاک‌های
`# %%` تبدیل می‌کند -- پس فایلِ خروجی همیشه دقیقاً با نوت‌بوک یکسان می‌ماند، و بعد از
هر تغییرِ نوت‌بوک کافی است همین اسکریپت دوباره اجرا شود:

    python convert_notebook_to_script.py

فایلِ خروجی (disk_detection_pipeline_modular.py) دو روشِ اجرا را پشتیبانی می‌کند:
  ۱) اجرای کلی/یکجا:      python disk_detection_pipeline_modular.py
  ۲) اجرای ماژول‌به‌ماژول: در VS Code / PyCharm / Spyder باز شود -- هرکدام از این
     ادیتورها `# %%` را به‌صورتِ بومی به‌عنوانِ مرزِ یک سلولِ قابلِ‌اجرای مستقل
     می‌شناسند (دکمه‌ی «Run Cell» بالای هر بلاک)، دقیقاً همان تجربه‌ی اجرای
     سلول‌به‌سلولِ نوت‌بوک، بدون نیاز به خودِ Jupyter.
"""
from pathlib import Path

import nbformat

SCRIPT_DIR = Path(__file__).resolve().parent
SRC = SCRIPT_DIR / "disk_detection_pipeline_modular.ipynb"
OUT = SCRIPT_DIR / "disk_detection_pipeline_modular.py"

HEADER = '''# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# این فایل نسخه‌ی هم‌ارزِ disk_detection_pipeline_modular.ipynb است، به‌صورتِ یک
# اسکریپتِ پایتونِ ساده با نشانه‌گذاریِ سلولیِ استانداردِ percent (`# %%`) -- هر `# %%`
# دقیقاً معادلِ یک سلولِ نوت‌بوک است. این فایل توسطِ convert_notebook_to_script.py از
# خودِ نوت‌بوک ساخته می‌شود (نه نوشته/نگه‌داری‌شده به‌صورتِ جداگانه)، پس بعد از هر
# تغییرِ نوت‌بوک باید آن اسکریپت را دوباره اجرا کرد تا این فایل به‌روز بماند -- هیچ
# منطقی اینجا دستی تکرار/کپی نشده.
#
# دو روشِ اجرا:
#   ۱) اجرای کلی/یکجا:      python disk_detection_pipeline_modular.py
#   ۲) اجرای ماژول‌به‌ماژول: این فایل را در VS Code / PyCharm / Spyder باز کنید --
#      هرکدام از این ادیتورها نشانه‌ی `# %%` را به‌صورتِ بومی به‌عنوانِ مرزِ یک سلولِ
#      قابلِ‌اجرای مستقل می‌شناسند (دکمه‌ی «Run Cell» بالای هر بلاک) -- دقیقاً همان
#      تجربه‌ی اجرای سلول‌به‌سلولِ نوت‌بوک، بدون نیاز به خودِ Jupyter.
#
# نکته: خطِ `%matplotlib inline` نوت‌بوک (که فقط در IPython/Jupyter معنا دارد) در
# این فایل کامنت شده -- خارج از Jupyter نیازی به آن نیست؛ `plt.show()` در تابعِ
# show() در اجرای مستقیم با پنجره‌ی تعاملیِ matplotlib کار می‌کند.

'''


def convert() -> int:
    nb = nbformat.read(SRC, as_version=4)
    lines = [HEADER]
    for cell in nb["cells"]:
        src = cell["source"]
        if cell["cell_type"] == "markdown":
            lines.append("# %% [markdown]\n")
            for line in src.split("\n"):
                lines.append(("# " + line).rstrip() + "\n")
            lines.append("\n")
        elif cell["cell_type"] == "code":
            lines.append("# %%\n")
            body_lines = []
            for line in src.split("\n"):
                if line.strip() == "%matplotlib inline":
                    body_lines.append(
                        "# %matplotlib inline  # فقط در Jupyter لازم است؛ خارج از آن بدون اثر/غیرلازم")
                else:
                    body_lines.append(line)
            lines.append("\n".join(body_lines).rstrip("\n") + "\n")
            lines.append("\n")

    OUT.write_text("".join(lines), encoding="utf-8")
    return len(nb["cells"])


if __name__ == "__main__":
    n = convert()
    print(f"نوشته شد: {OUT} (از {n} سلول نوت‌بوک)")

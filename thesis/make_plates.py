#!/usr/bin/env python3
"""Compose the chapter-four pipeline gallery into grouped plates.

Twenty-seven separate notebook outputs make the chapter read as a lab
report, so the intermediate stages are laid out as labelled panels on ten
plates instead. Panels are lettered in Latin (a), (b), (c) … because the
captions that reference them are Persian and a Latin letter inside Persian
text needs no bidi handling.

    python3 thesis/make_plates.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "figures", "pipeline", "gt_06")
OUT = os.path.join(HERE, "figures", "plates")
os.makedirs(OUT, exist_ok=True)

# plate name -> (columns, [source stems in reading order])
PLATES = {
    "plate_4_1_dish":      (3, ["01_input_image", "02_dish_detection", "03_dish_mask"]),
    "plate_4_2_disk_pre":  (3, ["04_tophat_a", "05_tophat_b", "06_threshold",
                                "07_closing", "08_opening", "09_distance_transform"]),
    "plate_4_3_disk_br":   (3, ["14_watershed_markers", "10_halo_gradient", "11_disk_edges",
                                "12_hough_candidates", "13_blob_watershed"]),
    "plate_4_5_radial":    (2, ["16_agar_canvas", "21_halo_base",
                                "22_halo_growth", "23_halo_angular_fix"]),
    "plate_4_6_branches":  (2, ["17_branch_otsu", "18_branch_watershed",
                                "19_branch_statistical", "20_branch_growth_model"]),
}
SINGLES = {
    "plate_4_4_disks_final": "15_disks_final",
    "plate_4_7_fusion":      "24_halo_fusion",
    "plate_4_8_bubbles":     "25_bubbles",
    "plate_4_9_eucast":      "26_eucast",
    "plate_4_10_report":     "27_final_report",
}

CELL = 900          # each panel is fitted into a CELL-wide box
PAD = 18
LABEL = 46          # strip under each panel carrying its letter
BG = (255, 255, 255)
INK = (30, 30, 30)


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def fit(im, box):
    """Scale to fit inside a square box, keeping the aspect ratio."""
    w, h = im.size
    s = min(box / w, box / h)
    return im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def plate(stems, cols, out):
    ims = [fit(Image.open(os.path.join(SRC, s + ".png")).convert("RGB"), CELL)
           for s in stems]
    rows = (len(ims) + cols - 1) // cols
    cw = max(i.width for i in ims)
    ch = max(i.height for i in ims)
    W = cols * cw + (cols + 1) * PAD
    H = rows * (ch + LABEL) + (rows + 1) * PAD
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    f = font(34)
    for k, im in enumerate(ims):
        r, c = divmod(k, cols)
        x = PAD + c * (cw + PAD) + (cw - im.width) // 2
        y = PAD + r * (ch + LABEL + PAD)
        sheet.paste(im, (x, y + (ch - im.height) // 2))
        tag = f"({chr(ord('a') + k)})"
        tw = d.textbbox((0, 0), tag, font=f)[2]
        d.text((PAD + c * (cw + PAD) + (cw - tw) // 2, y + ch + 6), tag, font=f, fill=INK)
    sheet.save(os.path.join(OUT, out + ".png"))
    return sheet.size, len(ims)


total = 0
for name, (cols, stems) in PLATES.items():
    size, n = plate(stems, cols, name)
    print(f"  {name:24} {n} panels  {size[0]}x{size[1]}")
    total += n
for name, stem in SINGLES.items():
    im = Image.open(os.path.join(SRC, stem + ".png")).convert("RGB")
    im.save(os.path.join(OUT, name + ".png"))
    print(f"  {name:24} 1 panel   {im.size[0]}x{im.size[1]}")
    total += 1
print(f"{len(PLATES) + len(SINGLES)} plates from {total} source images")

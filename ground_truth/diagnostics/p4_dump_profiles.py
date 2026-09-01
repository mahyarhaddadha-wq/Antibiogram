"""
گامِ ۱ از پ۴: استخراجِ پروفایلِ شعاعیِ کاملِ هر دیسک رویِ بومِ آگار.

تفاوتِ کلیدی با پروفایلِ ماژولِ ۱۶: این پروفایل تا سقفِ فیزیکیِ لبه‌ی ظرف می‌رود،
نه تا پنجره‌ی تطبیقی. برایِ برازشِ سیگموئید باید *هر دو مجانب* دیده شوند.

خروجی: یک JSON با یک رکورد به‌ازایِ هر دیسک.
"""
import copy
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path("/home/user/Antibiogram")
NB = REPO / "disk_detection_pipeline_modular.ipynb"
RAW = REPO / "ground_truth" / "raw_images"
GT_CSV = REPO / "ground_truth" / "ground_truth_expert_readings.csv"
OUT = Path("/tmp/radial_profiles.json")
CFG_MARKER = "cfg = Phase2Config()"
STOP_MARKER = "# ── ماژول ۱۵.۸ (جدید)"   # بعد از بوم + مرجعِ میدانِ دور

DUMP = r'''
import numpy as _np, json as _json

_out = []
for _d in dishes:
    _ox, _oy = _d["roi_offset_xy"]
    _canvas = _d["agar_canvas"]
    _agar = _d["agar_mask"] > 0
    _far = _d.get("far_field_ref")
    _ppm = _d.get("px_per_mm_est") or 0.0
    _mask = _d["processing_mask_roi"]
    _h, _w = _canvas.shape[:2]
    _dt = cv2.distanceTransform(_ensure_uint8_binary(_mask), cv2.DIST_L2, 3)
    _cands = _d["final_candidates"]

    for _i, _c in enumerate(_cands):
        _cx, _cy, _rr = float(_c["x"]), float(_c["y"]), float(_c["r"])
        _r_in = _rr * 1.05
        # تا سقفِ فیزیکیِ لبه‌ی ظرف -- نه تا پنجره‌ی تطبیقی
        _r_out = max(_r_in + 8.0, float(_dt[int(round(_cy)), int(round(_cx))]) - 2.0)

        _R = int(_np.ceil(_r_out)) + 2
        _x0, _y0 = max(0, int(_cx) - _R), max(0, int(_cy) - _R)
        _x1, _y1 = min(_w, int(_cx) + _R + 1), min(_h, int(_cy) + _R + 1)
        _patch = _canvas[_y0:_y1, _x0:_x1].astype(_np.float32)
        _pm = _agar[_y0:_y1, _x0:_x1]
        _yy, _xx = _np.ogrid[:_patch.shape[0], :_patch.shape[1]]
        _rad = _np.sqrt((_xx - (_cx - _x0)) ** 2 + (_yy - (_cy - _y0)) ** 2)

        # کنارگذاریِ جهت‌آگاهِ قلمروِ همسایه (نیم‌صفحه‌ی ووُرونوی)
        _safe = _np.ones(_patch.shape, dtype=bool)
        for _j, _o in enumerate(_cands):
            if _j == _i:
                continue
            _dx, _dy = float(_o["x"]) - _cx, float(_o["y"]) - _cy
            _d2 = _dx * _dx + _dy * _dy
            if _d2 < 1e-6:
                continue
            _proj = (_xx + _x0 - _cx) * _dx + (_yy + _y0 - _cy) * _dy
            _safe &= (_proj < 0.5 * _d2)

        _ring_w = max(2.0, 0.10 * _rr)
        _n = max(6, int(_np.ceil((_r_out - _r_in) / _ring_w)))
        _edges = _np.linspace(_r_in, _r_out, _n + 1)
        _centers = 0.5 * (_edges[:-1] + _edges[1:])
        _idx = _np.digitize(_rad, _edges) - 1
        _valid = (_idx >= 0) & (_idx < _n) & _pm & _safe
        if int(_np.count_nonzero(_valid)) < 32:
            continue
        _sums = _np.bincount(_idx[_valid], weights=_patch[_valid], minlength=_n)
        _cnts = _np.bincount(_idx[_valid], minlength=_n)
        _means = _sums / _np.maximum(_cnts, 1)

        _out.append({
            "disk_idx": _i + 1,
            "x": _cx + _ox, "y": _cy + _oy, "r_disk": _rr,
            "px_per_mm": _ppm,
            "ring_centers": [float(v) for v in _centers],
            "profile": [float(v) for v in _means],
            "counts": [int(v) for v in _cnts],
            "lawn_mean": float(_far["mean"]) if _far else None,
            "lawn_std": float(_far["std"]) if _far else None,
        })

print("PROFILES_JSON_START")
print(_json.dumps(_out))
print("PROFILES_JSON_END")
'''


def build(base, img):
    nb = copy.deepcopy(base)
    cells = list(nb["cells"])
    i = next(k for k, c in enumerate(cells) if CFG_MARKER in "".join(c["source"]))
    cells.insert(i + 1, nbformat.v4.new_code_cell(f'cfg.image_path = r"{img}"'))
    j = next(k for k, c in enumerate(cells) if STOP_MARKER in "".join(c["source"]))
    nb["cells"] = cells[:j] + [nbformat.v4.new_code_cell(DUMP)]
    return nb


def main():
    base = nbformat.read(NB, as_version=4)
    gt = defaultdict(list)
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            gt[row["image_file"]].append({
                "num": int(row["disk_number"]), "x": float(row["x_px"]),
                "y": float(row["y_px"]),
                "halo": float(row["halo_diameter_mm_expert"])
                if row["halo_diameter_mm_expert"].strip() else None})

    all_rows = []
    for img in sorted(RAW.glob("gt_*.jpg")):
        nb = build(base, img)
        NotebookClient(nb, kernel_name="antibiogram-test", timeout=1800).execute()
        text = "".join(o.get("text", "") for c in nb["cells"]
                       for o in c.get("outputs", []) if o.get("output_type") == "stream")
        m = re.search(r"PROFILES_JSON_START\n(.*?)\nPROFILES_JSON_END", text, re.S)
        if not m:
            print(f"[warn] {img.name}: no payload")
            continue
        recs = json.loads(m.group(1))

        gd = gt[img.name]
        best = min((math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                    for i, a in enumerate(gd) for b in gd[i + 1:]), default=200.0)
        radius = 0.5 * best
        for r in recs:
            r["image"] = img.name
            mt, dm = None, 1e18
            for g in gd:
                dd = math.hypot(r["x"] - g["x"], r["y"] - g["y"])
                if dd < dm:
                    dm, mt = dd, g
            r["gt_halo"] = mt["halo"] if (mt and dm <= radius) else None
            r["gt_num"] = mt["num"] if (mt and dm <= radius) else None
        all_rows += recs
        print(f"[done] {img.name}: {len(recs)} profiles")

    OUT.write_text(json.dumps(all_rows))
    print(f"\nwrote {len(all_rows)} profiles -> {OUT}")


if __name__ == "__main__":
    main()

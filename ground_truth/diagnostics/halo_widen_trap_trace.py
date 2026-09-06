"""
ریشه‌یابیِ ۵ بدترینِ کم‌برآوردِ قطرِ هاله (بخشِ ۱۲.۱۹ تاریخچه).

فرضیه: آیا گسترشِ تطبیقیِ پنجره‌ی جست‌وجو در _halo_radial_profile (ماژول ۱۶) واقعاً
فعال می‌شود، یا همگراییِ «قابل‌اتکا» (|contrast_sigma| >= ۳σ) همیشه در همان اولین
پنجره‌ی کوچک (widen_iter=0) به‌اشتباه ارضا می‌شود؟

روش: یک کپیِ رونوشت‌بردار (trace) از حلقه‌ی داخلیِ _halo_radial_profile، فراخوانی‌شده
دقیقاً با همان آرگومان‌هایی که segment_dish_halos به آن می‌دهد (bg_noise مشترکِ پتری،
other_centers)، برایِ همان ۵ دیسکی که بدترین کم‌برآوردِ قطر را در evaluation_results.csv
دارند: gt_08#5 (-21.1mm)، gt_06#1 (-15.9mm)، gt_01#9 (-14.9mm)، gt_05#3 (-13.6mm)،
gt_01#5 (-13.1mm).

نتیجه: در هر ۵ مورد -- بدونِ استثنا -- حلقه هرگز فراتر از widen_iter=0 نمی‌رود.
پنجره‌ی اولیه (halo_r_max_scale=4.0 × r_disk) به‌طورِ سیستماتیک از شعاعِ واقعیِ
کارشناس (که ۶.۳ تا ۱۱.۰ برابرِ r_disk است) کوچک‌تر است، ولی چون noise (نویزِ مشترکِ
پتری، معمولاً ۲-۳ واحدِ روشنایی) در برابرِ یک روندِ ملایمِ غیرزیستیِ داخلِ همان پنجره‌ی
کوچک بسیار کوچک است، contrast_sigma به‌آسانی از ۳σ عبور می‌کند و گسترشِ تطبیقی -- که
دقیقاً برایِ همین حالت طراحی شده بود -- هرگز فرصتِ اجرا پیدا نمی‌کند.

یک تصحیحِ آزمایشی (نیازِ تاییدِ همگرایی در دو پنجره‌ی متوالی، نه فقط یکی) امتحان و رد
شد: روی همین ۵ مورد فقط ۱ مورد (gt_01#9: از ۱۴.۹- به ۵.۵- میلی‌متر) واقعاً بهبود
یافت، ۲ مورد بدونِ تغییر ماند (چون در آخرین گسترشِ مجاز همان تلهٔ قبلی دوباره رخ داد و
هیچ‌گاه دو بار متوالی تایید نشد)، و ۲ موردِ دیگر به‌جایِ یک عددِ غلط، هیچ عددی
گزارش نکردند (FN). مهم‌تر، رویِ کلِ ۹۳ دیسکِ ۱۱ عکس، تعدادِ TPِ هاله از ۶۰ به ۴۲ افت
کرد -- یعنی ۱۸ هاله‌ی واقعیِ دیگر که قبلاً درست تشخیص داده می‌شدند، اکنون هرگز به
همگراییِ قابل‌اتکا نمی‌رسند و به "no_reliable_signal" سقوط می‌کنند. تصحیح رد و
کدِ ماژول ۱۶ به حالتِ قبل بازگردانده شد (این اسکریپت فقط برایِ مستندسازیِ خودِ
یافته‌ی ریشه‌ای نگه داشته شده، نه برایِ اعمالِ آن تصحیحِ خاص).

خروجی این اسکریپت صرفاً چاپِ رونوشتِ حلقه برایِ هر ۵ دیسک است (بدونِ تغییر در نوت‌بوک).
"""
import copy
import csv
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path("/home/user/Antibiogram")
NB = REPO / "disk_detection_pipeline_modular.ipynb"
GT_CSV = REPO / "ground_truth" / "ground_truth_expert_readings.csv"
CFG_MARKER = "cfg = Phase2Config()"
STOP_MARKER = "# ── ماژول ۱۶.۵ (بازطراحی دوم)"

TARGETS = [("gt_08.jpg", 5), ("gt_06.jpg", 1), ("gt_01.jpg", 9),
           ("gt_05.jpg", 3), ("gt_01.jpg", 5)]

TRACE_FN = r'''
import numpy as _np

def _trace_halo_profile(gray_img, mask_u8, x, y, r_disk, cfg, bg_noise=None, other_centers=None):
    """رونوشتِ حلقه‌ی داخلیِ _halo_radial_profile با چاپِ هر گامِ گسترشِ تطبیقی."""
    dt_border = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 3)
    max_allowed = float(dt_border[y, x]) - 2.0
    r_in = cfg.halo_r_start_scale * float(r_disk)
    n_rings = int(cfg.halo_num_rings)
    tail = max(2, int(round(cfg.halo_bg_tail_frac * n_rings)))
    near_n = max(2, n_rings // 8)
    h, w = gray_img.shape[:2]
    scale = float(cfg.halo_r_max_scale)
    max_widenings = int(cfg.halo_r_search_max_widenings)
    for widen_iter in range(max_widenings + 1):
        r_out = min(scale * float(r_disk), max_allowed)
        if r_out < r_in * 1.15:
            print(f"  widen={widen_iter}: too_close_to_border r_out={r_out:.1f}")
            return
        R = int(_np.ceil(r_out)) + 2
        x0, y0 = max(0, x - R), max(0, y - R)
        x1, y1 = min(w, x + R + 1), min(h, y + R + 1)
        patch = gray_img[y0:y1, x0:x1].astype(_np.float32)
        pmask = mask_u8[y0:y1, x0:x1] > 0
        yy, xx = _np.ogrid[:patch.shape[0], :patch.shape[1]]
        rad = _np.sqrt((xx - (x - x0)) ** 2 + (yy - (y - y0)) ** 2)
        neighbor_safe = _np.ones(patch.shape, dtype=bool)
        if other_centers:
            abs_x = xx + x0
            abs_y = yy + y0
            for ox, oy in other_centers:
                dx, dy = float(ox) - float(x), float(oy) - float(y)
                d2 = dx * dx + dy * dy
                if d2 < 1e-6:
                    continue
                proj = (abs_x - x) * dx + (abs_y - y) * dy
                neighbor_safe &= (proj < 0.5 * d2)
        edges = _np.linspace(r_in, r_out, n_rings + 1)
        ring_centers = 0.5 * (edges[:-1] + edges[1:])
        idx = _np.digitize(rad, edges) - 1
        valid = (idx >= 0) & (idx < n_rings) & pmask & neighbor_safe
        sums = _np.bincount(idx[valid], weights=patch[valid], minlength=n_rings)
        sumsq = _np.bincount(idx[valid], weights=patch[valid] ** 2, minlength=n_rings)
        cnts = _np.bincount(idx[valid], minlength=n_rings)
        profile = sums / _np.maximum(cnts, 1)
        ring_var = sumsq / _np.maximum(cnts, 1) - profile ** 2
        ring_std = _np.sqrt(_np.maximum(ring_var, 0.0))
        reliable = cnts >= 5
        pixel_noise = float(_np.median(ring_std[reliable])) if _np.any(reliable) else float(_np.median(ring_std))
        pixel_noise = max(pixel_noise, 1e-3)
        if bg_noise is not None and bg_noise > 0:
            pixel_noise = float(bg_noise)
        good = cnts > 0
        trustworthy_n = n_rings
        k = n_rings - 1
        while k >= 0 and not good[k]:
            trustworthy_n = k
            k -= 1
        if trustworthy_n < n_rings:
            interior_good = good[:trustworthy_n]
            if int(_np.count_nonzero(interior_good)) < 3:
                print(f"  widen={widen_iter}: insufficient_ring_coverage")
                return
            ii = _np.arange(trustworthy_n)
            profile = _np.interp(ii, ii[interior_good], profile[:trustworthy_n][interior_good])
            ring_centers = ring_centers[:trustworthy_n]
        elif _np.any(~good):
            ii = _np.arange(n_rings)
            profile = _np.interp(ii, ii[good], profile[good])
        n_rings_eff = len(profile)
        tail_eff = max(1, min(tail, n_rings_eff // 3))
        near_n_eff = max(1, min(near_n, n_rings_eff // 3))
        background = float(_np.median(profile[-tail_eff:]))
        inner_val = float(_np.median(profile[:near_n_eff]))
        noise = pixel_noise
        contrast_sigma = (inner_val - background) / noise
        band = cfg.halo_background_convergence_sigma * noise
        lo, hi = background - band, background + band
        in_band = (profile >= lo) & (profile <= hi)
        r_halo_radial = r_in
        crossed = False
        kfound = None
        for k in range(n_rings_eff):
            if bool(_np.all(in_band[k:])):
                r_halo_radial = float(ring_centers[k])
                crossed = True
                kfound = k
                break
        reliable_convergence = crossed and (abs(contrast_sigma) >= cfg.halo_extension_require_min_contrast_sigma)
        ppm_dbg = _ppm_for_trace
        r_mm = r_halo_radial / ppm_dbg if kfound is not None else -1.0
        print(f"  widen={widen_iter}: r_out={r_out/ppm_dbg:.2f}mm(r)  noise={noise:.3f}  "
              f"inner_val={inner_val:.2f}  background={background:.2f}  "
              f"contrast_sigma={contrast_sigma:+.2f}  crossed={crossed}(r={r_mm:.2f}mm)  "
              f"reliable={reliable_convergence}")
        if reliable_convergence or r_out >= max_allowed - 1.0 or widen_iter == max_widenings:
            print(f"  --> STOP at widen={widen_iter}  final_r={r_halo_radial/ppm_dbg:.2f}mm (radius)  "
                  f"max_allowed={max_allowed/ppm_dbg:.2f}mm")
            return
        scale *= float(cfg.halo_r_max_scale_growth)
'''


def build(base, img, tx, ty):
    nb = copy.deepcopy(base)
    cells = list(nb["cells"])
    i = next(k for k, c in enumerate(cells) if CFG_MARKER in "".join(c["source"]))
    cells.insert(i + 1, nbformat.v4.new_code_cell(f'cfg.image_path = r"{img}"'))
    j = next(k for k, c in enumerate(cells) if STOP_MARKER in "".join(c["source"]))
    debug = TRACE_FN + f'''
for _d in dishes:
    _ox, _oy = _d["roi_offset_xy"]
    _ppm_for_trace = _d.get("px_per_mm_est") or 1.0
    _best, _bd = None, 1e18
    for _c in _d["final_candidates"]:
        _dd = ((_c["x"] + _ox) - {tx}) ** 2 + ((_c["y"] + _oy) - {ty}) ** 2
        if _dd < _bd:
            _bd, _best = _dd, _c
    if _best is not None and _bd ** 0.5 < 30:
        _disks_in = [{{"x": cc["x"], "y": cc["y"], "r": cc["r"]}} for cc in _d["final_candidates"]]
        _petri_r = 0.5 * _d["diameter_px"]
        _bg_noise = _compute_dish_background_noise(_d["roi_gray_masked"], _d["processing_mask_roi"],
                                                    _disks_in, _petri_r, cfg)
        print(f"bg_noise={{_bg_noise}}")
        _other = [(dd["x"], dd["y"]) for dd in _disks_in
                 if not (dd["x"] == _best["x"] and dd["y"] == _best["y"])]
        _trace_halo_profile(_d["roi_gray_masked"], _d["processing_mask_roi"], _best["x"], _best["y"],
                            _best["r"], cfg, bg_noise=_bg_noise, other_centers=_other)
'''
    nb["cells"] = cells[:j] + [nbformat.v4.new_code_cell(debug)]
    return nb


def main():
    base = nbformat.read(NB, as_version=4)
    gt = {}
    with open(GT_CSV) as f:
        for row in csv.DictReader(f):
            gt.setdefault(row["image_file"], []).append(row)

    for img_name, dn in TARGETS:
        rows = gt[img_name]
        target = next(r for r in rows if r["disk_number"] == str(dn))
        tx, ty = float(target["x_px"]), float(target["y_px"])
        img_path = f"ground_truth/raw_images/{img_name}"
        nb = build(base, img_path, tx, ty)
        NotebookClient(nb, kernel_name="antibiogram-test", timeout=1800).execute()
        text = "".join(o.get("text", "") for c in nb["cells"]
                       for o in c.get("outputs", []) if o.get("output_type") == "stream")
        print(f"\n=== {img_name} disk#{dn} (GT={target['halo_diameter_mm_expert']}mm diam) ===")
        for line in text.splitlines():
            if line.startswith("bg_noise=") or line.startswith("  widen") or line.startswith("  -->"):
                print(line)


if __name__ == "__main__":
    main()

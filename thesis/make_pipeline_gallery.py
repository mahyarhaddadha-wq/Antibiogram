#!/usr/bin/env python3
"""Re-extract the per-cell visual output of the current pipeline notebook.

The images land in thesis/figures/pipeline/<image>/ and are the figures of
chapter four. The repository's own pipeline_module_gallery/ is left alone.

Runs the unmodified notebook once per image (only the input path is
overridden, exactly as evaluate_pipeline.py does) and writes every image
output of every cell to disk, numbered in execution order.
"""
import base64
import copy
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path("/home/user/Antibiogram")
NB = ROOT / "disk_detection_pipeline_modular.ipynb"
OUT = ROOT / "thesis" / "figures" / "pipeline"
IMAGES = sys.argv[1:] or ["gt_02", "gt_06", "gt_10"]


# Fixed ASCII names, in notebook execution order, so the chapter-four
# figure manifest in build_docx.js can address them directly.
NAMES = {
    1: "input_image", 2: "dish_detection", 3: "dish_mask", 4: "tophat_a",
    5: "tophat_b", 6: "threshold", 7: "closing", 8: "opening",
    9: "distance_transform", 10: "halo_gradient", 11: "disk_edges",
    12: "hough_candidates", 13: "blob_watershed", 14: "watershed_markers",
    15: "disks_final", 16: "agar_canvas", 17: "branch_otsu",
    18: "branch_watershed", 19: "branch_statistical", 20: "branch_growth_model",
    21: "halo_base", 22: "halo_growth", 23: "halo_angular_fix",
    24: "halo_fusion", 25: "bubbles", 26: "eucast", 27: "final_report",
}


for name in IMAGES:
    img = ROOT / "ground_truth" / "raw_images" / f"{name}.jpg"
    if not img.exists():
        print(f"skip {name}: no such image", flush=True)
        continue
    nb = nbformat.read(NB, as_version=4)
    override = nbformat.v4.new_code_cell(f'cfg.image_path = r"{img}"')
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code" and "cfg.halo_fusion_otsu_percentile" in c.source:
            nb.cells.insert(i + 1, override)
            break
    else:
        raise SystemExit("config-extension cell not found")

    print(f"running {name} ...", flush=True)
    NotebookClient(nb, timeout=1800, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()

    dst = OUT / name
    dst.mkdir(parents=True, exist_ok=True)
    for old in dst.glob("*.png"):
        old.unlink()

    n = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for o in cell.get("outputs", []):
            if "data" not in o or "image/png" not in o["data"]:
                continue
            n += 1
            label = NAMES.get(n, f"cell_{n:02d}")
            (dst / f"{n:02d}_{label}.png").write_bytes(
                base64.b64decode(o["data"]["image/png"]))
    if n != len(NAMES):
        print(f"  WARNING: {n} images, expected {len(NAMES)} -- the notebook "
              "cell order changed; update NAMES and the figure manifest.",
              flush=True)
    print(f"  {name}: {n} images -> {dst}", flush=True)
print("done", flush=True)

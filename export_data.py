"""
export_data.py
Processes all PDFs and writes pre-computed data to docs/data/ for the web dashboard.

Usage:
    python export_data.py                     # all PDFs in iter9_pdfs/
    python export_data.py iter9_pdfs/foo.pdf  # single file
"""

import json, sys
import numpy as np
from pathlib import Path
from models import *

# ── Import your model functions ────────────────────────────────────────────
# If everything lives in one file, just paste the functions above this block
# or uncomment and rename the import:
# from models import (parse_pdf, calc_product_cost, get_true_mins_maxes,
#                     normalize_attributes, calc_utilities, tau_callibrate_df,
#                     step_tau_df, point_generation_price, point_generation_ratio,
#                     try_all_fits, sanity_check)

# ── Configuration ──────────────────────────────────────────────────────────
FIT_S_MIN      = 0.50
FIT_S_MAX      = 0.99
FIT_RATIO_RANGE = np.arange(2.0, 20.0, 0.1)
PRICE_RANGE     = np.arange(0.1, 20.0, 0.1)

# ── JSON serialiser that handles numpy types and non-finite floats ─────────
def _serial(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))

# ── Pipeline ───────────────────────────────────────────────────────────────
def _run_pipeline(pdf_path):
    df = parse_pdf(pdf_path)
    df = calc_product_cost(df)
    df = get_true_mins_maxes(df)
    df = normalize_attributes(df)
    df = calc_utilities(df)
    df = tau_callibrate_df(df)
    df = step_tau_df(df)
    return df

# ── Export one PDF ─────────────────────────────────────────────────────────
def export_product(pdf_path, output_dir="docs/data"):
    print(f"  Processing {Path(pdf_path).name} …")
    df      = _run_pipeline(pdf_path)
    product = df["cu_product"].iloc[0]
    regions = df["region"].tolist()

    # ── Graph point series ─────────────────────────────────────────────────
    graphs = {"copper_price": {}, "aluminum_price": {}, "ratio": {}}

    for region in regions:
        row = df[df["region"] == region].iloc[0]

        # Copper price sweep
        y_cu = point_generation_price(df, region, price_range=PRICE_RANGE, variable="cu")
        graphs["copper_price"][region] = {
            "x": (PRICE_RANGE * 1000).tolist(),
            "y": [float(v) for v in y_cu],
            "observed_x": float(row.get("copper_price_per_kg", 0)) * 1000,
            "observed_y": float(row.get("copper_product_market_share", 0.5)),
        }

        # Aluminum price sweep
        y_al = point_generation_price(df, region, price_range=PRICE_RANGE, variable="al")
        graphs["aluminum_price"][region] = {
            "x": (PRICE_RANGE * 1000).tolist(),
            "y": [float(v) for v in y_al],
            "observed_x": float(row.get("aluminum_price_per_kg", 0)) * 1000,
            "observed_y": float(row.get("copper_product_market_share", 0.5)),
        }

        # Price-ratio sweep
        y_ratio = point_generation_ratio(df, region, ratio_range=FIT_RATIO_RANGE)
        graphs["ratio"][region] = {
            "x": FIT_RATIO_RANGE.tolist(),
            "y": [float(v) for v in y_ratio],
        }

    # ── Curve fits (on ratio data) ─────────────────────────────────────────
    fit_results = []
    for region in regions:
        y_fit = point_generation_ratio(df, region, ratio_range=FIT_RATIO_RANGE)
        fit   = try_all_fits(FIT_RATIO_RANGE, y_fit, s_min=FIT_S_MIN, s_max=FIT_S_MAX)
        clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                 for k, v in fit.items()}
        fit_results.append({"region": region} | clean)

    # ── Sanity check ───────────────────────────────────────────────────────
    sanity = sanity_check(df)

    # ── Bundle and write ───────────────────────────────────────────────────
    payload = {
        "product":     product,
        "regions":     regions,
        "graphs":      graphs,
        "fit_results": fit_results,
        "fit_bounds":  {"s_min": FIT_S_MIN, "s_max": FIT_S_MAX},
        "sanity_check": sanity,
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(output_dir) / f"{product}.json"
    with open(out, "w") as f:
        json.dump(payload, f, default=_serial, indent=2)

    print(f"    ✓  {out}  ({len(regions)} regions)")
    return product


# ── Export all PDFs in a folder ────────────────────────────────────────────
def export_all(pdf_dir="iter9_pdfs", output_dir="docs/data"):
    products = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        try:
            products.append(export_product(pdf, output_dir))
        except Exception as e:
            print(f"    ✗  {pdf.name}: {e}")

    manifest = {"products": products}
    manifest_path = Path(output_dir) / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {manifest_path}: {products}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        export_product(sys.argv[1])
    else:
        export_all()

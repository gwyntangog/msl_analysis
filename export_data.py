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
FIT_S_MIN      = 0
FIT_S_MAX      = 1
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

    # ── Pre-computed graph series ──────────────────────────────────────────
    graphs = {"copper_price": {}, "aluminum_price": {}, "ratio": {}}
    for region in regions:
        row = df[df["region"] == region].iloc[0]

        y_cu = point_generation_price(df, region, price_range=PRICE_RANGE, variable="cu")
        graphs["copper_price"][region] = {
            "x": (PRICE_RANGE * 1000).tolist(),
            "y": [float(v) for v in y_cu],
            "observed_x": float(row.get("copper_price_per_kg", 0)) * 1000,
            "observed_y": float(row.get("copper_product_market_share", 0.5)),
        }

        y_al = point_generation_price(df, region, price_range=PRICE_RANGE, variable="al")
        graphs["aluminum_price"][region] = {
            "x": (PRICE_RANGE * 1000).tolist(),
            "y": [float(v) for v in y_al],
            "observed_x": float(row.get("aluminum_price_per_kg", 0)) * 1000,
            "observed_y": float(row.get("copper_product_market_share", 0.5)),
        }

        y_ratio = point_generation_ratio(df, region, ratio_range=FIT_RATIO_RANGE)
        graphs["ratio"][region] = {
            "x": FIT_RATIO_RANGE.tolist(),
            "y": [float(v) for v in y_ratio],
        }

    # ── Curve fits ─────────────────────────────────────────────────────────
    fit_results = []
    for region in regions:
        y_fit = point_generation_ratio(df, region, ratio_range=FIT_RATIO_RANGE)
        fit   = try_all_fits(FIT_RATIO_RANGE, y_fit, s_min=FIT_S_MIN, s_max=FIT_S_MAX)
        clean = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                 for k, v in fit.items()}
        fit_results.append({"region": region} | clean)

    # ── Sanity check ───────────────────────────────────────────────────────
    sanity = sanity_check(df)

    # ── Region parameters for live client-side computation ─────────────────
    # These mirror exactly what calc_utility_row / ms_logit need
    def sf(val, default=0.0):
        """Safe float — returns default for NaN/inf/missing."""
        try:
            v = float(val)
            return default if not np.isfinite(v) else v
        except (TypeError, ValueError):
            return default

    region_params = {}
    for _, row in df.iterrows():
        r = str(row["region"])
        p = {}

        # Observed prices ($/kg) and market share
        p["copper_price_per_kg"]          = sf(row.get("copper_price_per_kg"))
        p["aluminum_price_per_kg"]        = sf(row.get("aluminum_price_per_kg"))
        p["copper_product_market_share"]  = sf(row.get("copper_product_market_share"), 0.5)
        p["tau_value"]                    = sf(row.get("tau_value"), 1.0)

        # Product cost components (per material)
        for m in ["cu", "al"]:
            p[f"{m}_non_material_cost_per_unit"] = sf(row.get(f"{m}_non_material_cost_per_unit"))
            p[f"{m}_copper_kg"]                  = sf(row.get(f"{m}_copper_kg"))
            p[f"{m}_aluminum_kg"]                = sf(row.get(f"{m}_aluminum_kg"))

        # Attribute 1 (cost) normalisation bounds
        p["attribute_1_min"] = sf(row.get("attribute_1_min"), 0.0)
        p["attribute_1_max"] = sf(row.get("attribute_1_max"), 1.0)

        # Attributes 2–5 are pre-calibrated and don't change with price
        for i in range(2, 6):
            for m in ["cu", "al"]:
                p[f"{m}_a{i}_callibrated"] = sf(row.get(f"{m}_a{i}_callibrated"))

        # Attribute weights
        for i in range(1, 6):
            p[f"weight_attribute_{i}"] = sf(row.get(f"weight_attribute_{i}"))

        region_params[r] = p

    # ── Slider defaults and ranges ($/tonne) ───────────────────────────────
    cu_kg = [p["copper_price_per_kg"]   for p in region_params.values() if p["copper_price_per_kg"]   > 0]
    al_kg = [p["aluminum_price_per_kg"] for p in region_params.values() if p["aluminum_price_per_kg"] > 0]

    price_meta = {
        "cu_default": round(float(np.mean(cu_kg)) * 1000) if cu_kg else 8000,
        "al_default": round(float(np.mean(al_kg)) * 1000) if al_kg else 2500,
        "cu_min": 100,   "al_min": 100,
        "cu_max": 20000, "al_max": 20000,
        "cu_step": 100,  "al_step": 50,
    }

    # ── Write JSON ─────────────────────────────────────────────────────────
    payload = {
        "product":        product,
        "regions":        regions,
        "graphs":         graphs,
        "fit_results":    fit_results,
        "fit_bounds":     {"s_min": FIT_S_MIN, "s_max": FIT_S_MAX},
        "sanity_check":   sanity,
        "region_params":  region_params,
        "price_meta":     price_meta,
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

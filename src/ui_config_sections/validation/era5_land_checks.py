"""
ERA5-Land validation metrics and helpers.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def _mask_nan(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _align_lengths(sim: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim arrays to the same length (min of both) to avoid broadcast errors."""
    a = np.asarray(sim).reshape(-1)
    b = np.asarray(ref).reshape(-1)
    n = min(len(a), len(b))
    return a[:n], b[:n]


def compute_bias_rmse(sim: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    sim_aligned, ref_aligned = _align_lengths(sim, ref)
    sim_masked, ref_masked = _mask_nan(sim_aligned, ref_aligned)
    if sim_masked.size == 0:
        return {"bias": np.nan, "rmse": np.nan, "ubrmse": np.nan}
    diff = sim_masked - ref_masked
    bias = float(diff.mean())
    rmse = float(np.sqrt(np.mean(diff**2)))
    ubrmse = float(np.sqrt(np.mean((diff - bias) ** 2)))
    return {"bias": bias, "rmse": rmse, "ubrmse": ubrmse}


def compute_kge(sim: np.ndarray, ref: np.ndarray) -> float:
    sim_aligned, ref_aligned = _align_lengths(sim, ref)
    sim_masked, ref_masked = _mask_nan(sim_aligned, ref_aligned)
    if sim_masked.size == 0:
        return float("nan")
    r = np.corrcoef(sim_masked, ref_masked)[0, 1]
    alpha = sim_masked.std() / (ref_masked.std() + 1e-9)
    beta = sim_masked.mean() / (ref_masked.mean() + 1e-9)
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def summarize_era5_land_validation(sim_outputs: Dict, era_bundle: Dict) -> Dict[str, Dict]:
    """
    Compare simulated outputs against ERA5-Land bundle.

    Args:
        sim_outputs: Dict with arrays (e.g., {"soil_moisture_layers": {...}, "fluxes": {...}}).
    era_bundle: Output from build_era5_land_bundle.
    """
    results: Dict[str, Dict] = {}
    # Soil moisture layers
    sim_sm = sim_outputs.get("soil_moisture_layers", {})
    era_sm = era_bundle.get("soil_moisture_layers", {})
    for name, era_arr in era_sm.items():
        if name not in sim_sm:
            continue
        metrics = compute_bias_rmse(sim_sm[name], era_arr)
        metrics["kge"] = compute_kge(sim_sm[name], era_arr)
        results[f"soil_moisture/{name}"] = metrics

    # Surface fluxes
    sim_fluxes = sim_outputs.get("fluxes", {})
    era_fluxes = era_bundle.get("fluxes", {})
    for key in ("latent_heat_flux", "sensible_heat_flux", "runoff"):
        if key in sim_fluxes and key in era_fluxes:
            sim_arr, era_arr = _align_lengths(sim_fluxes[key], era_fluxes[key])
            # Mask very small values to avoid low-variance effects
            mask = (np.abs(sim_arr) > 0.1) | (np.abs(era_arr) > 0.1)
            sim_arr = sim_arr[mask]
            era_arr = era_arr[mask]
            metrics = compute_bias_rmse(sim_arr, era_arr)
            metrics["kge"] = compute_kge(sim_arr, era_arr)
            results[f"fluxes/{key}"] = metrics
    if "rain" in sim_outputs and "rain" in era_bundle:
        metrics = compute_bias_rmse(sim_outputs["rain"], era_bundle["rain"])
        metrics["kge"] = compute_kge(sim_outputs["rain"], era_bundle["rain"])
        results["rain"] = metrics
    if "et0" in sim_outputs and "et0" in era_bundle:
        metrics = compute_bias_rmse(sim_outputs["et0"], era_bundle["et0"])
        metrics["kge"] = compute_kge(sim_outputs["et0"], era_bundle["et0"])
        results["et0"] = metrics
    if "ETc" in sim_outputs and "ETc" in era_bundle:
        metrics = compute_bias_rmse(sim_outputs["ETc"], era_bundle["ETc"])
        metrics["kge"] = compute_kge(sim_outputs["ETc"], era_bundle["ETc"])
        results["ETc"] = metrics
    return results

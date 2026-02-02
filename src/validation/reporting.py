"""
Helpers to format ERA5-Land validation outputs for UI/CLI.
"""

from __future__ import annotations

import json
from typing import Dict

import numpy as np

from src.validation.era5_land_checks import summarize_era5_land_validation


def rollout_to_sim_outputs(rollout: Dict) -> Dict[str, Dict]:
    """
    Convert a rollout dict (from evaluate_episode) into the structure expected by
    summarize_era5_land_validation. Only available variables are used.
    """
    sim: Dict[str, object] = {}
    if "R" in rollout:
        sim["rain"] = np.asarray(rollout["R"], dtype=np.float32)
    if "et0" in rollout:
        sim["et0"] = np.asarray(rollout["et0"], dtype=np.float32)
    if "Kc" in rollout:
        sim["Kc"] = np.asarray(rollout["Kc"], dtype=np.float32)
    # Extra diagnostics (not compared today but useful to surface):
    if "ETc" in rollout:
        sim["ETc"] = np.asarray(rollout["ETc"], dtype=np.float32)
    if "psi" in rollout:
        sim["psi"] = np.asarray(rollout["psi"], dtype=np.float32)
    if "S" in rollout:
        sim["S"] = np.asarray(rollout["S"], dtype=np.float32)
    if "fluxes" in rollout:
        sim["fluxes"] = rollout["fluxes"]
    if "soil_moisture_layers" in rollout:
        sim["soil_moisture_layers"] = rollout["soil_moisture_layers"]
    return sim


def format_validation_json(sim_outputs: Dict, era_bundle: Dict) -> str:
    results = summarize_era5_land_validation(sim_outputs, era_bundle)
    return json.dumps(results, indent=2, default=lambda x: float(x))


def format_validation_table(results: Dict[str, Dict]) -> str:
    """
    Render a simple markdown table from validation results.
    """
    if not results:
        return "_No matching variables to compare._"
    # Order: rain/et0/ETc, fluxes/*, soil_moisture/*, then others
    preferred = ["rain", "et0", "ETc"]
    flux_keys = sorted([k for k in results if k.startswith("fluxes/")])
    sm_keys = sorted([k for k in results if k.startswith("soil_moisture/")])
    remaining = sorted([k for k in results if k not in preferred + flux_keys + sm_keys])
    ordered = [k for k in preferred if k in results] + flux_keys + sm_keys + remaining

    lines = ["| Metric | Bias | RMSE | uBRMSE | KGE |", "| --- | --- | --- | --- | --- |"]
    for name in ordered:
        metrics = results[name]
        lines.append(
            f"| {name} | "
            f"{metrics.get('bias', float('nan')):.3f} | "
            f"{metrics.get('rmse', float('nan')):.3f} | "
            f"{metrics.get('ubrmse', float('nan')):.3f} | "
            f"{metrics.get('kge', float('nan')) if 'kge' in metrics else float('nan'):.3f} |"
        )
    return "\n".join(lines)

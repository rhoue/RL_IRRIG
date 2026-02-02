"""
CLI to validate model outputs against ERA5-Land.

Usage:
  python scripts/validate_era5_land.py --era5-file data/era5_land_sample.nc --rollout-npz outputs/rollout.npz

Requirements:
  - xarray, cdsapi dependencies are not needed here unless your ERA file requires specific engines.
  - Rollout NPZ should contain arrays named at least 'rain' (and optionally 'et0', 'fluxes', 'soil_moisture_layers').
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data_loader import load_era5_land_dataset
from src.validation.reporting import format_validation_json


def main():
    parser = argparse.ArgumentParser(description="Validate rollout against ERA5-Land bundle.")
    parser.add_argument("--era5-file", required=True, help="Path to ERA5-Land NetCDF/Zarr.")
    parser.add_argument(
        "--rollout-npz",
        required=True,
        help="Path to rollout NPZ containing rain/et0/etc.",
    )
    parser.add_argument("--resample-freq", default="1D", help="Resample frequency (default 1D).")
    args = parser.parse_args()

    if not Path(args.rollout_npz).exists():
        raise SystemExit(f"Rollout file not found: {args.rollout_npz}")
    rollout_npz = np.load(args.rollout_npz, allow_pickle=True)
    sim_outputs = {}
    for key in ("rain", "et0"):
        if key in rollout_npz:
            sim_outputs[key] = rollout_npz[key]
    if "fluxes" in rollout_npz:
        sim_outputs["fluxes"] = rollout_npz["fluxes"].item() if rollout_npz["fluxes"].shape == () else rollout_npz["fluxes"]
    if "soil_moisture_layers" in rollout_npz:
        sim_outputs["soil_moisture_layers"] = rollout_npz["soil_moisture_layers"].item() if rollout_npz["soil_moisture_layers"].shape == () else rollout_npz["soil_moisture_layers"]

    if "rain" not in sim_outputs:
        raise SystemExit("Rollout NPZ must contain at least a 'rain' array.")

    era_bundle = load_era5_land_dataset(
        args.era5_file,
        resample_freq=args.resample_freq,
        soil_depth_mapping=None,
    )
    print(format_validation_json(sim_outputs, era_bundle))


if __name__ == "__main__":
    main()

"""
Utility script to download a small ERA5-Land sample via CDS API.

Prereqs:
  - Create ~/.cdsapirc with your CDS credentials:
        url: https://cds.climate.copernicus.eu/api/v2
        key: <uid>:<api-key>
        verify: 1
  - pip install cdsapi

Usage (examples):
  python scripts/download_era5_land.py --out data/era5_land_sample.nc
  python scripts/download_era5_land.py --year 2020 --month 06 --day 15 --bbox 5 45 7 43

Notes:
  - Default request pulls a single day, global 0.1° grid, hourly, to keep the file small.
  - You can widen date range or variables as needed; adjust request dict accordingly.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_request(year: str, month: str, day: str, bbox: list[float]) -> dict:
    return {
        "format": "netcdf",
        "variable": [
            "total_precipitation",
            "potential_evaporation",
            "surface_runoff",
            "volumetric_soil_water_layer_1",
            "volumetric_soil_water_layer_2",
            "volumetric_soil_water_layer_3",
            "volumetric_soil_water_layer_4",
            "surface_latent_heat_flux",
            "surface_sensible_heat_flux",
        ],
        "year": year,
        "month": month,
        "day": day,
        "time": [f"{h:02d}:00" for h in range(24)],
        # Bounding box: North, West, South, East
        "area": bbox,
    }


def main():
    parser = argparse.ArgumentParser(description="Download ERA5-Land sample via CDS API.")
    parser.add_argument("--year", default="2020")
    parser.add_argument("--month", default="06")
    parser.add_argument("--day", default="15")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("N", "W", "S", "E"),
        default=[50.0, 0.0, 40.0, 10.0],
        help="Bounding box North West South East (default covers part of Europe).",
    )
    parser.add_argument("--out", default="data/era5_land_sample.nc", help="Output NetCDF path.")
    args = parser.parse_args()

    try:
        import cdsapi  # type: ignore
    except ImportError as exc:
        raise SystemExit("cdsapi is required. Install with `pip install cdsapi`.") from exc

    req = build_request(args.year, args.month, args.day, args.bbox)
    client = cdsapi.Client()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    client.retrieve("reanalysis-era5-land", req, str(out_path))
    print(f"✅ Downloaded ERA5-Land sample to {out_path}")


if __name__ == "__main__":
    main()

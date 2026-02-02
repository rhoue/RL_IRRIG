"""
Download ERA5-Land for a rainy-like season (France) via CDS API, month by month
to stay within cost limits. Designed for SW France (Hautes-Pyrénées / Adour) by default.

Requirements:
- ~/.cdsapirc with your CDS API key:
    url: https://cds.climate.copernicus.eu/api
    key: <your-api-key>
    verify: 1
- Accept the ERA5-Land licence on the CDS website.
- pip install cdsapi

Output:
- One NetCDF per month (e.g., data/era5_land_fr_spring2025_2025_03.nc).
- You can merge and/or convert to NetCDF3 using scripts/merge_era5_land_months.py if needed.
"""

from __future__ import annotations

from pathlib import Path
import cdsapi


DEFAULT_VARS = [
    # precipitation and evaporation
    "total_precipitation",
    "potential_evaporation",
    # runoff and soil moisture
    "surface_runoff",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    # optional fluxes
    "surface_latent_heat_flux",
    "surface_sensible_heat_flux",
]


def download_era5_land_season(
    out_path: str = "data/era5_land_fr_spring2025.nc",
    years: list[str] | None = None,
    months: list[str] | None = None,
    bbox: list[float] | None = None,
    max_days: int = 10,
    variables: list[str] | None = None,
):
    """
    Download ERA5-Land for a rainy-like season (default: spring 2025 over France) **month-by-month**
    to stay within CDS cost limits.

    Args:
        out_path: Output path for the NetCDF (may be a zip).
        years: List of years (as strings). Default: ["2025"].
        months: List of months. Default: March–May.
        bbox: [North, West, South, East] bounding box. Default: France.
    """
    c = cdsapi.Client()
    years = years or ["2025"]
    months = months or ["03", "04", "05"]
    days = [f"{d:02d}" for d in range(1, max_days + 1)]
    times = [f"{h:02d}:00" for h in range(24)]
    # Default bbox: South-West France, Hautes-Pyrénées / Adour area
    # Approx: North 43.8, West -1.0, South 42.7, East 0.8
    bbox_france = bbox or [43.8, -1.0, 42.7, 0.8]  # N W S E
    variables = variables or DEFAULT_VARS

    out_base = Path(out_path)
    for y in years:
        for m in months:
            req = {
                "format": "netcdf",
                "variable": variables,
                "year": [y],
                "month": [m],
                "day": days,
                "time": times,
                "area": bbox_france,
            }
            month_out = out_base.with_name(f"{out_base.stem}_{y}_{m}{out_base.suffix}")
            month_out.parent.mkdir(parents=True, exist_ok=True)
            c.retrieve("reanalysis-era5-land", req, str(month_out))
            print(f"✅ Downloaded {y}-{m} to {month_out}")


if __name__ == "__main__":
    download_era5_land_season()

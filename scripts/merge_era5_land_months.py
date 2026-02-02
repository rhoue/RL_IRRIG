"""
Merge multiple ERA5-Land monthly downloads into a single NetCDF3 file for the app.

Assumes each monthly file is a zip containing `data_0.nc` (CDS default).

Usage:
  python scripts/merge_era5_land_months.py \
    --inputs data/era5_land_fr_spring2025_2025_03.nc data/era5_land_fr_spring2025_2025_04.nc data/era5_land_fr_spring2025_2025_05.nc \
    --out data/era5_land_fr_spring2025_all_nc3.nc

Dependencies: xarray, netcdf4 (for reading NC4), and the files present in ./data.
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path
from typing import List

import xarray as xr


def _extract_nc_from_zip(zip_path: Path, dest: Path) -> Path:
    """Extract data_0.nc from a zip (.nc) and return the extracted path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.endswith(".nc")]
        if not members:
            raise FileNotFoundError(f"No .nc member found in {zip_path}")
        target = members[0]
        zf.extract(target, path=dest.parent)
        extracted = dest.parent / target
        extracted.rename(dest)
    return dest


def merge_months(input_paths: List[Path], out_path: Path, netcdf3: bool = True) -> None:
    datasets = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        for p in input_paths:
            if not p.exists():
                raise FileNotFoundError(f"Input file not found: {p}")
            extracted = tmpdir / f"{p.stem}_data.nc"
            if zipfile.is_zipfile(p):
                extracted = _extract_nc_from_zip(p, extracted)
            else:
                extracted = p
            ds = xr.open_dataset(extracted, engine="netcdf4")
            datasets.append(ds)

        merged = xr.concat(datasets, dim="valid_time")
        merged = merged.sortby("valid_time")
        encoding = {var: {} for var in merged.data_vars}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if netcdf3:
            merged.to_netcdf(out_path, format="NETCDF3_CLASSIC", encoding=encoding)
        else:
            merged.to_netcdf(out_path, encoding=encoding)
        print(f"✅ Wrote merged file: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge ERA5-Land monthly downloads into one NetCDF file.")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of monthly .nc zip files from CDS.")
    parser.add_argument("--out", required=True, help="Output path (e.g., data/era5_land_all_nc3.nc).")
    parser.add_argument(
        "--keep-format",
        action="store_true",
        help="Keep source NetCDF format (skip NetCDF3 conversion). Default: convert to NetCDF3_CLASSIC.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    out_path = Path(args.out)
    merge_months(input_paths, out_path, netcdf3=not args.keep_format)


if __name__ == "__main__":
    main()

"""
Lightweight ERA5-Land ingestion helpers.

These functions only run when the ERA5-Land feature flag is enabled to avoid
adding hard dependencies for the default synthetic pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def _require_xarray():
    try:
        import xarray as xr  # type: ignore
    except ImportError as exc:  # pragma: no cover - guard for optional dep
        raise ImportError(
            "xarray is required for ERA5-Land support. "
            "Install with `pip install xarray` (and netCDF4/zarr depending on your files)."
        ) from exc
    return xr


def load_era5_land_raw(
    path: str | Path,
    chunks: Optional[dict] = None,
    drop_vars: Optional[Iterable[str]] = None,
    engine: Optional[str] = None,
):
    """
    Open an ERA5-Land NetCDF/Zarr file without preprocessing.

    Args:
        path: Local path to the ERA5-Land file.
        chunks: Optional dask chunks to stream big files.
        drop_vars: Variables to drop eagerly (e.g., large QC flags).

    Returns:
        xarray.Dataset
    """
    xr = _require_xarray()
    errors = []
    # Try requested engine first, then common fallbacks
    engines = [engine] if engine else ["netcdf4", "h5netcdf", "scipy", None]
    for eng in engines:
        try:
            ds = xr.open_dataset(path, chunks=chunks, engine=eng)
            if drop_vars:
                ds = ds.drop_vars(list(drop_vars), errors="ignore")
            return ds
        except Exception as exc:  # pragma: no cover - fallback path
            errors.append(f"{eng}: {exc}")
            continue
    raise RuntimeError(f"Failed to open {path} with engines {engines}. Errors: {errors}")

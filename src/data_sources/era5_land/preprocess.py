"""
Preprocessing helpers for ERA5-Land grids and time steps.
"""

from __future__ import annotations

from typing import Dict, Optional


def _require_xarray():
    try:
        import xarray as xr  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "xarray is required for ERA5-Land support. "
            "Install with `pip install xarray` (and netCDF4/zarr depending on your files)."
        ) from exc
    return xr


def regrid_and_resample(
    ds,
    target_grid: Optional[Dict[str, object]] = None,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    time_name: str = "time",
    freq: Optional[str] = "1D",
    method: str = "nearest",
):
    """
    Align ERA5-Land data to the model grid and timestep.

    Args:
        ds: xarray.Dataset with ERA5-Land variables.
        target_grid: Optional dict with 'lat' and 'lon' arrays for interpolation.
        lat_name, lon_name, time_name: Dimension names.
        freq: Resampling frequency (None to skip).
        method: Interpolation method for xarray.interp.
    """
    xr = _require_xarray()
    work = ds
    # Fallback for datasets using 'valid_time' instead of 'time'
    if time_name not in work.dims and "valid_time" in work.dims:
        time_name = "valid_time"
    if target_grid and all(k in target_grid for k in ("lat", "lon")):
        target_lat = target_grid["lat"]
        target_lon = target_grid["lon"]
        work = work.interp({lat_name: target_lat, lon_name: target_lon}, method=method)
    if freq:
        work = work.resample({time_name: freq}).mean(keep_attrs=True)
    # Ensure sorted time for predictable downstream alignment
    if time_name in work.dims:
        work = work.sortby(time_name)
    # Cast to float32 to reduce memory footprint
    for var in work.data_vars:
        # Use numpy's can_cast to avoid xarray version differences
        try:
            import numpy as np
            if np.can_cast(work[var].dtype, "float32"):
                work[var] = work[var].astype("float32")
        except Exception:
            # Fallback: best-effort cast
            try:
                work[var] = work[var].astype("float32")
            except Exception:
                pass
    return work

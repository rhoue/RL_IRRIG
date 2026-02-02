"""
Feature extraction from ERA5-Land.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .preprocess import regrid_and_resample


def _require_xarray():
    try:
        import xarray as xr  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "xarray is required for ERA5-Land support. "
            "Install with `pip install xarray` (and netCDF4/zarr depending on your files)."
        ) from exc
    return xr


def _first_existing_var(ds, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if name in ds:
            return name
    return None


def _safe_to_numpy(data_array):
    np_array = np.asarray(data_array)
    return np_array.astype(np.float32)


def _nan_to_num(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Replace NaN/inf with a finite fill value."""
    return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)


def _to_time_series(da, time_name: str = "time") -> np.ndarray:
    """
    Convert an xarray.DataArray to a 1D time series by averaging over spatial dims.
    """
    if time_name not in da.dims and "valid_time" in da.dims:
        time_name = "valid_time"
    try:
        da = da.fillna(0)
    except Exception:
        pass
    if time_name in da.dims:
        # Move time to the first dimension
        da = da.transpose(time_name, ...)
    arr = _safe_to_numpy(da)
    if arr.ndim > 1:
        # Average across all non-time dimensions to get a single series
        arr = arr.mean(axis=tuple(range(1, arr.ndim)))
    return arr


def extract_surface_fluxes(
    ds,
    precip_vars: Iterable[str] = ("tp", "total_precipitation"),
    pet_vars: Iterable[str] = ("pev", "potential_evaporation", "pet"),
    runoff_vars: Iterable[str] = ("ro", "runoff", "surface_runoff", "sro"),
    latent_vars: Iterable[str] = ("slhf", "latent_heat_flux"),
    sensible_vars: Iterable[str] = ("sshf", "sensible_heat_flux"),
    time_name: str = "time",
) -> Dict[str, np.ndarray]:
    """
    Extract precipitation, PET proxy, and surface fluxes from ERA5-Land.

    Units are converted to mm where applicable; heat fluxes stay in W/m^2.
    """
    xr = _require_xarray()
    out: Dict[str, np.ndarray] = {}

    precip_name = _first_existing_var(ds, precip_vars)
    if precip_name is None:
        raise ValueError(f"None of the precipitation vars {list(precip_vars)} found in ERA5-Land dataset.")
    precip = ds[precip_name]
    # ERA5 precipitation is typically in meters; convert to mm
    if precip.attrs.get("units", "").lower() in ("m", "meter", "metres", "meters"):
        precip = precip * 1000.0
    out["rain"] = _to_time_series(precip, time_name=time_name)

    pet_name = _first_existing_var(ds, pet_vars)
    if pet_name is not None:
        pet = ds[pet_name]
        # ERA5 potential evaporation is negative (evaporation as upward flux)
        pet = np.abs(_to_time_series(pet, time_name=time_name))
        # If units are meters, convert to mm
        if pet_name in ds and isinstance(ds[pet_name], xr.DataArray):
            unit = ds[pet_name].attrs.get("units", "").lower()
            if unit in ("m", "meter", "metres", "meters"):
                pet = pet * 1000.0
        out["et0"] = pet

    runoff_name = _first_existing_var(ds, runoff_vars)
    if runoff_name is not None:
        runoff = ds[runoff_name]
        if runoff.attrs.get("units", "").lower() in ("m", "meter", "metres", "meters"):
            runoff = runoff * 1000.0
        out["runoff"] = _to_time_series(runoff, time_name=time_name)

    latent_name = _first_existing_var(ds, latent_vars)
    if latent_name is not None:
        out["latent_heat_flux"] = _to_time_series(ds[latent_name], time_name=time_name)

    sensible_name = _first_existing_var(ds, sensible_vars)
    if sensible_name is not None:
        out["sensible_heat_flux"] = _to_time_series(ds[sensible_name], time_name=time_name)

    if time_name in ds:
        out["time_index"] = np.asarray(ds[time_name].values)
    return out


def map_soil_moisture_layers(
    ds,
    depth_mapping: Dict[str, List[int]],
    var_names: Sequence[str] = ("swvl1", "swvl2", "swvl3", "swvl4"),
    time_name: str = "time",
) -> Dict[str, np.ndarray]:
    """
    Map ERA5-Land soil moisture layers to custom depth bins.

    Args:
        ds: ERA5-Land dataset with soil moisture layers.
        depth_mapping: Dict of output name -> list of layer indices to average.
            Example: {"0_7cm": [0], "7_28cm": [1], "28_100cm": [2], "100_289cm": [3]}
        var_names: Names of soil water volumetric layer variables in ERA5-Land.
    """
    available_vars = [name for name in var_names if name in ds]
    if not available_vars:
        raise ValueError("No ERA5-Land soil moisture variables found.")

    layers = []
    for name in var_names:
        if name in ds:
            layers.append(ds[name])
    if not layers:
        raise ValueError(f"Expected at least one soil moisture var in {var_names}")
    merged_layers = []
    for layer in layers:
        arr = _to_time_series(layer, time_name=time_name)
        merged_layers.append(arr)
    merged = np.stack(merged_layers, axis=-1)
    out: Dict[str, np.ndarray] = {}
    for out_name, indices in depth_mapping.items():
        if not indices:
            continue
        indices = [idx for idx in indices if idx < merged.shape[-1]]
        if not indices:
            continue
        out[out_name] = merged[..., indices].mean(axis=-1)
    if time_name in ds:
        out["time_index"] = np.asarray(ds[time_name].values)
    return out


def _default_kc(length: int) -> np.ndarray:
    kc = np.zeros(length, dtype=np.float32)
    for t in range(length):
        if t < 20:
            kc[t] = 0.3
        elif t < 50:
            kc[t] = 0.3 + (1.15 - 0.3) * (t - 20) / (50 - 20)
        elif t < 90:
            kc[t] = 1.15
        else:
            kc[t] = 1.15 + (0.7 - 1.15) * (t - 90) / max(length - 90, 1)
    return kc


def build_era5_land_bundle(
    ds,
    target_grid: Optional[Dict[str, object]] = None,
    soil_depth_mapping: Optional[Dict[str, List[int]]] = None,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
    time_name: str = "time",
    freq: Optional[str] = "1D",
) -> Dict[str, object]:
    """
    Full pipeline: align grid, extract fluxes, optional soil moisture mapping.
    """
    if time_name not in ds.dims and "valid_time" in ds.dims:
        time_name = "valid_time"
    aligned = regrid_and_resample(
        ds,
        target_grid=target_grid,
        lat_name=lat_name,
        lon_name=lon_name,
        time_name=time_name,
        freq=freq,
    )
    # Provide a sensible default soil depth mapping if SWVL layers exist
    if soil_depth_mapping is None and any(v in aligned.data_vars for v in ("swvl1", "swvl2", "swvl3", "swvl4")):
        soil_depth_mapping = {
            "layer_0_7cm": [0],
            "layer_7_28cm": [1],
            "layer_28_100cm": [2],
            "layer_100_289cm": [3],
        }
    fluxes = extract_surface_fluxes(aligned, time_name=time_name)
    bundle: Dict[str, object] = {
        "rain": _nan_to_num(fluxes["rain"]),
        "Kc": _nan_to_num(_default_kc(len(fluxes["rain"]))),
        "fluxes": fluxes,
    }
    if "et0" in fluxes:
        bundle["et0"] = _nan_to_num(fluxes["et0"])
        # Derived ETc reference (ET0 * Kc) for comparison with simulated ETc
        bundle["ETc"] = _nan_to_num(bundle["et0"] * bundle["Kc"])
    if soil_depth_mapping:
        bundle["soil_moisture_layers"] = map_soil_moisture_layers(
            aligned, depth_mapping=soil_depth_mapping, time_name=time_name
        )
        # Clean NaNs in soil moisture layers
        for k, v in bundle["soil_moisture_layers"].items():
            bundle["soil_moisture_layers"][k] = _nan_to_num(v)
        # Convert volumetric swvl layers to mm water using layer thickness
        depth_thickness_mm = {0: 70.0, 1: 210.0, 2: 720.0, 3: 1890.0}
        sm_mm: Dict[str, np.ndarray] = {}
        for name, vals in bundle["soil_moisture_layers"].items():
            idxs = soil_depth_mapping.get(name, [])
            thickness = sum(depth_thickness_mm.get(i, 0.0) for i in idxs)
            if thickness > 0:
                sm_mm[name] = vals * (thickness / 1000.0) * 1000.0  # volumetric * depth(m) *1000 -> mm
        if sm_mm:
            # Replace layers with mm values and add bucket_total_mm over root zone (default: top 28 cm)
            bundle["soil_moisture_layers"] = sm_mm
            root_layers = [k for k in ("layer_0_7cm", "layer_7_28cm") if k in sm_mm]
            if root_layers:
                root_sum = sum(sm_mm[k] for k in root_layers)
            else:
                root_sum = sum(sm_mm.values())
            bundle["soil_moisture_layers"]["bucket_total_mm"] = root_sum
    # Clean fluxes NaNs
    for k, v in list(fluxes.items()):
        if isinstance(v, np.ndarray):
            fluxes[k] = _nan_to_num(v)
    # Drop time_index from fluxes to avoid propagating datetime arrays into numeric metrics
    fluxes.pop("time_index", None)
    if "time_index" in fluxes:
        bundle["time_index"] = fluxes["time_index"]
    return bundle

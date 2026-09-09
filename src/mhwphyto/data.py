"""Data access layer.

The point of this module is that every downstream function takes an
xarray.Dataset with the same variable names, so swapping synthetic data for
real observations is a one-line change and no analysis code moves.

    ds = load(source="synthetic")          # works offline, today
    ds = load(source="local", path=...)    # after someone runs the downloads
    ds = load(source="cmems", ...)         # needs credentials + network

IMPORTANT: the product identifiers below were written from memory and are
NOT verified. Confirm every one of them in the Copernicus Marine catalogue
before building against them. Treat them as a shopping list, not gospel.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

# --------------------------------------------------------------- catalogue
# TO VERIFY -- product/dataset IDs change between catalogue versions.
CANDIDATE_PRODUCTS = {
    "sst_l4_global": dict(
        note="Daily gap-free L4 SST, ~0.05 deg, 1982-present. Basis for MHW detection.",
        cmems_product="SST_GLO_SST_L4_REP_OBSERVATIONS_010_011",
        alternative="NOAA OISST v2.1 via ERDDAP / THREDDS (0.25 deg, 1981-present)",
        variables=["analysed_sst"],
    ),
    "ocean_colour_chl": dict(
        note="Merged multi-sensor chlorophyll-a, L4 gap-filled, 1997-present.",
        cmems_product="OCEANCOLOUR_GLO_BGC_L4_MY_009_104",
        alternative="ESA OC-CCI v6; NASA OB.DAAC L3 mapped",
        variables=["CHL"],
    ),
    "phyto_functional_types": dict(
        note="Chlorophyll split by phytoplankton functional type -- the community-composition angle.",
        cmems_product="OCEANCOLOUR_GLO_BGC_L4_MY_009_104 (PFT variables)",
        variables=["DIATO", "DINO", "NANO", "PICO", "HAPTO", "PROKAR"],
    ),
    "biogeochem_reanalysis": dict(
        note="MLD, nitrate, phosphate, net primary production. Mechanism variables.",
        cmems_product="GLOBAL_MULTIYEAR_BGC_001_029",
        variables=["nppv", "no3", "po4", "o2"],
    ),
    "physics_reanalysis": dict(
        note="Mixed layer depth and subsurface temperature.",
        cmems_product="GLOBAL_MULTIYEAR_PHY_001_030",
        variables=["mlotst", "thetao"],
    ),
    "bgc_argo": dict(
        note="Subsurface chlorophyll fluorescence, backscatter, nitrate. Ground truth below the skin.",
        access="Euro-Argo / Coriolis GDAC (Ifremer), argopy python package",
        variables=["CHLA", "BBP700", "NITRATE", "TEMP"],
    ),
    "coastal_insitu": dict(
        note="French coastal phytoplankton and harmful-bloom counts, species level.",
        access="REPHY / SOMLIT / EMODnet Biology",
    ),
}

REQUIRED_VARS = ("sst", "chl")
OPTIONAL_VARS = ("mld", "par", "nppv", "no3")

# canonical renaming from common source names to ours
RENAME_MAP = {
    "analysed_sst": "sst", "sst": "sst", "sea_surface_temperature": "sst",
    "CHL": "chl", "chlor_a": "chl", "chl": "chl", "chl_ocx": "chl",
    "mlotst": "mld", "mld": "mld", "MLD": "mld",
    "par": "par", "PAR": "par",
    "nppv": "nppv", "no3": "no3",
}


def print_catalogue() -> None:
    for key, meta in CANDIDATE_PRODUCTS.items():
        print(f"\n{key}")
        for k, v in meta.items():
            print(f"    {k:14s} {v}")


def harmonise(ds: xr.Dataset) -> xr.Dataset:
    """Rename variables/coords to canonical names and fix SST units."""
    ren = {k: v for k, v in RENAME_MAP.items() if k in ds.variables}
    ds = ds.rename(ren)
    for a, b in (("latitude", "lat"), ("longitude", "lon")):
        if a in ds.coords:
            ds = ds.rename({a: b})
    if "sst" in ds and float(np.nanmean(ds["sst"].values)) > 200:  # kelvin
        ds["sst"] = ds["sst"] - 273.15
        ds["sst"].attrs["units"] = "degree_Celsius"
    missing = [v for v in REQUIRED_VARS if v not in ds]
    if missing:
        raise KeyError(f"dataset lacks required variables {missing}; "
                       f"has {list(ds.data_vars)}")
    return ds


def load(source: str = "synthetic", path: str | Path | None = None, **kwargs) -> xr.Dataset:
    if source == "synthetic":
        from .synth import make_dataset
        return make_dataset(**kwargs)

    if source == "local":
        if path is None:
            raise ValueError("source='local' needs path=")
        p = Path(path)
        ds = xr.open_mfdataset(str(p), combine="by_coords") if any(c in str(p) for c in "*?") \
            else xr.open_dataset(p)
        return harmonise(ds)

    if source == "cmems":
        # pip install copernicusmarine ; then `copernicusmarine login`
        try:
            import copernicusmarine as cm
        except ImportError as exc:
            raise ImportError(
                "pip install copernicusmarine, then run `copernicusmarine login`"
            ) from exc
        required = {"dataset_id", "variables", "bbox", "start", "end"}
        if not required.issubset(kwargs):
            raise ValueError(f"cmems needs {sorted(required)}")
        w, s, e, n = kwargs["bbox"]
        ds = cm.open_dataset(
            dataset_id=kwargs["dataset_id"],
            variables=kwargs["variables"],
            minimum_longitude=w, maximum_longitude=e,
            minimum_latitude=s, maximum_latitude=n,
            start_datetime=kwargs["start"], end_datetime=kwargs["end"],
            username=os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"),
            password=os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD"),
        )
        return harmonise(ds)

    raise ValueError(f"unknown source {source!r}")


def cache_to_zarr(ds: xr.Dataset, path: str | Path) -> Path:
    """Write a regional subset to zarr. Do this BEFORE the hackathon --
    venue bandwidth and portal throttling break more teams than bad ideas."""
    p = Path(path)
    ds.chunk({"time": 365}).to_zarr(p, mode="w", consolidated=True)
    return p

"""
Py-ART radar data loaders.

This module provides functions for loading radar data using the Py-ART library.
"""

import warnings
import numpy as np
import pandas as pd
import xarray as xr
import pyart
from datetime import timedelta as td


def open_mf_pyart_dataset(fpaths, elevation):
    """
    Open multiple radar files using Py-ART and combine into a dataset.

    Loads multiple radar files, extracts a specific sweep, and concatenates
    them into a single xarray Dataset.

    Parameters
    ----------
    fpaths : list
        List of file paths to radar files.
    elevation : int
        Sweep number (elevation index) to extract.

    Returns
    -------
    xarray.Dataset
        Combined dataset with all time steps.

    Examples
    --------
    >>> ds = open_mf_pyart_dataset(["radar1.nc", "radar2.nc"], elevation=0)
    """
    radars = [pyart.io.read(fpath) for fpath in fpaths]
    dss = [extract_sweep_from_pyart_radar(rad, elevation) for rad in radars]
    if determine_need_normalization(dss):
        dss = normalize_range_wise(dss)
    big_ds = xr.concat(dss, dim='time')
    return big_ds


def extract_sweep_from_pyart_radar(radar, sweep_number):
    """
    Extract a single sweep from a Py-ART radar object as an xarray Dataset.

    Converts Py-ART radar data to xarray format with proper coordinates
    and metadata.

    Parameters
    ----------
    radar : pyart.core.Radar
        Py-ART Radar object.
    sweep_number : int
        Index of the sweep to extract.

    Returns
    -------
    xarray.Dataset
        Dataset containing the sweep data with coordinates:
        'azimuth', 'range', 'elevation', 'rtime', 'time', 'longitude',
        'latitude', 'altitude'.

    Examples
    --------
    >>> radar = pyart.io.read("radar_file.nc")
    >>> ds = extract_sweep_from_pyart_radar(radar, 0)
    """
    sweep = radar.extract_sweeps([sweep_number])
    coords = {
        'azimuth': sweep.azimuth,
        'range': sweep.range
    }

    site = {
        'longitude': sweep.longitude,
        'latitude': sweep.latitude,
        'altitude': sweep.altitude
    }

    fields = sweep.fields

    ds = xr.Dataset()

    for key in coords.keys():
        ccopy = coords[key].copy()
        da = xr.DataArray(ccopy.pop('data'), dims=key, attrs=ccopy)
        da.name = key
        ds.coords[key] = da

    time = pd.to_datetime(sweep.time['units'].split(' ')[-1][:-1])
    rtime = [time + td(seconds=int(i)) for i in sweep.time['data']]
    ds = ds.assign_coords({
        'elevation': (['azimuth'], sweep.elevation['data']),
        'rtime': (['azimuth'], rtime)
    })

    ds = ds.assign_coords({'time': ds.rtime.to_numpy().min()})

    for key in fields.keys():
        ccopy = fields[key].copy()
        da = xr.DataArray(
            ccopy.pop('data'),
            dims=['azimuth', 'range'],
            coords={'range': ds['range'], 'azimuth': ds['azimuth']},
            attrs=ccopy
        )
        ds[key] = da

    for key in site.keys():
        ccopy = site[key].copy()
        da = xr.DataArray(ccopy.pop('data'), attrs=ccopy)
        da.name = key
        ds.coords[key] = da

    ds.attrs = {
        'fixed_angle': sweep.fixed_angle['data'],
        'other': sweep.metadata
    }
    ds = ds.interp({'azimuth': np.arange(0.5, 360, 1)}, method='nearest')
    return ds


def determine_need_normalization(ds_list):
    """
    Determine if range normalization is needed across datasets.

    Checks if datasets have different range dimensions that would
    require interpolation before concatenation.

    Parameters
    ----------
    ds_list : list
        List of xarray Datasets.

    Returns
    -------
    bool
        True if normalization is needed, False otherwise.

    Examples
    --------
    >>> needs_norm = determine_need_normalization([ds1, ds2])
    """
    if len(set([ds.range.shape for ds in ds_list])) > 1:
        return True
    else:
        return False


def normalize_range_wise(ds_list):
    """
    Normalize range dimensions across multiple datasets.

    Interpolates all datasets to have the same range coordinates,
    using the smallest range array as the target.

    Parameters
    ----------
    ds_list : list
        List of xarray Datasets with potentially different range coordinates.

    Returns
    -------
    list
        List of xarray Datasets with normalized range coordinates.

    Warns
    -----
    UserWarning
        When range interpolation is being performed.

    Examples
    --------
    >>> normalized = normalize_range_wise([ds1, ds2])
    """
    warnings.warn("Different range values, interpolating", category=Warning)
    ds_ranges = [ds.range.data for ds in ds_list]
    range_lengths = [len(r) for r in ds_ranges]
    ds_dict = {
        'ranges': ds_ranges,
        'rlength': range_lengths
    }
    ds_df = pd.DataFrame.from_dict(ds_dict)
    min_range = ds_df.sort_values('rlength').iloc[0, 0]
    interped_dss = [ds.interp({'range': min_range}, method='nearest') for ds in ds_list]
    return interped_dss

"""
Radar data loaders using xradar.

This module provides functions for loading radar data from various formats
using the xradar library.
"""

import numpy as np
import pandas as pd
import xarray as xr
import xradar as xd
from datetime import datetime as dt


def open_mf_ceara_dataset(fpaths, elevation):
    """
    Open multiple Ceará radar files and combine into a dataset.

    Loads IRIS format radar files from Ceará, Brazil.

    Parameters
    ----------
    fpaths : str or list
        Single file path or list of file paths to radar files.
    elevation : int
        Sweep number (elevation index) to extract.

    Returns
    -------
    xarray.Dataset
        Combined dataset with all time steps and radar coordinates.

    Notes
    -----
    Hardcoded coordinates for Ceará radar site are applied.

    Examples
    --------
    >>> ds = open_mf_ceara_dataset(["/data/radar1.nc"], elevation=0)
    """
    if isinstance(fpaths, str):
        fpaths = [fpaths]

    engine = 'iris'
    group = f'sweep_{elevation}'

    datasets = []
    for fpath in fpaths:
        try:
            ds = xr.open_dataset(fpath, engine=engine, group=group).rename({'time': 'rtime'})
            datasets.append(ds)
        except Exception as E:
            print(f"{fpath} wasn't loaded, critical error opening : {E}")

    reindexed = []
    for ds in datasets:
        reindexed_ds = xd.util.reindex_angle(
            ds, start_angle=0.5, stop_angle=360, angle_res=1, direction=1, tolerance=2
        )
        reindexed_ds = reindexed_ds.assign_coords({'time': reindexed_ds.rtime.to_numpy().min()})
        reindexed.append(reindexed_ds)

    mf_ds = xr.concat(reindexed, dim='time')
    mf_ds.coords['latitude'] = np.array(-5.069189970000025)
    mf_ds.coords['longitude'] = np.array(-39.267139989999976)

    return mf_ds


def open_mf_guyane_dataset(fpaths, elevation):
    """
    Open multiple French Guiana radar files and combine into a dataset.

    Loads GAMIC format radar files from French Guiana.

    Parameters
    ----------
    fpaths : str or list
        Single file path or list of file paths to radar files.
    elevation : int
        Sweep number (elevation index) to extract.

    Returns
    -------
    xarray.Dataset
        Combined dataset with all time steps.

    Examples
    --------
    >>> ds = open_mf_guyane_dataset(["/data/radar1.h5"], elevation=0)
    """
    if isinstance(fpaths, str):
        fpaths = [fpaths]

    engine = 'gamic'
    group = f'sweep_{elevation}'

    datasets = []
    for fpath in fpaths:
        try:
            ds = xr.open_dataset(fpath, engine=engine, group=group).rename({'time': 'rtime'})
            ds = xd.util.remove_duplicate_rays(ds)
            datasets.append(ds)
        except Exception as E:
            print(f"{fpath} wasn't loaded, critical error opening : {E}")

    reindexed = []
    for ds in datasets:
        reindexed_ds = xd.util.reindex_angle(
            ds, start_angle=0.5, stop_angle=360, angle_res=1, direction=1, tolerance=2
        )
        reindexed_ds = reindexed_ds.assign_coords({'time': reindexed_ds.rtime.to_numpy().min()})
        reindexed.append(reindexed_ds)

    mf_ds = xr.concat(reindexed, dim='time')

    return mf_ds


def open_mf_pernambuco_dataset(fpaths, elevation):
    """
    Open multiple Pernambuco radar files and combine into a dataset.

    Loads Rainbow format radar files from Pernambuco, Brazil. Handles
    files with multiple sweeps at the same time step.

    Parameters
    ----------
    fpaths : str or list
        Single file path or list of file paths to radar files.
    elevation : int
        Sweep number (elevation index) to extract.

    Returns
    -------
    xarray.Dataset
        Combined dataset with all time steps.

    Notes
    -----
    Timestamps are extracted from filenames in format 'YYYYMMDDHHmm'.

    Examples
    --------
    >>> ds = open_mf_pernambuco_dataset(["/data/202301011200.vol"], elevation=0)
    """
    if isinstance(fpaths, str):
        fpaths = [fpaths]

    engine = 'rainbow'
    group = f'sweep_{elevation}'

    fnames = [fpath.split('/')[-1] for fpath in fpaths]
    dates = [dt.strptime(fname[:12], '%Y%m%d%H%M') for fname in fnames]

    recife_df = pd.DataFrame.from_dict({'time': dates, 'paths': fpaths})
    recife_df = recife_df.set_index('time')

    single_times = sorted(set(recife_df.index.tolist()))
    single_step_dfs = [recife_df.loc[time] for time in single_times]

    single_step_dss = []
    for dfs in single_step_dfs:
        if isinstance(dfs.paths, str):
            paths = [dfs.paths]
        else:
            paths = dfs.paths.tolist()
        try:
            ds = xr.open_mfdataset(paths, engine=engine, group=group).rename({'time': 'rtime'})
            reindexed_ds = xd.util.reindex_angle(
                ds, start_angle=0.5, stop_angle=360, angle_res=1, direction=1, tolerance=2
            )
            reindexed_ds = reindexed_ds.assign_coords({'time': reindexed_ds.rtime.to_numpy().min()})
            single_step_dss.append(reindexed_ds)
        except Exception as E:
            print(f"{paths} weren't loaded, critical error opening : {E}")

    mf_ds = xr.concat(single_step_dss, dim='time')

    return mf_ds

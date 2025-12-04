"""
Satellite precipitation product loaders.

This module provides functions for loading satellite precipitation products
including IMERG and GSMaP.
"""

import gzip
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime as dt


def convert_to_dt(imerg_ds):
    """
    Convert time index of IMERG dataset to datetime objects.

    Parameters
    ----------
    imerg_ds : xarray.Dataset or xarray.DataArray
        Loaded dataset of IMERG data.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Dataset with converted time index.

    Examples
    --------
    >>> ds = xr.open_dataset("imerg.nc")
    >>> ds = convert_to_dt(ds)
    """
    new_index = [
        dt.strptime(str(it), '%Y-%m-%d %H:%M:%S')
        for it in imerg_ds.time.data
    ]
    imerg_ds['time'] = new_index
    return imerg_ds


def load_gsmap_to_xr(fpath):
    """
    Load a raw GSMaP file to an xarray DataArray.

    Reads gzipped binary GSMaP data files and converts to xarray format
    with proper coordinates.

    Parameters
    ----------
    fpath : str
        Path to the GSMaP gzip file.

    Returns
    -------
    xarray.DataArray
        DataArray with loaded GSMaP data. Dimensions are ['lat', 'lon', 'time'].
        Values are precipitation rate in mm/hr.

    Notes
    -----
    GSMaP data has 0.1 degree resolution globally (3600 x 1200 grid).
    Time is extracted from the filename.

    Examples
    --------
    >>> da = load_gsmap_to_xr("/data/gsmap.20230101.0000.dat.gz")
    """
    gz = gzip.GzipFile(fpath, 'rb')
    dd = np.frombuffer(gz.read(), dtype=np.float32)
    # 1200 = nb of lat ; 3600 = np of lon; pre=mm/hr
    pre = dd.reshape((1200, 3600, 1))
    nm = fpath.split('/')[-1]
    time = np.array(
        pd.to_datetime(nm.split('.')[1] + (nm.split('.')[2]))
    ).reshape((1))

    # Create lon and lat coordinates
    lon = np.linspace(0.05, 359.95, 3600)  # centroids of lon pixels
    lat = np.linspace(59.95, -59.95, 1200)  # centroids of lat pixels
    lon[lon > 180] = lon[lon > 180] - 360

    gsmap_xr = xr.DataArray(
        data=pre,
        dims=['lat', 'lon', 'time'],
        coords={'lat': lat, 'lon': lon, 'time': time}
    )
    gsmap_xr = gsmap_xr.sortby('lon')
    return gsmap_xr


def open_mf_gsmap(fpaths):
    """
    Open multiple GSMaP files and combine into a single dataset.

    Parameters
    ----------
    fpaths : str or list
        Single file path or list of file paths to GSMaP files.

    Returns
    -------
    xarray.DataArray
        Combined DataArray with all time steps.

    Examples
    --------
    >>> da = open_mf_gsmap(["/data/gsmap1.dat.gz", "/data/gsmap2.dat.gz"])
    """
    if isinstance(fpaths, str):
        fpaths = [fpaths]
    xrs = [load_gsmap_to_xr(fpath) for fpath in fpaths]
    dataset = xr.concat(xrs, dim='time')
    return dataset


def open_mf_imerg(fpaths, field=None):
    """
    Open multiple IMERG HDF5 files and combine into a single dataset.

    Parameters
    ----------
    fpaths : str or list
        Single file path or list of file paths to IMERG files.
    field : str or list, optional
        Variable name(s) to load. If None, all variables are loaded.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Combined dataset. Returns DataArray if field is specified.

    Examples
    --------
    >>> ds = open_mf_imerg(["/data/imerg1.nc5"], field='precipitationCal')
    """
    if isinstance(fpaths, str):
        fpaths = [fpaths]

    if isinstance(field, str):
        field = [field]

    ds = xr.open_mfdataset(fpaths, group='Grid/')
    ds = convert_to_dt(ds)

    if isinstance(field, list):
        return ds[field]
    else:
        return ds


def load_daily_gsmap(date, filepaths_df):
    """
    Load raw GSMaP files for a whole day.

    Parameters
    ----------
    date : str
        Date string in 'YYYY-MM-DD' format.
    filepaths_df : pandas.DataFrame
        DataFrame containing datetimes as index and 'paths' column with
        file paths. Created using get_df_dates_filepaths function.

    Returns
    -------
    xarray.DataArray
        DataArray with loaded GSMaP data for the whole day.

    Examples
    --------
    >>> df = get_df_dates_filepaths("/data/gsmap")
    >>> daily = load_daily_gsmap("2023-01-01", df)
    """
    daily_paths = filepaths_df.loc[date]
    daily_index = daily_paths.index.tolist()
    paths = daily_paths.paths.tolist()
    xrs = [load_gsmap_to_xr(fpath) for fpath in paths]
    dataset = xr.concat(xrs, daily_index)
    dataset = dataset.rename({'concat_dim': 'time'})
    return dataset


def load_daily_imerg(date, filepaths_df, field=None):
    """
    Load raw IMERG files for a whole day.

    Parameters
    ----------
    date : str
        Date string in 'YYYY-MM-DD' format.
    filepaths_df : pandas.DataFrame
        DataFrame containing datetimes as index and 'paths' column with
        file paths. Created using get_df_dates_filepaths function.
    field : str or list, optional
        Variable name(s) to load, such as 'precipitationCal'.
        If None, all variables are loaded.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        DataArray/Dataset with loaded IMERG data for the whole day.

    Examples
    --------
    >>> df = get_df_dates_filepaths("/data/imerg")
    >>> daily = load_daily_imerg("2023-01-01", df, field='precipitationCal')
    """
    if isinstance(field, str):
        field = [field]
    fpaths = filepaths_df.loc[date].paths.tolist()
    ds = xr.open_mfdataset(fpaths, group='Grid/')
    ds = convert_to_dt(ds)

    if isinstance(field, list):
        return ds[field]
    else:
        return ds


def load_v7_by_date(sat_v7, date, field=None):
    daily_spp = xr.open_mfdataset(sat_v7.l2_catalog.loc[str(date), 'paths'].tolist(), decode_timedelta=True)
    if field:
        return daily_spp[field]
    return daily_spp

def load_v6_by_date(sat_v6, date, field=None):
    daily_spp = xr.open_mfdataset(sat_v6.l2_catalog.loc[str(date), 'paths'].tolist(), group='Grid', decode_timedelta=True)
    if field:
        return daily_spp[field]
    return daily_spp

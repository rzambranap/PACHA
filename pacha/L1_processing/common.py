"""
Functions, methods, etc common to L1 processing.
These functions are used by multiple instruments' L1 processing modules.

"""

import numpy as np
import xarray as xr
import pandas as pd

def normalize_geospatial_ds(ds: xr.Dataset):
    # this function normalizes a geospatial dataset or dataarray
    # it ensures it uses np.Datatime for time data
    # it transposes the dataset into time, lat, lon dimension-order
    # it sorts by time then lat, then lon 
    ds = ds.assign_coords(time=("time", [pd.to_datetime(str(i)) for i in ds.time.values]))
    # ensure equal spacing for lat lon coordinates
    lons = ds.lon.values
    lon_mean_spacing = np.diff(lons).mean()
    lats = ds.lat.values
    lat_mean_spacing = np.diff(lats).mean()

    lon_tol = lon_mean_spacing * 0.005
    lat_tol = lat_mean_spacing * 0.005

    decimals = str(lon_tol).split('.')[1]
    # count of zeros before first non-zero digit
    n_zeros = len(decimals) - len(decimals.lstrip('0'))

    # round lon and lat to n_zeros decimal places
    ds = ds.assign_coords(lon=("lon", np.round(ds.lon.values, n_zeros)))
    ds = ds.assign_coords(lat=("lat", np.round(ds.lat.values, n_zeros)))

    # sort by time, lat, lon
    ds = ds.sortby(['time', 'lat', 'lon'])

    #transpose to time, lat, lon
    ds = ds.transpose('time', 'lat', 'lon')

    return ds

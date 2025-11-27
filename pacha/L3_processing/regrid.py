"""
Regridding module for L3 processing.

This module provides functions for regridding radar and other geospatial data
onto regular latitude-longitude grids.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from ..utils.geospatial import get_bbox_from_xarray


def regrid_dataarray(da, resolution=None, lat_grid=None, lon_grid=None):
    """
    Regrid a 2D DataArray to a regular latitude-longitude grid.

    Interpolates scattered or irregular data onto a uniform grid using
    linear interpolation.

    Parameters
    ----------
    da : xarray.DataArray
        Input 2D DataArray with 'lon' and 'lat' coordinates.
    resolution : float, optional
        Grid resolution in degrees. If None and lat_grid/lon_grid are None,
        resolution is computed from the data. Default is None.
    lat_grid : array-like, optional
        Latitude values for the output grid. Default is None.
    lon_grid : array-like, optional
        Longitude values for the output grid. Default is None.

    Returns
    -------
    xarray.DataArray
        Regridded DataArray on the specified regular grid.

    Raises
    ------
    TypeError
        If the input DataArray has more than 2 dimensions.

    Examples
    --------
    >>> da = radar_ds['reflectivity']
    >>> regridded = regrid_dataarray(da, resolution=0.01)
    """
    if len(da.shape) > 2:
        raise TypeError(
            f"Only dataarrays of 2 dims for this function, da has {len(da.shape)} dims"
        )
    if (lat_grid is None) or (lon_grid is None):
        if resolution is None:
            resolution = get_naive_resolution_from_radar(da)
        lat_grid, lon_grid = calc_lat_lon_grids_from_radar(da, resolution)

    lons, lats = da.lon.values.flatten(), da.lat.values.flatten()
    values = da.fillna(0).values.flatten()  # in order to not get a nan-filled da
    ppoints = np.array([lons, lats]).T

    grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)
    grid_z = griddata(ppoints, values, (grid_lon, grid_lat), method='linear')

    nda = xr.DataArray(grid_z, coords=[("lat", lat_grid), ("lon", lon_grid)])
    return nda


def get_naive_resolution_from_radar(ds):
    """
    Estimate grid resolution from radar data based on azimuth spacing.

    Computes the median longitude difference between adjacent azimuths
    as an estimate of appropriate grid resolution.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        Radar data with 'azimuth', 'range', and 'lon' coordinates.

    Returns
    -------
    float
        Estimated grid resolution in degrees.

    Examples
    --------
    >>> resolution = get_naive_resolution_from_radar(radar_da)
    >>> print(f"Suggested resolution: {resolution:.4f} degrees")
    """
    azi1 = ds.azimuth[0].data
    azi2 = ds.azimuth[1].data

    p1 = ds.sel({'azimuth': azi1})
    p2 = ds.sel({'azimuth': azi2})

    diff_coords = np.abs(p1.lon - p2.lon)
    res = diff_coords.median().data
    return res


def calc_lat_lon_grids_from_radar(ds, resolution):
    """
    Calculate latitude and longitude grids for regridding radar data.

    Creates regularly spaced latitude and longitude arrays based on the
    bounding box of the input data and the specified resolution.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        Input data with 'lon' and 'lat' coordinates.
    resolution : float
        Grid resolution in degrees.

    Returns
    -------
    lat_grid : numpy.ndarray
        Array of latitude values for the output grid.
    lon_grid : numpy.ndarray
        Array of longitude values for the output grid.

    Examples
    --------
    >>> lat_grid, lon_grid = calc_lat_lon_grids_from_radar(radar_ds, 0.01)
    """
    bbox = get_bbox_from_xarray(ds)
    lon_grid = np.arange(bbox.min_lon, bbox.max_lon, resolution)
    lat_grid = np.arange(bbox.min_lat, bbox.max_lat, resolution)
    return lat_grid, lon_grid

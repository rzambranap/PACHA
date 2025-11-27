"""
Spatial interpolation module for L3 processing.

This module provides functions for spatial interpolation of scattered point
data (e.g., rain gauge measurements) onto regular grids.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from ..utils.geospatial import get_bbox_from_df


def inverse_distance_weighting(df_data, df_metadata, resolution):
    """
    Perform inverse distance weighting (IDW) interpolation from point data to grid.

    Interpolates scattered point observations (e.g., rain gauge data) onto a
    regular latitude-longitude grid using inverse distance weighting.

    Parameters
    ----------
    df_data : pandas.DataFrame or pandas.Series
        Data values to interpolate. For DataFrame, rows are time steps and
        columns are station IDs.
    df_metadata : pandas.DataFrame
        Metadata containing 'lat' and 'lon' columns with station coordinates.
    resolution : float
        Grid resolution in degrees.

    Returns
    -------
    xarray.DataArray
        Interpolated data on a regular grid. Dimensions are ['time', 'lat', 'lon']
        for time-series input or ['lat', 'lon'] for single-time input.

    Examples
    --------
    >>> gauge_data = pd.DataFrame({'station1': [1.0, 2.0], 'station2': [1.5, 2.5]})
    >>> metadata = pd.DataFrame({'lat': [-5.0, -5.1], 'lon': [-39.0, -39.1]})
    >>> gridded = inverse_distance_weighting(gauge_data, metadata, resolution=0.1)
    """
    # Get bounding box from metadata
    bbox = get_bbox_from_df(df_metadata)
    min_lat, max_lat = bbox.min_lat, bbox.max_lat
    min_lon, max_lon = bbox.min_lon, bbox.max_lon

    # Create grid
    lat_grid = np.arange(min_lat, max_lat, resolution)
    lon_grid = np.arange(min_lon, max_lon, resolution)
    lon, lat = np.meshgrid(lon_grid, lat_grid)

    # Prepare output xarray
    if isinstance(df_data, pd.DataFrame):
        times = df_data.index
        output = xr.DataArray(
            np.zeros((len(times), len(lat_grid), len(lon_grid))),
            coords=[times, lat_grid, lon_grid],
            dims=['time', 'lat', 'lon']
        )
    else:
        output = xr.DataArray(
            np.zeros((len(lat_grid), len(lon_grid))),
            coords=[lat_grid, lon_grid],
            dims=['lat', 'lon']
        )

    # Prepare coordinates and values
    coords = df_metadata[['lat', 'lon']].values
    if isinstance(df_data, pd.DataFrame):
        for time in df_data.index:
            values = df_data.loc[time].values
            output.loc[time] = idw_interpolation(coords, values, lat, lon)
    else:
        values = df_data.values
        output[:] = idw_interpolation(coords, values, lat, lon)

    return output


def idw_interpolation(coords, values, lat, lon, power=2):
    """
    Perform IDW interpolation for a single time step.

    Calculates interpolated values at grid points using inverse distance
    weighting from scattered observation points.

    Parameters
    ----------
    coords : numpy.ndarray
        Array of shape (n_stations, 2) containing [lat, lon] for each station.
    values : numpy.ndarray
        Array of observation values at each station.
    lat : numpy.ndarray
        2D array of latitude values for the output grid.
    lon : numpy.ndarray
        2D array of longitude values for the output grid.
    power : float, optional
        Power parameter for inverse distance weighting. Higher values give
        more weight to nearby points. Default is 2.

    Returns
    -------
    numpy.ndarray
        2D array of interpolated values matching the shape of lat/lon grids.

    Notes
    -----
    The IDW formula is: z = sum(w_i * z_i) / sum(w_i)
    where w_i = 1 / d_i^power and d_i is the distance to observation i.

    Examples
    --------
    >>> coords = np.array([[-5.0, -39.0], [-5.1, -39.1]])
    >>> values = np.array([10.0, 15.0])
    >>> lat, lon = np.meshgrid(np.linspace(-5.2, -4.9, 10),
    ...                        np.linspace(-39.2, -38.9, 10))
    >>> result = idw_interpolation(coords, values, lat, lon)
    """
    tree = cKDTree(coords)
    grid_shape = lat.shape
    lat_lon_pairs = np.vstack([lat.ravel(), lon.ravel()]).T
    distances, indices = tree.query(lat_lon_pairs, k=len(coords))
    weights = 1 / distances**power
    weights /= weights.sum(axis=1)[:, None]
    interpolated_values = np.sum(weights * values[indices], axis=1)
    return interpolated_values.reshape(grid_shape)

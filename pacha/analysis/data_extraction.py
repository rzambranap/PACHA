"""
Data extraction module for analysis.

This module provides utility functions for extracting data values at specific
geographic coordinates from gridded datasets.
"""

import numpy as np


def get_val_coord(lon, lat, scan):
    """
    Extract value from a gridded dataset at specific coordinates.

    Finds the nearest grid point to the specified longitude and latitude
    and returns the data value at that location.

    Parameters
    ----------
    lon : float
        Longitude coordinate of the point of interest.
    lat : float
        Latitude coordinate of the point of interest.
    scan : xarray.Dataset or xarray.DataArray
        Gridded data with 'lat' and 'lon' coordinates (can be 1D or 2D).

    Returns
    -------
    xarray.DataArray
        Time series of values at the specified coordinate.

    Examples
    --------
    >>> radar_data = load_radar_sweep("radar.nc")
    >>> station_value = get_val_coord(-39.5, -5.1, radar_data)
    >>> print(station_value.values)
    """
    if len(scan.lat.shape) > 2:
        abslat = np.abs(scan.lat.mean(dim='time') - lat)
        abslon = np.abs(scan.lon.mean(dim='time') - lon)
    else:
        abslat = np.abs(scan.lat - lat)
        abslon = np.abs(scan.lon - lon)
    c = np.maximum(abslon.values, abslat.values)
    xloc, yloc = np.where(c == np.min(c))
    point_value = scan[:, xloc[0], yloc[0]]
    return point_value


def get_coords_station(gauge_meta, station):
    """
    Get geographic coordinates for a station from metadata.

    Parameters
    ----------
    gauge_meta : pandas.DataFrame
        DataFrame with station metadata, indexed by station ID,
        containing 'lat' and 'lon' columns.
    station : str or int
        Station identifier (must be in the index of gauge_meta).

    Returns
    -------
    lat : float
        Latitude of the station.
    lon : float
        Longitude of the station.

    Examples
    --------
    >>> metadata = pd.DataFrame({'lat': [-5.0], 'lon': [-39.0]}, index=['ST001'])
    >>> lat, lon = get_coords_station(metadata, 'ST001')
    >>> print(f"Station at ({lat}, {lon})")
    """
    st_df = gauge_meta.loc[station, :]
    lat = st_df.lat
    lon = st_df.lon
    return lat, lon

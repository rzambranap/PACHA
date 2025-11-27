"""
Radar L1 processing module.

This module provides functions for Level 1 processing of radar data, including
georeferencing and coordinate transformations for radar sweeps.
"""

import numpy as np
import xradar as xd
import warnings


def calculate_georeferencing(ds):
    """
    Calculate georeferencing coordinates for an xarray dataset.

    Computes geographic coordinates (latitude, longitude, altitude) for radar
    data based on range, azimuth, and elevation information.

    Parameters
    ----------
    ds : xarray.Dataset
        Input xarray dataset containing radar data with 'range', 'azimuth',
        'elevation', 'altitude', 'longitude', and 'latitude' coordinates.

    Returns
    -------
    xarray.Dataset
        Output xarray dataset with georeferencing coordinates added as
        'lon', 'lat', and 'alt' coordinate arrays.

    Examples
    --------
    >>> ds = load_radar_data("radar_file.nc")
    >>> georef_ds = calculate_georeferencing(ds)
    >>> print(georef_ds.lon.shape)  # (n_azimuths, n_ranges)
    """
    mfds = ds.copy()
    ranges = mfds.range.to_numpy()
    azimuths = mfds.azimuth.to_numpy()
    elevations = mfds.elevation.mean().data
    site_altitude = mfds.altitude.to_numpy()

    x_origin = mfds.longitude.data
    y_origin = mfds.latitude.data

    xs = np.empty((len(azimuths), len(ranges)))
    ys = np.empty_like(xs)
    zs = np.empty_like(xs)
    for az in range(0, len(azimuths)):
        for r in range(0, len(ranges)):
            x, y, z = xd.georeference.antenna_to_cartesian(
                ranges[r], azimuths[az], elevations, site_altitude=site_altitude
            )
            xs[az, r] = x
            ys[az, r] = y
            zs[az, r] = z

    lon, lat = cartesian_to_geographic_aeqd(xs, ys, x_origin, y_origin)

    alt = mfds.altitude.data
    alts = zs + alt

    out_ds = mfds.assign_coords(
        {'lon': (['azimuth', 'range'], lon),
         'lat': (['azimuth', 'range'], lat),
         'alt': (['azimuth', 'range'], alts)}
    )

    return out_ds


def cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, R=6370997.0):
    """
    Convert azimuthal equidistant Cartesian coordinates to geographic coordinates.

    Transforms a set of Cartesian/Cartographic coordinates (x, y) to
    geographic coordinate system (lat, lon) using an azimuthal equidistant
    map projection.

    The transformation uses the following equations:

    .. math::

        lat = \\arcsin(\\cos(c) * \\sin(lat_0) +
                       (y * \\sin(c) * \\cos(lat_0) / \\rho))

        lon = lon_0 + \\arctan2(
            x * \\sin(c),
            \\rho * \\cos(lat_0) * \\cos(c) - y * \\sin(lat_0) * \\sin(c))

        \\rho = \\sqrt(x^2 + y^2)

        c = \\rho / R

    Parameters
    ----------
    x : array-like
        Cartesian x coordinates in the same units as R, typically meters.
    y : array-like
        Cartesian y coordinates in the same units as R, typically meters.
    lon_0 : float
        Longitude of the center of the projection, in degrees.
    lat_0 : float
        Latitude of the center of the projection, in degrees.
    R : float, optional
        Earth radius in the same units as x and y. Default is 6370997.0 meters.

    Returns
    -------
    lon : numpy.ndarray
        Longitude of Cartesian coordinates in degrees.
    lat : numpy.ndarray
        Latitude of Cartesian coordinates in degrees.

    References
    ----------
    .. [1] Snyder, J. P. Map Projections--A Working Manual. U. S. Geological
        Survey Professional Paper 1395, 1987, pp. 191-202.

    Examples
    --------
    >>> x = np.array([10000, 20000])  # meters
    >>> y = np.array([5000, 10000])   # meters
    >>> lon, lat = cartesian_to_geographic_aeqd(x, y, -39.0, -5.0)
    """
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))

    lat_0_rad = np.deg2rad(lat_0)
    lon_0_rad = np.deg2rad(lon_0)

    rho = np.sqrt(x * x + y * y)
    c = rho / R

    with warnings.catch_warnings():
        # division by zero may occur here but is properly addressed below so
        # the warnings can be ignored
        warnings.simplefilter("ignore", RuntimeWarning)
        lat_rad = np.arcsin(
            np.cos(c) * np.sin(lat_0_rad) + y * np.sin(c) * np.cos(lat_0_rad) / rho
        )
    lat_deg = np.rad2deg(lat_rad)
    # fix cases where the distance from the center of the projection is zero
    lat_deg[rho == 0] = lat_0

    x1 = x * np.sin(c)
    x2 = rho * np.cos(lat_0_rad) * np.cos(c) - y * np.sin(lat_0_rad) * np.sin(c)
    lon_rad = lon_0_rad + np.arctan2(x1, x2)
    lon_deg = np.rad2deg(lon_rad)
    # Longitudes should be from -180 to 180 degrees
    lon_deg[lon_deg > 180] -= 360.0
    lon_deg[lon_deg < -180] += 360.0

    return lon_deg, lat_deg

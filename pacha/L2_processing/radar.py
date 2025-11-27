"""
Radar L2 processing module.

This module provides functions for Level 2 processing of radar data, including
reflectivity-to-rain rate conversions and precipitation estimation.
"""

import numpy as np
import xarray as xr


def calc_rate_from_refl(ref, a=200, b=1.6, offset=None):
    """
    Calculate rain rate from radar reflectivity using Z-R relationship.

    Uses the Marshall-Palmer or similar Z-R relationship: Z = a * R^b,
    solved for R to convert reflectivity (dBZ) to rain rate (mm/hr).

    Parameters
    ----------
    ref : array-like or xarray.DataArray
        Radar reflectivity values in dBZ.
    a : float, optional
        Z-R relationship coefficient 'a'. Default is 200 (Marshall-Palmer).
    b : float, optional
        Z-R relationship exponent 'b'. Default is 1.6 (Marshall-Palmer).
    offset : float, optional
        Offset to apply to reflectivity before conversion. Default is None.

    Returns
    -------
    array-like or xarray.DataArray
        Rain rate values in mm/hr.

    Notes
    -----
    The conversion formula is: R = ((10^(dBZ/10)) / a)^(1/b)

    Examples
    --------
    >>> ref = np.array([30, 40, 50])  # dBZ
    >>> rate = calc_rate_from_refl(ref)
    >>> print(rate)
    """
    if offset is not None:
        ref = ref + offset
    radar_final = ((10**(ref / 10)) / a) ** (1 / b)
    return radar_final


def calc_refl_from_rate(rate_df, a=200, b=1.6):
    """
    Calculate radar reflectivity from rain rate using Z-R relationship.

    Uses the Marshall-Palmer or similar Z-R relationship: Z = a * R^b,
    to convert rain rate (mm/hr) to reflectivity (dBZ).

    Parameters
    ----------
    rate_df : array-like or xarray.DataArray
        Rain rate values in mm/hr.
    a : float, optional
        Z-R relationship coefficient 'a'. Default is 200 (Marshall-Palmer).
    b : float, optional
        Z-R relationship exponent 'b'. Default is 1.6 (Marshall-Palmer).

    Returns
    -------
    array-like or xarray.DataArray
        Radar reflectivity values in dBZ.

    Notes
    -----
    The conversion formula is: dBZ = 10 * log10(a * R^b)

    Examples
    --------
    >>> rate = np.array([1, 5, 10])  # mm/hr
    >>> refl = calc_refl_from_rate(rate)
    >>> print(refl)
    """
    non_zero = rate_df + 0.0001
    final = 10 * np.log10(a * (non_zero**b))
    return final


def calculate_radar_estimate_averaging_elevations(raw_DSs, ref_var, a=205, b=1.44, offset=5.7):
    """
    Calculate radar rain rate estimate by averaging reflectivities from multiple elevations.

    Converts reflectivity to rain rate for each elevation and then averages
    the results to produce a final precipitation estimate.

    Parameters
    ----------
    raw_DSs : list of xarray.Dataset
        List of xarray Datasets containing reflectivity data for different
        radar elevations.
    ref_var : str
        Name of the reflectivity variable in the Datasets.
    a : float, optional
        Z-R relationship coefficient 'a'. Default is 205.
    b : float, optional
        Z-R relationship exponent 'b'. Default is 1.44.
    offset : float, optional
        Offset value to apply to reflectivity before conversion. Default is 5.7.

    Returns
    -------
    xarray.Dataset
        Dataset containing the radar estimate of rain rate with coordinates
        'time', 'azimuth', 'range', 'lon', and 'lat'.

    Examples
    --------
    >>> ds_elev1 = load_radar_sweep("sweep_1.nc")
    >>> ds_elev2 = load_radar_sweep("sweep_2.nc")
    >>> ds_elev3 = load_radar_sweep("sweep_3.nc")
    >>> rain_rate = calculate_radar_estimate_averaging_elevations(
    ...     [ds_elev1, ds_elev2, ds_elev3], 'DBZ')
    """
    # Extract reflectivity data from raw_DSs
    reflectivities = [i[ref_var] for i in raw_DSs]

    # Extract necessary coordinate data
    azis = reflectivities[0].azimuth.data
    rans = reflectivities[0].range.data
    time = reflectivities[0].time.data
    several_lons = [refl.lon.data for refl in reflectivities]
    several_lats = [refl.lat.data for refl in reflectivities]

    # Convert reflectivity data to numpy arrays
    reflectivities = [i.to_numpy() for i in reflectivities]
    reflectivities = [np.ma.masked_invalid(i) for i in reflectivities]

    # Calculate rain rates from reflectivities
    rates = [calc_rate_from_refl(refl, a=a, b=b, offset=offset) for refl in reflectivities]

    # Average the rain rates from different elevations
    rate_final = np.ma.stack(rates).mean(axis=0)

    # Compute average latitudes and longitudes
    lat_final = np.stack(several_lats).mean(axis=0)
    lon_final = np.stack(several_lons).mean(axis=0)

    # Create a new xarray Dataset for the final radar estimate
    final_ds = xr.Dataset(
        data_vars={'rain_rate': (['time', 'azimuth', 'range'], rate_final)},
        coords={'time': time, 'azimuth': azis, 'range': rans}
    )

    # Assign latitude and longitude coordinates to the final Dataset
    final_ds = final_ds.assign_coords({
        'lon': (['azimuth', 'range'], lon_final),
        'lat': (['azimuth', 'range'], lat_final)
    })

    return final_ds

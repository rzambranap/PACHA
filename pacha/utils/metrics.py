"""
Statistical metrics module.

This module provides functions for calculating various statistical metrics
for precipitation estimation validation including KGE, RMSE, bias, and
contingency scores.
"""

import numpy as np
import pandas as pd
from math import sqrt, log10


def kge(a, b):
    """
    Calculate Kling-Gupta Efficiency (KGE) between two arrays.

    Parameters
    ----------
    a : numpy.ndarray
        Observed values.
    b : numpy.ndarray
        Simulated values.

    Returns
    -------
    float
        Kling-Gupta Efficiency value. Perfect score is 1.

    Notes
    -----
    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    where r is correlation, alpha is variability ratio, beta is bias ratio.

    Examples
    --------
    >>> obs = np.array([1.0, 2.0, 3.0])
    >>> sim = np.array([1.1, 2.1, 2.9])
    >>> print(kge(obs, sim))
    """
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a)
    std_b = np.std(b)
    corr = np.corrcoef(a, b)[0, 1]

    kge_val = 1 - np.sqrt(
        (corr - 1) ** 2 +
        (std_a / mean_a - std_b / mean_b) ** 2 +
        (mean_a / mean_b - 1) ** 2
    )
    return kge_val


def get_val_coord(lon, lat, scan):
    """
    Extract value from gridded data at specified coordinates.

    Parameters
    ----------
    lon : float
        Longitude coordinate.
    lat : float
        Latitude coordinate.
    scan : xarray.Dataset
        2D grid of data with lat/lon coordinates.

    Returns
    -------
    xarray.DataArray
        Value at the specified or nearest grid point.

    Examples
    --------
    >>> val = get_val_coord(-39.5, -5.0, radar_ds)
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
    Get coordinates for a station from metadata.

    Parameters
    ----------
    gauge_meta : pandas.DataFrame
        DataFrame with station metadata including lat/lon.
    station : str or int
        Station identifier.

    Returns
    -------
    lat : float
        Station latitude.
    lon : float
        Station longitude.

    Examples
    --------
    >>> lat, lon = get_coords_station(metadata, 'ST001')
    """
    st_df = gauge_meta.loc[station, :]
    lat = st_df.lat
    lon = st_df.lon
    return lat, lon


def rad_sat_psnrs(rad, sat):
    """
    Calculate time series of correlations between radar and satellite.

    Parameters
    ----------
    rad : xarray.DataArray
        Radar data with time dimension.
    sat : xarray.DataArray
        Satellite data with time dimension.

    Returns
    -------
    pandas.DataFrame
        DataFrame with time index and 'psnrs' column containing correlations.

    Examples
    --------
    >>> scores = rad_sat_psnrs(radar_da, satellite_da)
    """
    time = rad.time
    psnrs = []
    for step in time:
        single_sat = sat.sel(time=step, method='nearest')
        single_sat = single_sat.data
        single_rad = rad.sel(time=step, method='nearest').data
        rad_mask = np.ma.masked_invalid(single_rad)
        gm_mask = np.ma.masked_array(single_sat, mask=rad_mask.mask)
        correlation_matrix = np.ma.corrcoef(
            rad_mask.flatten(), gm_mask.flatten()
        )
        correlation_xy = correlation_matrix[0, 1]
        psnrs.append(correlation_xy)

    score_df = pd.DataFrame.from_dict({'time': time, 'psnrs': psnrs})
    score_df = score_df.set_index('time')
    return score_df


def corr2_coeff(A, B):
    """
    Calculate row-wise correlation coefficient between two 2D arrays.

    Parameters
    ----------
    A : numpy.ndarray
        First 2D array.
    B : numpy.ndarray
        Second 2D array.

    Returns
    -------
    numpy.ndarray
        Correlation coefficient matrix.

    Examples
    --------
    >>> corr = corr2_coeff(array1, array2)
    """
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    ssA = (A_mA ** 2).sum(1)
    ssB = (B_mB ** 2).sum(1)

    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def pearsonr_2D(x, y):
    """
    Compute Pearson correlation where x is 1D and y is 2D.

    Parameters
    ----------
    x : numpy.ndarray
        1D array.
    y : numpy.ndarray
        2D array.

    Returns
    -------
    numpy.ndarray
        Array of correlation coefficients.

    Examples
    --------
    >>> rho = pearsonr_2D(x_array, y_matrix)
    """
    upper = np.sum(
        (x - np.mean(x)) * (y - np.mean(y, axis=1)[:, None]),
        axis=1
    )
    lower = np.sqrt(
        np.sum(np.power(x - np.mean(x), 2)) *
        np.sum(np.power(y - np.mean(y, axis=1)[:, None], 2), axis=1)
    )
    rho = upper / lower
    return rho


def psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio between two images.

    Parameters
    ----------
    img1 : numpy.ndarray
        First image array.
    img2 : numpy.ndarray
        Second image array.

    Returns
    -------
    float
        PSNR value in dB.

    Examples
    --------
    >>> psnr_val = psnr(observed, simulated)
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = np.array([img1.max(), img2.max()])
    PIXEL_MAX = PIXEL_MAX.max()
    return 20 * log10(PIXEL_MAX / sqrt(mse))


def contingency_dataframe_timewise(observed, simulated, threshold=0, prefix=None):
    """
    Calculate time-wise contingency scores for gridded data.

    Parameters
    ----------
    observed : xarray.DataArray
        Observed precipitation data.
    simulated : xarray.DataArray
        Simulated/estimated precipitation data.
    threshold : float, optional
        Precipitation threshold for detection. Default is 0.
    prefix : str, optional
        Prefix for column names. Default is None.

    Returns
    -------
    pandas.DataFrame
        DataFrame with POD, FAR, F_SCORE, and contingency counts.

    Examples
    --------
    >>> cont_df = contingency_dataframe_timewise(obs, sim, threshold=0.5)
    """
    observed_over = (observed > threshold).sum(dim=['lat', 'lon'])

    true_positive = (
        (observed > threshold) * (simulated > threshold)
    ).sum(dim=['lat', 'lon'])
    false_positive = (
        (observed < threshold) * (simulated > threshold)
    ).sum(dim=['lat', 'lon'])
    false_negative = (
        (observed > threshold) * (simulated < threshold)
    ).sum(dim=['lat', 'lon'])
    true_negative = (
        (observed < threshold) * (simulated < threshold)
    ).sum(dim=['lat', 'lon'])

    POD = (true_positive / observed_over).to_dataframe()
    FAR = (false_positive / (false_positive + true_positive)).to_dataframe()
    F_SCORE = (false_positive / (false_positive + true_negative)).to_dataframe()

    cont_df = pd.concat([
        POD, FAR, F_SCORE,
        true_positive.to_dataframe(),
        false_negative.to_dataframe(),
        false_positive.to_dataframe(),
        true_negative.to_dataframe()
    ], axis=1)
    cont_df.columns = [
        'POD', 'FAR', 'F_SCORE', 'true_positive',
        'false_negative', 'false_positive', 'true_negative'
    ]
    if prefix is not None:
        cont_df.columns = [prefix + '_' + i for i in cont_df.columns]

    return cont_df


def count_pixels_over_threshold(datasets, threshold, rename_cols=None):
    """
    Count pixels exceeding a threshold for multiple datasets.

    Parameters
    ----------
    datasets : list or xarray.Dataset
        Dataset(s) to analyze.
    threshold : float
        Threshold value.
    rename_cols : list, optional
        New column names for the output.

    Returns
    -------
    pandas.DataFrame
        DataFrame with pixel counts over threshold.

    Examples
    --------
    >>> counts = count_pixels_over_threshold([radar, satellite], 0.5)
    """
    if type(datasets) != list:
        datasets = [datasets]
    counts = []
    for ds in datasets:
        count = (ds > threshold).sum(dim=['lat', 'lon']).to_dataframe()
        counts.append(count)
    if len(counts) > 1:
        count_df = pd.concat(counts, axis=1)
    else:
        count_df = counts[0]

    count_df = count_df.dropna()

    if rename_cols is None:
        return count_df
    else:
        try:
            count_df.columns = rename_cols
            return count_df
        except Exception as E:
            print('rename_cols wrong format')
            print(E)
            return count_df


def average_over_domain(datasets, threshold=None, rename_cols=None):
    """
    Calculate domain-average precipitation for multiple datasets.

    Parameters
    ----------
    datasets : list or xarray.Dataset
        Dataset(s) to average.
    threshold : float, optional
        Only include values above threshold. Default is None (include all).
    rename_cols : list, optional
        New column names for the output.

    Returns
    -------
    pandas.DataFrame
        DataFrame with domain-averaged values.

    Examples
    --------
    >>> avg = average_over_domain([radar, satellite], threshold=0.1)
    """
    if type(datasets) != list:
        datasets = [datasets]
    averages = []
    for ds in datasets:
        if threshold is not None:
            average = (ds.where(ds > threshold)).mean(dim=['lat', 'lon']).to_dataframe()
        else:
            average = ds.mean(dim=['lat', 'lon']).to_dataframe()
        averages.append(average)

    if len(averages) > 1:
        average_df = pd.concat(averages, axis=1)
    else:
        average_df = averages[0]

    average_df = average_df.dropna()

    if rename_cols is None:
        return average_df
    else:
        try:
            average_df.columns = rename_cols
            return average_df
        except Exception as E:
            print('rename_cols wrong format')
            print(E)
            return average_df


def calc_rel_bias(ds, char_pcp='radar'):
    """
    Calculate relative bias between two variables in a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with exactly two variables.
    char_pcp : str, optional
        Substring identifying the reference variable. Default is 'radar'.

    Returns
    -------
    dict
        Dictionary with 'relative_bias' key.

    Raises
    ------
    ValueError
        If dataset doesn't have exactly two variables.

    Examples
    --------
    >>> bias = calc_rel_bias(ds, char_pcp='radar')
    """
    fields = list(ds.keys())
    if len(fields) != 2:
        raise ValueError(
            f'Dataset must have two variables, has {len(fields)}'
        )

    idx_pcp = [i for i in range(len(fields)) if char_pcp in fields[i]][0]
    field1 = fields.pop(idx_pcp)
    field2 = fields[0]

    f1_full = ds[field1].sum()
    f2_full = ds[field2].sum()

    rel_bias = {
        'relative_bias': np.round(
            (((f2_full - f1_full) / f1_full)).values,
            decimals=4
        )
    }
    return rel_bias


def calculate_relative_bias(observed, simulated):
    """
    Calculate relative bias between observed and simulated values.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed values.
    simulated : numpy.ndarray
        Simulated values.

    Returns
    -------
    float
        Relative bias value.

    Examples
    --------
    >>> bias = calculate_relative_bias(obs, sim)
    """
    sum_obs = np.sum(observed)
    sum_sim = np.sum(simulated)
    bias = np.round((sum_sim - sum_obs) / sum_obs, decimals=4)
    return bias


def calc_pixel_wise_contingency(ds, char_pcp='radar', threshold=2):
    """
    Calculate pixel-wise POD and FAR for given thresholds.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with exactly two variables.
    char_pcp : str, optional
        Substring identifying the reference variable. Default is 'radar'.
    threshold : float or list, optional
        Threshold value(s). Default is 2.

    Returns
    -------
    pandas.DataFrame
        DataFrame with POD and FAR indexed by threshold.

    Raises
    ------
    ValueError
        If dataset doesn't have exactly two variables.

    Examples
    --------
    >>> cont = calc_pixel_wise_contingency(ds, threshold=[0.5, 1.0, 2.0])
    """
    fields = list(ds.keys())
    if len(fields) != 2:
        raise ValueError(
            f'Dataset must have two variables, has {len(fields)}'
        )

    idx_pcp = [i for i in range(len(fields)) if char_pcp in fields[i]][0]
    field1 = fields.pop(idx_pcp)
    field2 = fields[0]

    ok = False
    if isinstance(threshold, list):
        ok = True
    if isinstance(threshold, np.ndarray):
        ok = True
    if not ok:
        threshold = [threshold]

    pods = []
    fars = []
    for t in threshold:
        mask_pod = ds[field1] > t
        mask_far = ds[field1] < t

        detect_sat = (ds[field2].where(mask_pod) > t).sum()
        pod = detect_sat / mask_pod.sum()

        false_sat = (ds[field2].where(mask_far) > t).sum()
        far = 1 - false_sat / mask_far.sum()

        fars.append(far.values)
        pods.append(pod.values)

    out_dict = {
        'threshold': threshold,
        'POD': np.array(pods),
        'FAR': np.array(fars)
    }

    df = pd.DataFrame.from_dict(out_dict)
    df = df.set_index('threshold')
    return df


def calc_area_wise_contingency(ds, char_pcp='radar', threshold=2):
    """
    Calculate area-averaged POD and FAR for given thresholds.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with exactly two variables.
    char_pcp : str, optional
        Substring identifying the reference variable. Default is 'radar'.
    threshold : float or list, optional
        Threshold value(s). Default is 2.

    Returns
    -------
    pandas.DataFrame
        DataFrame with POD and FAR indexed by threshold.

    Raises
    ------
    ValueError
        If dataset doesn't have exactly two variables.

    Examples
    --------
    >>> cont = calc_area_wise_contingency(ds, threshold=[0.5, 1.0, 2.0])
    """
    ds = ds.mean(dim=['lat', 'lon'])
    fields = list(ds.keys())
    if len(fields) != 2:
        raise ValueError(
            f'Dataset must have two variables, has {len(fields)}'
        )

    idx_pcp = [i for i in range(len(fields)) if char_pcp in fields[i]][0]
    field1 = fields.pop(idx_pcp)
    field2 = fields[0]

    ok = False
    if isinstance(threshold, list):
        ok = True
    if isinstance(threshold, np.ndarray):
        ok = True
    if not ok:
        threshold = [threshold]

    pods = []
    fars = []
    for t in threshold:
        mask_pod = ds[field1] > t
        mask_far = ds[field1] < t

        detect_sat = (ds[field2].where(mask_pod) > t).sum()
        pod = detect_sat / mask_pod.sum()

        false_sat = (ds[field2].where(mask_far) > t).sum()
        far = false_sat / mask_far.sum()

        fars.append(far.values)
        pods.append(pod.values)

    out_dict = {
        'threshold': threshold,
        'POD': np.array(pods),
        'FAR': np.array(fars)
    }

    df = pd.DataFrame.from_dict(out_dict)
    df = df.set_index('threshold')
    return df


def calc_corr_count_above_threshold(ds, threshold=0):
    """
    Calculate correlation of rain support area between two datasets.

    Parameters
    ----------
    ds : xarray.Dataset
        Two-variable dataset.
    threshold : float or array-like, optional
        Threshold value(s). Default is 0.

    Returns
    -------
    float or list
        Correlation coefficient(s).

    Examples
    --------
    >>> corr = calc_corr_count_above_threshold(ds, threshold=1.0)
    """
    if isinstance(threshold, (list, np.ndarray)):
        corrs = []
        for t in threshold:
            corr = calc_corr_count_above_threshold(ds, t)
            corrs.append(corr)
        return corrs
    corr = ds.where(ds > threshold).sum(dim=['lat', 'lon']).to_dataframe().corr().iloc[0, 1]
    corr = np.round(corr, decimals=3)
    return corr


def kge_ds(ds, char_pcp='radar', **kwargs):
    """
    Calculate KGE for a two-variable dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with exactly two variables.
    char_pcp : str, optional
        Substring identifying the reference variable. Default is 'radar'.

    Returns
    -------
    dict
        Dictionary with 'kge' key.

    Raises
    ------
    ValueError
        If dataset doesn't have exactly two variables.

    Examples
    --------
    >>> result = kge_ds(ds)
    >>> print(result['kge'])
    """
    fields = list(ds.keys())
    if len(fields) != 2:
        raise ValueError(
            f'Dataset must have two variables, has {len(fields)}'
        )

    idx_pcp = [i for i in range(len(fields)) if char_pcp in fields[i]][0]
    field1 = fields.pop(idx_pcp)
    field2 = fields[0]
    y = ds.to_dataframe()[field2].to_numpy()
    x = ds.to_dataframe()[field1].to_numpy()
    d_out = {'kge': np.round(kge(y, x), decimals=3)}
    return d_out


def calc_contingency(ds, mode='pixel', char_pcp='radar', threshold=2):
    """
    Calculate contingency scores (POD, FAR) for a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with exactly two variables.
    mode : str, optional
        'pixel' or 'area' averaging. Default is 'pixel'.
    char_pcp : str, optional
        Substring identifying the reference variable. Default is 'radar'.
    threshold : float or list, optional
        Threshold value(s). Default is 2.

    Returns
    -------
    pandas.DataFrame
        DataFrame with POD and FAR.

    Raises
    ------
    ValueError
        If dataset doesn't have exactly two variables.

    Examples
    --------
    >>> cont = calc_contingency(ds, mode='pixel', threshold=[0.5, 1.0])
    """
    if mode == 'area':
        ds = ds.mean(dim=['lat', 'lon'])

    fields = list(ds.keys())
    if len(fields) != 2:
        raise ValueError(
            f'Dataset must have two variables, has {len(fields)}'
        )

    idx_pcp = [i for i in range(len(fields)) if char_pcp in fields[i]][0]
    field1 = fields.pop(idx_pcp)
    field2 = fields[0]

    ok = False
    if isinstance(threshold, list):
        ok = True
    if isinstance(threshold, np.ndarray):
        ok = True
    if not ok:
        threshold = [threshold]

    pods = []
    fars = []
    for t in threshold:
        mask_pod = ds[field1] > t
        mask_far = ds[field1] < t

        detect_sat = (ds[field2].where(mask_pod) > t).sum()
        pod = detect_sat / mask_pod.sum()

        false_sat = (ds[field2].where(mask_far) > t).sum()
        far = false_sat / mask_far.sum()

        fars.append(far.values)
        pods.append(pod.values)

    out_dict = {
        'threshold': threshold,
        'POD_' + mode: np.array(pods),
        'FAR_' + mode: np.array(fars)
    }

    df = pd.DataFrame.from_dict(out_dict)
    df = df.set_index('threshold')
    return df

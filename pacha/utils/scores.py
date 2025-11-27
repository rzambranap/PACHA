"""
Scoring functions module.

This module provides high-level scoring functions for comparing precipitation
datasets using various statistical metrics.
"""

import numpy as np
import pandas as pd
import xarray as xr


def calculate_correlation(a, b, skipna=False):
    """
    Calculate Pearson correlation coefficient between two arrays.

    Parameters
    ----------
    a : numpy.ndarray
        First array.
    b : numpy.ndarray
        Second array.
    skipna : bool, optional
        If True, ignore NaN values. Default is False.

    Returns
    -------
    dict
        Dictionary with 'correlation' key.

    Raises
    ------
    ValueError
        If arrays have different shapes when skipna=True.

    Examples
    --------
    >>> corr = calculate_correlation(obs, sim, skipna=True)
    """
    if skipna:
        if a.shape != b.shape:
            raise ValueError(f"a and b of different shapes: {a.shape}, {b.shape}")
        df = pd.DataFrame.from_dict({'a': a, 'b': b}).dropna()
        return calculate_correlation(df.a.to_numpy(), df.b.to_numpy(), skipna=False)
    return {'correlation': np.corrcoef(a, b)[0, 1]}


def calculate_rmse(observed, simulated, skipna=False):
    """
    Calculate Root Mean Square Error between observed and simulated values.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed values.
    simulated : numpy.ndarray
        Simulated values.
    skipna : bool, optional
        If True, ignore NaN values. Default is False.

    Returns
    -------
    dict
        Dictionary with 'rmse' key.

    Raises
    ------
    ValueError
        If arrays have different shapes when skipna=True.

    Examples
    --------
    >>> rmse = calculate_rmse(obs, sim)
    """
    if skipna:
        if observed.shape != simulated.shape:
            raise ValueError(f"a and b of different shapes: {observed.shape}, {simulated.shape}")
        df = pd.DataFrame.from_dict({'observed': observed, 'simulated': simulated}).dropna()
        return calculate_rmse(df.observed.to_numpy(), df.simulated.to_numpy())
    rmse = np.sqrt(np.mean((observed - simulated) ** 2))
    return {'rmse': rmse}


def calculate_relative_bias(observed, simulated, skipna=False):
    """
    Calculate relative bias between observed and simulated values.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed values.
    simulated : numpy.ndarray
        Simulated values.
    skipna : bool, optional
        If True, ignore NaN values. Default is False.

    Returns
    -------
    dict
        Dictionary with 'relative_bias' key.

    Raises
    ------
    ValueError
        If arrays have different shapes when skipna=True.

    Examples
    --------
    >>> bias = calculate_relative_bias(obs, sim)
    """
    if skipna:
        if observed.shape != simulated.shape:
            raise ValueError(f"a and b of different shapes: {observed.shape}, {simulated.shape}")
        df = pd.DataFrame.from_dict({'observed': observed, 'simulated': simulated}).dropna()
        return calculate_relative_bias(df.observed.to_numpy(), df.simulated.to_numpy())
    sum_obs = np.sum(observed)
    sum_sim = np.sum(simulated)
    bias = (sum_sim - sum_obs) / sum_obs
    return {'relative_bias': bias}


def calculate_kge(observed, simulated, skipna=False):
    """
    Calculate Kling-Gupta Efficiency (KGE) between observed and simulated values.

    Parameters
    ----------
    observed : numpy.ndarray
        Observed values.
    simulated : numpy.ndarray
        Simulated values.
    skipna : bool, optional
        If True, ignore NaN values. Default is False.

    Returns
    -------
    dict
        Dictionary with 'kge' key.

    Raises
    ------
    ValueError
        If arrays have different shapes when skipna=True.

    Examples
    --------
    >>> kge = calculate_kge(obs, sim)
    """
    if skipna:
        if observed.shape != simulated.shape:
            raise ValueError(f"a and b of different shapes: {observed.shape}, {simulated.shape}")
        df = pd.DataFrame.from_dict({'observed': observed, 'simulated': simulated}).dropna()
        return calculate_kge(df.observed.to_numpy(), df.simulated.to_numpy())

    a = observed
    b = simulated
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a)
    std_b = np.std(b)
    corr = np.corrcoef(a, b)[0, 1]

    kge = 1 - np.sqrt(
        (corr - 1) ** 2 +
        (std_a / mean_a - std_b / mean_b) ** 2 +
        (mean_a / mean_b - 1) ** 2
    )
    return {'kge': kge}


def calculate_contingency_scores(obs, sim, thresholds,
                                 detection_threshold=0.2,
                                 export_confusion_matrix=False,
                                 threshold_is_global=False,
                                 skipna=False):
    """
    Calculate contingency scores (POD, POND, FAR) for precipitation detection.

    Parameters
    ----------
    obs : numpy.ndarray
        Observed values.
    sim : numpy.ndarray
        Simulated values.
    thresholds : float or list
        Threshold value(s) for observation classification.
    detection_threshold : float, optional
        Threshold for simulation detection. Default is 0.2.
    export_confusion_matrix : bool, optional
        If True, also return confusion matrices. Default is False.
    threshold_is_global : bool, optional
        If True, use same threshold for obs and sim. Default is False.
    skipna : bool, optional
        If True, ignore NaN values. Default is False.

    Returns
    -------
    dict or list
        Score dictionary (single threshold) or list of score dicts.
    dict, optional
        Confusion matrices (if export_confusion_matrix=True).

    Raises
    ------
    ValueError
        If arrays have different shapes when skipna=True.

    Examples
    --------
    >>> scores = calculate_contingency_scores(obs, sim, [0.5, 1.0, 2.0])
    """
    args = [thresholds]
    kwargs = {
        'detection_threshold': detection_threshold,
        'export_confusion_matrix': export_confusion_matrix,
        'threshold_is_global': threshold_is_global,
        'skipna': skipna
    }
    if skipna:
        if obs.shape != sim.shape:
            raise ValueError(f"a and b of different shapes: {obs.shape}, {sim.shape}")
        df = pd.DataFrame.from_dict({'observed': obs, 'simulated': sim}).dropna()
        kwargs['skipna'] = False
        return calculate_contingency_scores(
            df.observed.to_numpy(), df.simulated.to_numpy(), *args, **kwargs
        )

    if not isinstance(thresholds, (list, np.ndarray)):
        thresholds = [thresholds]

    results = []
    confusion_matrices = {}

    for threshold in thresholds:
        if threshold_is_global:
            detection_threshold = threshold
        obs_true = obs > threshold
        sim_true = sim > detection_threshold

        obs_neg = obs < threshold
        sim_neg = sim < detection_threshold

        true_positives = np.sum(obs_true * sim_true)
        true_negatives = np.sum(obs_neg * sim_neg)

        false_positives = np.sum(sim_true * obs_neg)
        false_negatives = np.sum(sim_neg * obs_true)

        observed_true_col = [true_positives, false_negatives]
        observed_neg_col = [true_negatives, false_positives]

        cmatrix_dict = {
            'observed_true': observed_true_col,
            'observed_false': observed_neg_col
        }

        index_col = ['predicted_true', 'predicted_false']

        confusion_matrix = pd.DataFrame.from_dict(cmatrix_dict)
        confusion_matrix.index = index_col

        pod = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else np.nan
        pond = true_negatives / (false_negatives + true_negatives) if (false_negatives + true_negatives) > 0 else np.nan
        far = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else np.nan

        scores_dict = {
            'threshold': threshold,
            'pod': pod,
            'pond': pond,
            'far': far
        }

        results.append(scores_dict)

        if export_confusion_matrix:
            confusion_matrices[threshold] = confusion_matrix

    if export_confusion_matrix:
        return results, confusion_matrices
    if len(thresholds) == 1:
        return results[0]

    return results


def check_only_n_variables(a, n=2):
    """
    Check if dataset has exactly n variables.

    Parameters
    ----------
    a : xarray.Dataset or pandas.DataFrame
        Data to check.
    n : int, optional
        Expected number of variables. Default is 2.

    Returns
    -------
    bool
        True if check passes.

    Raises
    ------
    TypeError
        If a is not xarray.Dataset or pandas.DataFrame.
    ValueError
        If number of variables doesn't match n.

    Examples
    --------
    >>> check_only_n_variables(ds, n=2)
    """
    if isinstance(a, xr.Dataset):
        nvars = len(list(a.data_vars))
    elif isinstance(a, pd.DataFrame):
        nvars = len(a.columns)
    else:
        raise TypeError(f"a is {type(a)}, expecting xr.Dataset or pd.DataFrame")
    if nvars != n:
        raise ValueError(f"a has {nvars} variables, we were expecting {n}")
    return True


def apply_function_to_multi_input_dataset(input_data, function, reference_var, *args, **kwargs):
    """
    Apply a scoring function to all non-reference variables in a dataset.

    Parameters
    ----------
    input_data : xarray.Dataset or pandas.DataFrame
        Input data.
    function : callable
        Scoring function to apply.
    reference_var : str
        Name of the reference variable.
    *args, **kwargs
        Additional arguments passed to function.

    Returns
    -------
    dict
        Dictionary mapping variable names to score results.

    Raises
    ------
    TypeError
        If input type is not supported.
    ValueError
        If reference variable is not in dataset.

    Examples
    --------
    >>> scores = apply_function_to_multi_input_dataset(ds, calculate_kge, 'radar')
    """
    if isinstance(input_data, xr.Dataset):
        dvars = list(input_data.data_vars)
    elif isinstance(input_data, pd.DataFrame):
        dvars = input_data.columns
    else:
        raise TypeError(f'Type {type(input_data)} is still unsupported')

    if reference_var not in dvars:
        raise ValueError(
            f"Reference variable {reference_var} not in dataset. Variables: {dvars}"
        )

    dvars_to_score = [var for var in dvars if var != reference_var]
    pairings = [[reference_var, dvar_to_score] for dvar_to_score in dvars_to_score]

    dvars_scores = {scored_var: None for scored_var in dvars_to_score}
    for pair, dvar_to_score in zip(pairings, dvars_to_score):
        score_result = apply_function_to_dual_input(
            input_data[pair], function, reference_var, *args, **kwargs
        )
        dvars_scores[dvar_to_score] = score_result
    return dvars_scores


def get_remaining_field(dataset, known_field):
    """
    Get the remaining field from a two-variable dataset.

    Parameters
    ----------
    dataset : xarray.Dataset or pandas.DataFrame
        Dataset with exactly two variables.
    known_field : str
        Name of one of the fields.

    Returns
    -------
    str
        Name of the other field.

    Raises
    ------
    ValueError
        If dataset type is not supported.

    Examples
    --------
    >>> other = get_remaining_field(ds, 'radar')
    """
    check_only_n_variables(dataset, 2)
    if isinstance(dataset, pd.DataFrame):
        all_columns = dataset.columns
    elif isinstance(dataset, xr.Dataset):
        all_columns = dataset.data_vars
    else:
        raise ValueError("Only pandas DataFrame or xarray Dataset supported.")

    return [col for col in all_columns if col != known_field][0]


def apply_function_to_dual_input(input_data, function, reference_variable, *args, **kwargs):
    """
    Apply a scoring function to a two-variable dataset.

    Parameters
    ----------
    input_data : xarray.Dataset or pandas.DataFrame
        Dataset with exactly two variables.
    function : callable
        Scoring function to apply.
    reference_variable : str
        Name of the reference variable.
    *args, **kwargs
        Additional arguments passed to function.

    Returns
    -------
    dict
        Score result from function.

    Examples
    --------
    >>> result = apply_function_to_dual_input(ds, calculate_kge, 'radar')
    """
    check_only_n_variables(input_data, 2)

    if isinstance(input_data, xr.Dataset):
        dvars = list(input_data.data_vars)
        input_data = input_data.to_dataframe().loc[:, dvars]

    observed = input_data[reference_variable]
    simulated = input_data[get_remaining_field(input_data, reference_variable)]

    result = function(observed, simulated, *args, **kwargs)
    return result


def score_rbias(ds, reference_var, **kwargs):
    """
    Calculate relative bias score for a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to score.
    reference_var : str
        Reference variable name.

    Returns
    -------
    dict
        Relative bias scores for each non-reference variable.
    """
    score = apply_function_to_multi_input_dataset(
        ds, calculate_relative_bias, reference_var, **kwargs
    )
    return score


def score_rmse(ds, reference_var, **kwargs):
    """
    Calculate RMSE score for a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to score.
    reference_var : str
        Reference variable name.

    Returns
    -------
    dict
        RMSE scores for each non-reference variable.
    """
    score = apply_function_to_multi_input_dataset(
        ds, calculate_rmse, reference_var, **kwargs
    )
    return score


def score_kge(ds, reference_var, **kwargs):
    """
    Calculate KGE score for a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to score.
    reference_var : str
        Reference variable name.

    Returns
    -------
    dict
        KGE scores for each non-reference variable.
    """
    score = apply_function_to_multi_input_dataset(
        ds, calculate_kge, reference_var, **kwargs
    )
    return score


def score_contingency(ds, reference_var, threshold, **kwargs):
    """
    Calculate contingency scores for a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to score.
    reference_var : str
        Reference variable name.
    threshold : float or list
        Threshold value(s).

    Returns
    -------
    dict
        Contingency scores for each non-reference variable.
    """
    score = apply_function_to_multi_input_dataset(
        ds, calculate_contingency_scores, reference_var, threshold, **kwargs
    )
    return score


def score_coeff_var_corr(ds, reference_var, **kwargs):
    """
    Calculate coefficient of variation correlation.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to score.
    reference_var : str
        Reference variable name.

    Returns
    -------
    dict
        CV correlation for each non-reference variable.

    Raises
    ------
    TypeError
        If ds is not xarray.Dataset.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Only xr.Dataset supported, got {type(ds)}")
    cv_ds = calculate_CV_for_ds(ds)
    correls = apply_function_to_multi_input_dataset(
        cv_ds, calculate_correlation, 'rain_rate', skipna=True
    )
    result = change_deepest_keys(correls, {'correlation': 'coeff_var_correlation'})
    return result


def score_support_size_corr(ds, reference_var, threshold, **kwargs):
    """
    Calculate support size (rain area) correlation.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to score.
    reference_var : str
        Reference variable name.
    threshold : float
        Threshold for rain detection.

    Returns
    -------
    dict
        Support size correlation for each non-reference variable.

    Raises
    ------
    TypeError
        If ds is not xarray.Dataset.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Only xr.Dataset supported, got {type(ds)}")
    ds_support_size = (ds > threshold).sum(dim=['lat', 'lon'])
    correls = apply_function_to_multi_input_dataset(
        ds_support_size, calculate_correlation, 'rain_rate', skipna=True
    )
    result = change_deepest_keys(correls, {'correlation': 'support_size_correlation'})
    return result


def score_spatial_avg_corr(ds, reference_var, **kwargs):
    """
    Calculate spatial average correlation.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to score.
    reference_var : str
        Reference variable name.

    Returns
    -------
    dict
        Spatial average correlation for each non-reference variable.

    Raises
    ------
    TypeError
        If ds is not xarray.Dataset.
    """
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Only xr.Dataset supported, got {type(ds)}")
    ds_avg = ds.mean(dim=['lat', 'lon'])
    correls = apply_function_to_multi_input_dataset(
        ds_avg, calculate_correlation, 'rain_rate', skipna=True
    )
    result = change_deepest_keys(correls, {'correlation': 'spatial_avg_correlation'})
    return result


def calculate_CV_for_ds(ds, dim=None):
    """
    Calculate coefficient of variation for a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to calculate CV for.
    dim : list, optional
        Dimensions to calculate over. Default is ['lat', 'lon'].

    Returns
    -------
    xarray.Dataset
        Dataset with CV values.
    """
    if dim is None:
        dim = ['lat', 'lon']
    ds_mean = ds.mean(dim=dim)
    ds_std = ds.std(dim=dim)
    ds_cv = ds_std / ds_mean
    return ds_cv


def change_deepest_keys(d, key_mapping):
    """
    Replace innermost dictionary keys using a mapping.

    Parameters
    ----------
    d : dict
        Nested dictionary.
    key_mapping : dict
        Mapping of old keys to new keys.

    Returns
    -------
    dict
        Dictionary with replaced keys.
    """
    new_dict = {}
    for outer_key, inner_dict in d.items():
        new_inner_dict = {}
        for inner_key, value in inner_dict.items():
            new_key = key_mapping.get(inner_key, inner_key)
            new_inner_dict[new_key] = value
        new_dict[outer_key] = new_inner_dict
    return new_dict


def calculate_r2score(a, b, skipna=False):
    """
    Calculate R-squared (coefficient of determination).

    Parameters
    ----------
    a : numpy.ndarray
        Observed values.
    b : numpy.ndarray
        Simulated values.
    skipna : bool, optional
        If True, ignore NaN values. Default is False.

    Returns
    -------
    dict
        Dictionary with 'r2' key.

    Examples
    --------
    >>> r2 = calculate_r2score(obs, sim)
    """
    if skipna:
        if a.shape != b.shape:
            raise ValueError(f"a and b of different shapes: {a.shape}, {b.shape}")
        df = pd.DataFrame.from_dict({'a': a, 'b': b}).dropna()
        return calculate_r2score(df.a.to_numpy(), df.b.to_numpy(), skipna=False)
    ss_res = np.sum((a - b) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return {'r2': r2}


def score_r2(ds, reference_var, **kwargs):
    """
    Calculate R-squared score for a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to score.
    reference_var : str
        Reference variable name.

    Returns
    -------
    dict
        R-squared scores for each non-reference variable.
    """
    score = apply_function_to_multi_input_dataset(
        ds, calculate_r2score, reference_var, **kwargs
    )
    return score


def score_correlation(ds, reference_var, **kwargs):
    """
    Calculate correlation score for a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to score.
    reference_var : str
        Reference variable name.

    Returns
    -------
    dict
        Correlation scores for each non-reference variable.
    """
    score = apply_function_to_multi_input_dataset(
        ds, calculate_correlation, reference_var, **kwargs
    )
    return score

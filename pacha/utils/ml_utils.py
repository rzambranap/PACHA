"""
Machine learning utilities module.

This module provides utility functions for machine learning tasks such as
dataset splitting for training and testing.
"""

import numpy as np
import xarray as xr


def split_dataset(ds,
                  training=0.8,
                  testing=0.2,
                  dim='time',
                  method='random',
                  **kwargs):
    """
    Split a dataset into training and testing sets.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to be split.
    training : float, optional
        Ratio of dataset to use for training. Default is 0.8.
    testing : float, optional
        Ratio of dataset to use for testing. Default is 0.2.
    dim : str, optional
        Dimension along which to split. Default is 'time'.
    method : str, optional
        Split method: 'random' or 'sequential'. Default is 'random'.
    **kwargs
        Additional arguments:
        - seed : int - Random seed for reproducibility (for 'random' method).

    Returns
    -------
    train_ds : xarray.Dataset
        Training subset of the dataset.
    test_ds : xarray.Dataset
        Testing subset of the dataset.

    Raises
    ------
    ValueError
        If method is not 'random' or 'sequential'.
    ValueError
        If training + testing != 1.
    TypeError
        If ds is not an xarray.Dataset.
    ValueError
        If split results in zero elements.

    Examples
    --------
    >>> train, test = split_dataset(ds, training=0.8, testing=0.2)
    >>> train, test = split_dataset(ds, method='sequential')
    >>> train, test = split_dataset(ds, method='random', seed=42)
    """
    valid_methods = ['random', 'sequential']
    if method not in valid_methods:
        raise ValueError(f'Unknown method {method}, valid methods are {valid_methods}')

    sum_ratios = training + testing
    if np.abs(sum_ratios - 1) > 0.001:
        raise ValueError(f'sum of set_ratios != 1, = {sum_ratios}')
    else:
        ds = ds.copy()

    if not isinstance(ds, xr.Dataset):
        raise TypeError(f'The object ds must be an xarray.Dataset, it is a {type(ds)}')

    nsamples = ds['time'].size
    split_ratios = [training, testing]
    train_elts, test_elts = [np.floor(i * nsamples).astype(int) for i in split_ratios]

    if not all(x > 0 for x in [train_elts, test_elts]):
        raise ValueError(f'Not enough elements, splits of {train_elts, test_elts}')

    if method == 'sequential':
        train_idx = ds['time'][:train_elts]
        test_idx = ds['time'][train_elts:train_elts + test_elts]

    if method == 'random':
        tidcs = ds['time'].copy().to_numpy()
        if 'seed' in kwargs:
            nseed = kwargs['seed']
            np.random.seed(nseed)
        np.random.shuffle(tidcs)
        train_idx = tidcs[:train_elts]
        test_idx = tidcs[train_elts:train_elts + test_elts]

    train_ds = ds.sel(time=train_idx)
    test_ds = ds.sel(time=test_idx)

    return train_ds, test_ds

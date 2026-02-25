"""
Satellite Precipitation Product (SPP) correction module.

This module implements a clustered quantile matching correction workflow
for satellite precipitation products, based on the spp_correction notebook.
The approach uses KMeans clustering to group time steps by precipitation
regime, then applies quantile matching correction parameters per cluster.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans

from .quantile_matching import Fuser


def _default_correction_func(x, a, b, c):
    """
    Default quadratic correction function.

    Parameters
    ----------
    x : array-like
        Input precipitation values.
    a : float
        Quadratic coefficient.
    b : float
        Linear coefficient.
    c : float
        Constant coefficient.

    Returns
    -------
    array-like
        Corrected values: a*x**2 + b*x + c
    """
    return a * x**2 + b * x + c


def calculate_features_for_clustering(x, threshold=1.5):
    """
    Calculate statistical features for KMeans clustering.

    Computes spatial statistics (mean, support fraction, standard deviation,
    coefficient of variation) for each time step in the dataset.

    Parameters
    ----------
    x : xarray.Dataset or xarray.DataArray
        Input dataset or data array with at least 'lat', 'lon', and 'time' dims.
    threshold : float, optional
        Precipitation threshold used to compute support fraction. Default is 1.5.

    Returns
    -------
    dict
        Dictionary mapping variable names to DataFrames with columns
        ['spm', 'sup', 'std', 'cvr'] indexed by time.

    Raises
    ------
    TypeError
        If x is not an xarray.Dataset or xarray.DataArray.
    TypeError
        If required dimensions 'lat' and 'lon' are not present.

    Examples
    --------
    >>> features = calculate_features_for_clustering(ds)
    >>> features_df = features['imerg_v7'].dropna()
    """
    if not isinstance(x, (xr.Dataset, xr.DataArray)):
        raise TypeError(
            f"input isn't of valid type, valid types are xr.Dataset or xr.DataArray,\n"
            f"x is {type(x)}"
        )

    needed_dims = ['lat', 'lon']
    is_ds = isinstance(x, xr.Dataset)
    is_da = isinstance(x, xr.DataArray)

    if is_ds:
        dvars = list(x.data_vars)
        dims = x[dvars[0]].dims
    if is_da:
        dims = x.dims

    present_dims = set(dims)
    if not all(d in present_dims for d in needed_dims):
        raise TypeError(f'needed dims are {needed_dims}, x only has {present_dims}')

    if is_ds:
        dvars = list(x.data_vars)
        single_tstep = x[dvars[0]].isel({'time': 0})
        cols = dvars
    if is_da:
        try:
            single_tstep = x.isel({'time': 0})
        except Exception:
            single_tstep = x
        cols = [x.name]

    nvalid_pixels = single_tstep.count().values
    support_over_threshold = (x > threshold).sum(dim=['lat', 'lon'])

    spm = (x.mean(dim=['lat', 'lon'])).to_dataframe().loc[:, cols]
    sup = (support_over_threshold / nvalid_pixels).to_dataframe().loc[:, cols]
    std = x.std(dim=['lat', 'lon']).to_dataframe().loc[:, cols]
    cvr = std / spm

    feat_names = ['spm', 'sup', 'std', 'cvr']
    features_by_var = {}
    for col in cols:
        feats = pd.concat(
            [spm.loc[:, col], sup.loc[:, col], std.loc[:, col], cvr.loc[:, col]],
            axis=1
        )
        feats.columns = feat_names
        features_by_var[col] = feats

    return features_by_var


def subdivide_dataset(ds, nsubdivisions):
    """
    Subdivide a dataset into nsubdivisions x nsubdivisions equal spatial parts.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with 'lat' and 'lon' coordinates.
    nsubdivisions : int
        Number of subdivisions along each spatial dimension.

    Returns
    -------
    list of xarray.Dataset
        List of spatially subdivided datasets. The list has
        nsubdivisions**2 elements ordered row-by-row (lat first, then lon).

    Examples
    --------
    >>> parts = subdivide_dataset(ds, 2)
    >>> len(parts)
    4
    """
    lon_splits = np.array_split(ds.lon, nsubdivisions)
    lat_splits = np.array_split(ds.lat, nsubdivisions)

    subsets = []
    for i in range(nsubdivisions):
        for j in range(nsubdivisions):
            subset = ds.sel(lon=lon_splits[j], lat=lat_splits[i])
            subsets.append(subset)

    return subsets


class ClusteredCorrector:
    """
    Clustered quantile matching corrector for satellite precipitation products.

    Implements a correction workflow where time steps are grouped into clusters
    by precipitation regime (using KMeans on spatial statistics), and each
    cluster receives its own quantile matching correction parameters.

    Parameters
    ----------
    fuser_params : dict, optional
        Parameters for the Fuser instance. Must include 'method'.
        Defaults to quantile_matching_by_parts with land/sea separation.
    n_clusters : int, optional
        Number of KMeans clusters. Default is 4.
    random_state : int, optional
        Random seed for KMeans reproducibility. Default is 12.

    Attributes
    ----------
    fuser_params : dict
        Stored fuser configuration.
    n_clusters : int
        Number of clusters.
    random_state : int
        KMeans random seed.
    fuser : Fuser
        Quantile matching fuser instance.
    kmeans : sklearn.cluster.KMeans or None
        Trained KMeans model (None until fit is called).
    params_by_label : dict or None
        Correction parameters per cluster label (None until fit is called).

    Examples
    --------
    >>> corrector = ClusteredCorrector(n_clusters=4)
    >>> corrector.fit(train_ds, target_variable='imerg_v7')
    >>> corrected_ds = corrector.apply(new_ds, target_variable='precipitation')
    >>> corrector.save('my_corrector.pkl')
    >>> loaded = ClusteredCorrector.load('my_corrector.pkl')
    """

    def __init__(self, fuser_params=None, n_clusters=4, random_state=12):
        """Initialize the ClusteredCorrector."""
        if fuser_params is None:
            fuser_params = {
                'method': 'quantile_matching_by_parts',
                'reference_variable': 'rain_rate',
                'sep_land_sea': True,
                'func': _default_correction_func,
            }
        self.fuser_params = fuser_params
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.fuser = Fuser(fuser_params)
        self.kmeans = None
        self.params_by_label = None

    def fit(self, ds, target_variable):
        """
        Train KMeans classifier and compute per-cluster correction parameters.

        Parameters
        ----------
        ds : xarray.Dataset
            Training dataset containing both the satellite variable and the
            reference variable (as required by the Fuser).
        target_variable : str
            Name of the variable to use for clustering feature extraction.

        Returns
        -------
        self
            Returns self to allow method chaining.

        Examples
        --------
        >>> corrector.fit(train_ds, target_variable='imerg_v7')
        """
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state
        )

        features_df = calculate_features_for_clustering(ds)[target_variable].dropna()
        self.kmeans.fit(features_df)

        features_df = calculate_features_for_clustering(ds)[target_variable].dropna()
        features_df = features_df.copy()
        features_df.loc[:, 'label'] = self.kmeans.predict(features_df)

        dates_by_label = {
            i: sorted(features_df.loc[features_df.label == i].index)
            for i in range(self.n_clusters)
        }
        dss_by_label = {
            i: ds.sel({'time': dates_by_label[i]})
            for i in range(self.n_clusters)
        }
        self.params_by_label = {
            i: self.fuser.calculate_fusing_params(dss_by_label[i])
            for i in range(self.n_clusters)
        }
        return self

    def apply(self, ds, target_variable, nsubdivisions=2):
        """
        Apply the trained correction to a dataset.

        The dataset is optionally subdivided spatially before applying the
        per-cluster correction, then merged back.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset to correct. Must contain the target variable.
        target_variable : str
            Name of the variable to use for clustering feature extraction.
        nsubdivisions : int, optional
            Number of spatial subdivisions along each axis. Default is 2
            (producing a 2x2 spatial grid). Use 1 for no subdivision.

        Returns
        -------
        xarray.Dataset
            Corrected dataset with corrected values replacing the original.

        Raises
        ------
        RuntimeError
            If the corrector has not been fitted yet.

        Examples
        --------
        >>> corrected = corrector.apply(new_ds, target_variable='precipitation')
        """
        if self.kmeans is None or self.params_by_label is None:
            raise RuntimeError(
                'ClusteredCorrector must be fitted before calling apply. '
                'Call fit() first.'
            )

        if nsubdivisions > 1:
            subdivisions = subdivide_dataset(ds, nsubdivisions)
        else:
            subdivisions = [ds]

        corrected_parts = []
        for subdivision in subdivisions:
            feats_df = (
                calculate_features_for_clustering(subdivision)[target_variable]
                .fillna(20)
            )
            feats_df = feats_df.copy()
            feats_df.loc[:, 'label'] = self.kmeans.predict(feats_df)

            dates_by_label = {
                i: sorted(feats_df.loc[feats_df.label == i].index)
                for i in range(self.n_clusters)
            }
            dss_by_label = {
                i: subdivision.sel({'time': dates_by_label[i]})
                for i in range(self.n_clusters)
            }

            corrected_dss = []
            for label in range(self.n_clusters):
                corrected = self.fuser.apply_params(
                    dss_by_label[label], self.params_by_label[label]
                )
                if hasattr(corrected, 'drop_duplicates'):
                    corrected = corrected.drop_duplicates('time')
                corrected_dss.append(corrected)

            merged = xr.merge(corrected_dss)
            corrected_parts.append(merged)

        return xr.merge(corrected_parts)

    def save(self, filepath):
        """
        Save the corrector (KMeans model + correction parameters) to a pickle file.

        Parameters
        ----------
        filepath : str
            Path to save the pickle file.

        Examples
        --------
        >>> corrector.save('my_corrector.pkl')
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Load a corrector from a pickle file.

        Parameters
        ----------
        filepath : str
            Path to the pickle file.

        Returns
        -------
        ClusteredCorrector
            Loaded corrector instance.

        Examples
        --------
        >>> corrector = ClusteredCorrector.load('my_corrector.pkl')
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def save_params(self, filepath):
        """
        Save the correction parameters to a JSON file.

        Saves only the numeric correction parameters (offsets and polynomial
        coefficients) without the function reference, making the file
        human-readable and portable.

        Parameters
        ----------
        filepath : str
            Path to save the JSON file.

        Raises
        ------
        RuntimeError
            If the corrector has not been fitted yet.

        Examples
        --------
        >>> corrector.save_params('correction_params.json')
        """
        if self.params_by_label is None:
            raise RuntimeError(
                'No parameters to save. Call fit() first.'
            )
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        serializable = self._params_to_serializable(self.params_by_label)
        with open(filepath, 'w') as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load_params(cls, filepath, func=None):
        """
        Load correction parameters from a JSON file.

        Parameters
        ----------
        filepath : str
            Path to the JSON file produced by save_params.
        func : callable, optional
            Correction function to attach to each parameter set.
            Defaults to _default_correction_func if not provided.

        Returns
        -------
        dict
            Parameters dictionary with the same structure as params_by_label,
            including the function reference.

        Examples
        --------
        >>> params = ClusteredCorrector.load_params('correction_params.json')
        """
        if func is None:
            func = _default_correction_func
        with open(filepath, 'r') as f:
            raw = json.load(f)
        return cls._params_from_serializable(raw, func)

    @staticmethod
    def _params_to_serializable(params_by_label):
        """Convert params_by_label to a JSON-serializable dict."""
        serializable = {}
        for label, label_params in params_by_label.items():
            serializable[str(label)] = {}
            for region_key, region_params in label_params.items():
                serializable[str(label)][region_key] = {
                    'method': region_params['method'],
                    'offset': float(np.asarray(region_params['offset'])),
                    'p1_popt': [float(v) for v in region_params['p1_popt']],
                    'p2_popt': [float(v) for v in region_params['p2_popt']],
                }
        return serializable

    @staticmethod
    def _params_from_serializable(raw, func):
        """Reconstruct params_by_label from a serializable dict."""
        params_by_label = {}
        for label_str, label_params in raw.items():
            if label_str.startswith('_'):
                continue
            label = int(label_str)
            params_by_label[label] = {}
            for region_key, region_params in label_params.items():
                params_by_label[label][region_key] = {
                    'method': region_params['method'],
                    'offset': np.array(region_params['offset']),
                    'p1_popt': np.array(region_params['p1_popt']),
                    'p2_popt': np.array(region_params['p2_popt']),
                    'func': func,
                }
        return params_by_label

    def __getstate__(self):
        """Custom pickle state: avoids serializing Fuser's bound private methods."""
        return {
            'fuser_params': self.fuser_params,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'kmeans': self.kmeans,
            'params_by_label_serial': (
                self._params_to_serializable(self.params_by_label)
                if self.params_by_label is not None else None
            ),
        }

    def __setstate__(self, state):
        """Custom pickle restore: reconstructs Fuser and params from saved state."""
        self.fuser_params = state['fuser_params']
        self.n_clusters = state['n_clusters']
        self.random_state = state['random_state']
        self.fuser = Fuser(self.fuser_params)
        self.kmeans = state['kmeans']
        if state['params_by_label_serial'] is not None:
            func = self.fuser_params.get('func', _default_correction_func)
            self.params_by_label = self._params_from_serializable(
                state['params_by_label_serial'], func
            )
        else:
            self.params_by_label = None

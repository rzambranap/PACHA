"""
Quantile matching module for data fusion.

This module provides functions and classes for merging precipitation datasets
using quantile matching and related bias correction techniques.
"""

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from ..utils.geospatial import divide_dataset_land_sea


def fit_qs(qs, fit_func, char_pcp='radar'):
    """
    Fit a function to quantile data.

    Parameters
    ----------
    qs : xarray.Dataset
        Dataset containing quantile values for two variables.
    fit_func : callable
        Function to fit (e.g., polynomial function).
    char_pcp : str, optional
        Substring identifying the precipitation variable. Default is 'radar'.

    Returns
    -------
    numpy.ndarray
        Optimal parameters for the fit function.

    Raises
    ------
    ValueError
        If the dataset does not contain exactly two variables.

    Examples
    --------
    >>> qs = ds.quantile(np.linspace(0.01, 0.99, 99))
    >>> popt = fit_qs(qs, lambda x, a, b: a*x + b)
    """
    fields = list(qs.keys())
    if len(fields) != 2:
        raise ValueError(
            f'Dataset must have two variables, but has {len(fields)} fields'
        )

    idx_pcp = [i for i in range(len(fields)) if char_pcp in fields[i]][0]
    field1 = fields.pop(idx_pcp)
    field2 = fields[0]

    x = qs[field2].data
    y = qs[field1].data

    popt, pcov = curve_fit(fit_func, x, y, maxfev=100000)
    return popt


def echelon_down(x, offset=0, speed=5):
    """
    Smooth step-down function using hyperbolic tangent.

    Creates a smooth transition from 1 to 0 centered at offset.

    Parameters
    ----------
    x : array-like
        Input values.
    offset : float, optional
        Center point of the transition. Default is 0.
    speed : float, optional
        Steepness of the transition. Higher values = sharper transition.
        Default is 5.

    Returns
    -------
    array-like
        Values between 0 and 1, transitioning from 1 to 0 around offset.

    Examples
    --------
    >>> y = echelon_down(np.linspace(-2, 2, 100), offset=0, speed=5)
    """
    y = (1 + ((np.tanh(-speed * (x - offset))))) / 2
    return y


def echelon_up(x, offset=0, speed=5):
    """
    Smooth step-up function using hyperbolic tangent.

    Creates a smooth transition from 0 to 1 centered at offset.

    Parameters
    ----------
    x : array-like
        Input values.
    offset : float, optional
        Center point of the transition. Default is 0.
    speed : float, optional
        Steepness of the transition. Higher values = sharper transition.
        Default is 5.

    Returns
    -------
    array-like
        Values between 0 and 1, transitioning from 0 to 1 around offset.

    Examples
    --------
    >>> y = echelon_up(np.linspace(-2, 2, 100), offset=0, speed=5)
    """
    y = 1 - echelon_down(x, offset=offset, speed=speed)
    return y


def rescale_qs_to_calc(qs_to_calc, cut_off_q):
    """
    Rescale quantile values for two-part matching.

    Parameters
    ----------
    qs_to_calc : array-like
        Original quantile values (0 to 1).
    cut_off_q : float
        Quantile value at which to split (0 to 1).

    Returns
    -------
    xqs1 : array-like
        Rescaled quantiles for the lower part.
    xqs2 : array-like
        Rescaled quantiles for the upper part.

    Examples
    --------
    >>> xqs1, xqs2 = rescale_qs_to_calc(np.linspace(0, 1, 100), 0.75)
    """
    xqs1 = qs_to_calc * cut_off_q
    xqs2 = cut_off_q + (qs_to_calc * (1 - cut_off_q))
    return xqs1, xqs2


def match_quantiles(ds,
                    fit_func,
                    reference_variable='radar',
                    quantiles_to_calc=None,
                    suffix='matched',
                    by_parts=False,
                    cut_off_q=None,
                    output_params=False):
    """
    Apply quantile matching to correct bias between two datasets.

    Fits a transfer function between quantiles of reference and target
    variables, then applies the correction.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with exactly two variables to match.
    fit_func : callable
        Function to fit for the transfer (e.g., polynomial).
    reference_variable : str, optional
        Substring identifying the reference variable. Default is 'radar'.
    quantiles_to_calc : array-like, optional
        Quantile values to use for fitting. Default is np.linspace(0.01, 0.99, 99).
    suffix : str, optional
        Suffix for the corrected variable name. Default is 'matched'.
    by_parts : bool, optional
        Whether to use two-part matching. Default is False.
    cut_off_q : float, optional
        Quantile cutoff for two-part matching. Required if by_parts=True.
    output_params : bool, optional
        If True, return parameters instead of matched dataset. Default is False.

    Returns
    -------
    xarray.Dataset or dict
        Matched dataset with new variable, or parameters dict if output_params=True.

    Raises
    ------
    ValueError
        If dataset does not contain exactly two variables.

    Examples
    --------
    >>> def quad_func(x, a, b, c):
    ...     return a*x**2 + b*x + c
    >>> matched_ds = match_quantiles(ds, quad_func)
    """
    if quantiles_to_calc is None:
        quantiles_to_calc = np.linspace(0.01, 0.99, 99)

    qs = ds.quantile(quantiles_to_calc)
    popt = fit_qs(qs, fit_func, char_pcp=reference_variable)
    ds_out = ds.copy()
    fields = list(ds.keys())

    if len(fields) != 2:
        raise ValueError(
            f'Dataset must have two variables, but has {len(fields)} fields'
        )

    idx_pcp = [i for i in range(len(fields)) if reference_variable in fields[i]][0]
    _ = fields.pop(idx_pcp)
    field2 = fields[0]

    if by_parts:
        if cut_off_q == 0:
            return match_quantiles(
                ds, fit_func,
                reference_variable=reference_variable,
                quantiles_to_calc=quantiles_to_calc
            )
        if isinstance(quantiles_to_calc, dict):
            xqs1 = quantiles_to_calc['xqs1']
            xqs2 = quantiles_to_calc['xqs2']
        elif isinstance(quantiles_to_calc, (list, np.ndarray)):
            xqs1, xqs2 = rescale_qs_to_calc(quantiles_to_calc, cut_off_q)

        offset = ds.quantile(cut_off_q)[field2].data
        part1 = match_quantiles(
            ds, fit_func,
            reference_variable=reference_variable,
            quantiles_to_calc=xqs1
        )[field2 + '_' + suffix]
        p1_popts = part1.attrs['popt']

        part2 = match_quantiles(
            ds, fit_func,
            reference_variable=reference_variable,
            quantiles_to_calc=xqs2
        )[field2 + '_' + suffix]
        p2_popts = part2.attrs['popt']

        part1_ech = part1 * echelon_down(ds[field2], offset=offset, speed=20)
        part2_ech = part2 * echelon_up(ds[field2], offset=offset, speed=20)
        ds_out[field2 + '_' + suffix] = part1_ech + part2_ech

        if output_params:
            params = {'method': 'qmatch_by_parts'}
            params.update({'offset': offset})
            params.update({'p1_popt': p1_popts, 'p2_popt': p2_popts})
            params.update({'func': fit_func})
            return params
        return ds_out

    fld = fit_func(ds_out[field2], *popt)
    fld_modif = xr.where(fld < 0, 0, fld)
    ds_out[field2 + '_' + suffix] = fld_modif
    ds_out[field2 + '_' + suffix].attrs['popt'] = popt

    if output_params:
        params = {'method': 'qmatch'}
        params.update({'popt': popt})
        params.update({'func': fit_func})
        return params
    return ds_out


class Fuser:
    """
    Data fusion class for combining precipitation datasets.

    Supports multiple fusion methods including quantile matching,
    quantile matching by parts, and linear regression.

    Parameters
    ----------
    fusing_params : dict
        Dictionary of fusion parameters. Must include 'method' key.
        Optional keys depend on method:
        - 'reference_variable': substring identifying reference (default 'radar')
        - 'quantiles_to_calc': array of quantile values
        - 'func': custom fitting function
        - 'sep_land_sea': whether to process land/sea separately
        - 'cut_off_q': quantile cutoff for by_parts method

    Attributes
    ----------
    fusing_params : dict
        Stored fusion parameters.
    char_pcp : str
        Reference variable identifier.
    func : callable
        Fitting function.

    Methods
    -------
    calculate_fusing_params(ds)
        Calculate fusion parameters from training data.
    apply_params(ds, params)
        Apply pre-calculated parameters to new data.
    export_matching_ds(ds, params=None)
        Export dataset with matched variable added.

    Examples
    --------
    >>> params = {'method': 'quantile_matching', 'reference_variable': 'radar'}
    >>> fuser = Fuser(params)
    >>> train_params = fuser.calculate_fusing_params(training_ds)
    >>> matched_ds = fuser.export_matching_ds(test_ds, train_params)
    """

    def __init__(self, fusing_params):
        """Initialize the Fuser with fusion parameters."""
        method = fusing_params['method']
        init_funcs = {
            'quantile_matching': self.__init_qmatch,
            'quantile_matching_by_parts': self.__init_qmatch_byparts,
            'linear_regression': self.__init_linreg
        }
        self.fusing_params = fusing_params
        self.params_list = list(fusing_params.keys())

        if 'reference_variable' in self.params_list:
            self.char_pcp = fusing_params['reference_variable']
        else:
            self.char_pcp = 'radar'

        init_funcs[method]()
        return

    def __init_qmatch(self):
        """Initialize quantile matching method."""
        params_list = self.params_list
        if 'quantiles_to_calc' in params_list:
            self.qs_to_calc = self.fusing_params['quantiles_to_calc']
        else:
            self.qs_to_calc = np.linspace(0, 1, 10000, endpoint=False)
        if 'func' in params_list:
            self.func = self.fusing_params['func']
        else:
            self.func = self.__base_func
        if 'sep_land_sea' in self.params_list:
            self.sep_bool = self.fusing_params['sep_land_sea']
        else:
            self.sep_bool = False
        self.calculate_fusing_params = self.__calc_qmatch_params
        self.low_level_applier = self.__qmatch_applier
        self.apply_params = self.__apply_qmatch_params
        self.export_matching_ds = self.__export_matched_ds
        return

    def __init_qmatch_byparts(self):
        """Initialize quantile matching by parts method."""
        params_list = self.params_list
        if 'cut_off_q' in params_list:
            self.cut_off_q = self.fusing_params['cut_off_q']
        else:
            self.cut_off_q = 0.75
        if 'quantiles_to_calc' in params_list:
            self.qs_to_calc = self.fusing_params['quantiles_to_calc']
        else:
            self.qs_to_calc = np.linspace(0, 1, 10000, endpoint=False)
        if 'func' in params_list:
            self.func = self.fusing_params['func']
        else:
            self.func = self.__base_func
        if 'sep_land_sea' in self.params_list:
            self.sep_bool = self.fusing_params['sep_land_sea']
        else:
            self.sep_bool = False
        self.calculate_fusing_params = self.__calc_qmatch_by_parts_params
        self.low_level_applier = self.__qmatchbyparts_applier
        self.apply_params = self.__apply_qmatch_by_parts_params
        self.export_matching_ds = self.__export_matched_ds
        return

    def __init_linreg(self):
        """Initialize linear regression method."""
        params_list = self.params_list
        if 'sep_land_sea' in params_list:
            self.sep_bool = self.fusing_params['sep_land_sea']
        else:
            self.sep_bool = False
        self.calculate_fusing_params = self.__calc_linreg_params
        self.low_level_applier = self.__linreg_applier
        self.apply_params = self.__apply_linreg_params
        self.func = self.__lin_func
        self.export_matching_ds = self.__export_matched_ds
        return

    def __calc_linreg_params(self, ds, skip=False):
        """Calculate linear regression parameters."""
        if self.sep_bool and not skip:
            land, sea = divide_dataset_land_sea(ds)
            land_params = self.__calc_linreg_params(land, skip=True)
            sea_params = self.__calc_linreg_params(sea, skip=True)
            return {'land_params': land_params, 'sea_params': sea_params}
        else:
            dfmean_nona = ds.mean(dim=['lat', 'lon']).to_dataframe().dropna()
            f1, f2 = self.define_fields(ds)
            popt, _ = curve_fit(self.func, dfmean_nona.loc[:, f2], dfmean_nona.loc[:, f1])
            params = {'popt': popt}
            params.update({'method': 'linear_regression'})
        return params

    def __calc_qmatch_by_parts_params(self, ds, skip=False):
        """Calculate quantile matching by parts parameters."""
        if self.sep_bool and not skip:
            land, sea = divide_dataset_land_sea(ds)
            land_params = self.__calc_qmatch_by_parts_params(land, skip=True)
            sea_params = self.__calc_qmatch_by_parts_params(sea, skip=True)
            return {'land_params': land_params, 'sea_params': sea_params}
        else:
            params = match_quantiles(
                ds, fit_func=self.func, output_params=True,
                reference_variable=self.char_pcp,
                quantiles_to_calc=self.qs_to_calc,
                by_parts=True, cut_off_q=self.cut_off_q
            )
        return params

    def __calc_qmatch_params(self, ds, skip=False):
        """Calculate quantile matching parameters."""
        if self.sep_bool and not skip:
            land, sea = divide_dataset_land_sea(ds)
            land_params = self.__calc_qmatch_params(land, skip=True)
            sea_params = self.__calc_qmatch_params(sea, skip=True)
            return {'land_params': land_params, 'sea_params': sea_params}
        else:
            params = match_quantiles(
                ds, fit_func=self.func, output_params=True,
                reference_variable=self.char_pcp,
                quantiles_to_calc=self.qs_to_calc,
                by_parts=False
            )
        return params

    def __apply_qmatch_by_parts_params(self, ds, params, skip=False):
        """Apply quantile matching by parts parameters."""
        if self.sep_bool and not skip:
            land, sea = divide_dataset_land_sea(ds)
            land_applied = self.__apply_qmatch_by_parts_params(
                land, params['land_params'], skip=True
            )
            sea_applied = self.__apply_qmatch_by_parts_params(
                sea, params['sea_params'], skip=True
            )
            y = xr.merge([land_applied, sea_applied])
            return y
        else:
            y = self.low_level_applier(ds, params)
        return y

    def __apply_qmatch_params(self, ds, params, skip=False):
        """Apply quantile matching parameters."""
        if self.sep_bool and not skip:
            land, sea = divide_dataset_land_sea(ds)
            land_applied = self.__apply_qmatch_params(
                land, params['land_params'], skip=True
            )
            sea_applied = self.__apply_qmatch_params(
                sea, params['sea_params'], skip=True
            )
            y = xr.merge([land_applied, sea_applied])
            return y
        else:
            y = self.low_level_applier(ds, params)
        return y

    def __apply_linreg_params(self, ds, params, skip=False):
        """Apply linear regression parameters."""
        if self.sep_bool and not skip:
            land, sea = divide_dataset_land_sea(ds)
            land_applied = self.__apply_linreg_params(
                land, params['land_params'], skip=True
            )
            sea_applied = self.__apply_linreg_params(
                sea, params['sea_params'], skip=True
            )
            y = xr.merge([land_applied, sea_applied])
            return y
        else:
            y = self.low_level_applier(ds, params)
        return y

    def __export_matched_ds(self, ds, params=None):
        """Export dataset with matched variable."""
        field1, field2 = self.define_fields(ds)
        if params is None:
            params = self.calculate_fusing_params(ds)
        f2_matched = self.apply_params(ds[field2], params)
        ds_out = ds.copy()
        if isinstance(f2_matched, xr.DataArray):
            ds_out[field2 + '_matched'] = f2_matched
        else:
            ds_out[field2 + '_matched'] = f2_matched[field2]
        return ds_out

    def __qmatchbyparts_applier(self, ds, params):
        """Apply quantile matching by parts to data."""
        ds_out = ds.copy()
        offset = params['offset']
        p1_popts = params['p1_popt']
        p2_popts = params['p2_popt']
        func = params['func']
        part1 = func(ds, *p1_popts)
        part2 = func(ds, *p2_popts)
        part1_ech = part1 * echelon_down(ds, offset=offset, speed=20)
        part2_ech = part2 * echelon_up(ds, offset=offset, speed=20)
        ds_out = part1_ech + part2_ech
        ds_out = xr.where(ds_out < 0, 0, ds_out)
        return ds_out

    def __qmatch_applier(self, ds, params):
        """Apply quantile matching to data."""
        ds_out = ds.copy()
        popts = params['popt']
        func = params['func']
        ds_out = func(ds, *popts)
        ds_out = xr.where(ds_out < 0, 0, ds_out)
        return ds_out

    def __linreg_applier(self, ds, params):
        """Apply linear regression to data."""
        ds_out = ds.copy()
        multiplier = params['popt']
        ds_out = multiplier * ds_out
        return ds_out

    def define_fields(self, ds):
        """
        Define reference and target fields from dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Dataset with exactly two variables.

        Returns
        -------
        field1 : str
            Name of the reference field (containing char_pcp).
        field2 : str
            Name of the target field.

        Raises
        ------
        ValueError
            If dataset does not contain exactly two variables.
        """
        fields = list(ds.keys())
        if len(fields) != 2:
            raise ValueError(
                f'Dataset must have two variables, but has {len(fields)} fields'
            )

        idx_pcp = [i for i in range(len(fields)) if self.char_pcp in fields[i]][0]
        field1 = fields.pop(idx_pcp)
        field2 = fields[0]
        return field1, field2

    @staticmethod
    def __base_func(x, a, b, c):
        """Quadratic transfer function."""
        return np.abs(a) * x ** 2 + (b) * x ** 1 + (c) * x ** 0

    @staticmethod
    def __lin_func(x, a):
        """Linear transfer function."""
        return a * x

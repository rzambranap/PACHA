"""
Daily precipitation accumulation pipeline.

This module provides a modular, verifiable, and script-friendly pipeline for
computing daily precipitation accumulation from half-hourly and hourly rain
intensity estimates.

The pipeline consists of the following steps:
1. Load and normalize dataset
2. Check timestep completeness
3. Gap fill/interpolate missing data
4. Calculate daily accumulation
5. Save output with metadata provenance

Supported Products
------------------
- IMERG: 30-minute temporal resolution (48 timesteps/day)
- GSMaP: 1-hour temporal resolution (24 timesteps/day)
- Radar: Variable resolution (typically 10-minute)

Example Usage
-------------
>>> from pacha.L2_processing.daily_accumulation import (
...     DailyAccumulationPipeline,
...     compute_daily_accumulation
... )
>>>
>>> # Simple usage with compute function
>>> daily_accum = compute_daily_accumulation(
...     input_path='/data/imerg/',
...     product_type='imerg',
...     date='2023-01-15',
...     output_path='/data/output/daily_2023-01-15.nc'
... )
>>>
>>> # Or use the pipeline class for more control
>>> pipeline = DailyAccumulationPipeline(
...     product_type='imerg',
...     input_path='/data/imerg/',
...     output_path='/data/output/'
... )
>>> pipeline.process_date_range('2023-01-01', '2023-01-31')
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# Configure module logger
logger = logging.getLogger(__name__)


# Product configuration: expected timesteps per day and temporal resolution
PRODUCT_CONFIG: Dict[str, Dict] = {
    'imerg': {
        'timesteps_per_day': 48,
        'temporal_resolution_hours': 0.5,
        'precip_variable': 'precipitation',
        'description': 'IMERG half-hourly precipitation product'
    },
    'gsmap': {
        'timesteps_per_day': 24,
        'temporal_resolution_hours': 1.0,
        'precip_variable': 'precipitationCal',
        'description': 'GSMaP hourly precipitation product'
    },
    'radar': {
        'timesteps_per_day': 144,  # 10-minute resolution
        'temporal_resolution_hours': 1/6,
        'precip_variable': 'rain_rate',
        'description': 'Radar precipitation estimates'
    }
}


def load_dataset(
    input_path: str,
    product_type: str,
    date: str,
    precip_variable: Optional[str] = None
) -> xr.Dataset:
    """
    Load and normalize a precipitation dataset for a given date.

    This function loads data files for the specified date and normalizes
    the dataset to ensure consistent time coordinates and dimension ordering.

    Parameters
    ----------
    input_path : str
        Path to the directory containing input data files.
    product_type : str
        Type of precipitation product. One of 'imerg', 'gsmap', or 'radar'.
    date : str
        Date string in 'YYYY-MM-DD' format.
    precip_variable : str, optional
        Name of the precipitation variable in the dataset. If None, uses
        the default for the product type.

    Returns
    -------
    xr.Dataset
        Normalized dataset with precipitation data for the specified date.

    Raises
    ------
    ValueError
        If product_type is not recognized.
    FileNotFoundError
        If no data files are found for the specified date.

    Examples
    --------
    >>> ds = load_dataset('/data/imerg/', 'imerg', '2023-01-15')
    >>> print(ds)
    """
    if product_type not in PRODUCT_CONFIG:
        raise ValueError(
            f"Unknown product type '{product_type}'. "
            f"Supported types: {list(PRODUCT_CONFIG.keys())}"
        )

    logger.info(f"Loading {product_type} data for date {date} from {input_path}")

    config = PRODUCT_CONFIG[product_type]
    if precip_variable is None:
        precip_variable = config['precip_variable']

    # Try to find and load data files
    # This is a simplified implementation - actual file loading may vary
    # based on the data source organization
    try:
        ds = _load_files_for_date(input_path, product_type, date, precip_variable)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Normalize the dataset
    ds = normalize_dataset(ds, product_type)

    logger.info(f"Successfully loaded dataset with {len(ds.time)} timesteps")

    return ds


def _load_files_for_date(
    input_path: str,
    product_type: str,
    date: str,
    precip_variable: str
) -> xr.Dataset:
    """
    Internal function to load data files for a specific date.

    Parameters
    ----------
    input_path : str
        Path to the directory containing input data files.
    product_type : str
        Type of precipitation product.
    date : str
        Date string in 'YYYY-MM-DD' format.
    precip_variable : str
        Name of the precipitation variable.

    Returns
    -------
    xr.Dataset
        Loaded dataset.
    """
    import glob
    import os

    date_dt = pd.to_datetime(date)
    date_str = date_dt.strftime('%Y%m%d')

    # Look for NetCDF files matching the date
    patterns = [
        os.path.join(input_path, f'*{date_str}*.nc'),
        os.path.join(input_path, f'*{date_str}*.nc4'),
        os.path.join(input_path, date_str[:4], date_str[4:6], f'*{date_str}*.nc'),
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No data files found for date {date} in {input_path}"
        )

    logger.debug(f"Found {len(files)} files for date {date}")

    # Load files
    if len(files) == 1:
        ds = xr.open_dataset(files[0])
    else:
        ds = xr.open_mfdataset(files, combine='by_coords')

    return ds


def normalize_dataset(ds: xr.Dataset, product_type: str) -> xr.Dataset:
    """
    Normalize a dataset to ensure consistent structure.

    Normalizes time coordinates to pandas datetime objects, ensures
    consistent dimension ordering (time, lat, lon), and sorts by time.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to normalize.
    product_type : str
        Type of precipitation product.

    Returns
    -------
    xr.Dataset
        Normalized dataset.

    Examples
    --------
    >>> ds = xr.open_dataset('data.nc')
    >>> ds_norm = normalize_dataset(ds, 'imerg')
    """
    logger.debug("Normalizing dataset...")

    # Convert time to datetime if needed
    if 'time' in ds.dims:
        try:
            time_values = pd.to_datetime(ds.time.values)
            ds = ds.assign_coords(time=time_values)
        except Exception:
            pass  # Keep original time if conversion fails

    # Sort by time
    if 'time' in ds.dims:
        ds = ds.sortby('time')

    # Ensure proper dimension order
    expected_dims = ['time', 'lat', 'lon']

    # Handle alternative coordinate names
    dim_mapping = {
        'latitude': 'lat',
        'longitude': 'lon',
        'y': 'lat',
        'x': 'lon'
    }

    for old_name, new_name in dim_mapping.items():
        if old_name in ds.dims and new_name not in ds.dims:
            ds = ds.rename({old_name: new_name})

    # Transpose if all expected dimensions exist
    if all(dim in ds.dims for dim in expected_dims):
        ds = ds.transpose('time', 'lat', 'lon', ...)

    return ds


def check_timestep_completeness(
    ds: xr.Dataset,
    product_type: str,
    date: str,
    max_missing_fraction: float = 0.2
) -> Tuple[bool, Dict]:
    """
    Check if dataset has expected number of timesteps for the day.

    Verifies that the dataset contains the expected number of timesteps
    based on the product type's temporal resolution.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to check.
    product_type : str
        Type of precipitation product.
    date : str
        Date string in 'YYYY-MM-DD' format.
    max_missing_fraction : float, optional
        Maximum fraction of missing timesteps allowed (0 to 1). Default is 0.2.

    Returns
    -------
    is_complete : bool
        True if dataset meets completeness criteria.
    report : dict
        Dictionary containing completeness report with keys:
        - expected_timesteps: int
        - actual_timesteps: int
        - missing_timesteps: int
        - missing_fraction: float
        - missing_times: list of datetime

    Examples
    --------
    >>> ds = load_dataset('/data/imerg/', 'imerg', '2023-01-15')
    >>> is_complete, report = check_timestep_completeness(ds, 'imerg', '2023-01-15')
    >>> print(f"Missing: {report['missing_fraction']:.1%}")
    """
    if product_type not in PRODUCT_CONFIG:
        raise ValueError(f"Unknown product type '{product_type}'")

    config = PRODUCT_CONFIG[product_type]
    expected = config['timesteps_per_day']
    resolution_hours = config['temporal_resolution_hours']

    # Get actual timesteps
    actual = len(ds.time)

    # Generate expected time series for the day
    date_dt = pd.to_datetime(date)
    expected_times = pd.date_range(
        start=date_dt,
        end=date_dt + timedelta(days=1),
        freq=f'{int(resolution_hours * 60)}min',
        inclusive='left'
    )

    # Find missing times
    actual_times = pd.DatetimeIndex(ds.time.values)
    missing_times = expected_times.difference(actual_times)

    missing_count = len(missing_times)
    missing_fraction = missing_count / expected if expected > 0 else 0

    is_complete = missing_fraction <= max_missing_fraction

    report = {
        'expected_timesteps': expected,
        'actual_timesteps': actual,
        'missing_timesteps': missing_count,
        'missing_fraction': missing_fraction,
        'missing_times': missing_times.tolist()
    }

    status = "COMPLETE" if is_complete else "INCOMPLETE"
    logger.info(
        f"Timestep check: {status} - "
        f"{actual}/{expected} timesteps present ({missing_fraction:.1%} missing)"
    )

    if missing_count > 0 and logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Missing times: {missing_times.tolist()[:5]}...")

    return is_complete, report


def gap_fill(
    ds: xr.Dataset,
    product_type: str,
    date: str,
    method: str = 'linear',
    max_gap_hours: Optional[float] = None,
    fill_value: float = 0.0
) -> xr.Dataset:
    """
    Fill temporal gaps in the precipitation dataset.

    Interpolates or fills missing timesteps based on the specified method.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with potential gaps.
    product_type : str
        Type of precipitation product.
    date : str
        Date string in 'YYYY-MM-DD' format.
    method : str, optional
        Interpolation method. One of:
        - 'linear': Linear interpolation (default)
        - 'nearest': Nearest neighbor
        - 'zero': Fill with zeros
        - 'previous': Forward fill with previous value
    max_gap_hours : float, optional
        Maximum gap size (in hours) to interpolate. Larger gaps are filled
        with fill_value. If None, uses 2x temporal resolution.
    fill_value : float, optional
        Value to use for gaps larger than max_gap_hours. Default is 0.0.

    Returns
    -------
    xr.Dataset
        Dataset with gaps filled, including 'gap_filled' attribute.

    Notes
    -----
    The output dataset includes a 'gap_filled_mask' variable indicating
    which timesteps were filled.

    Examples
    --------
    >>> ds = load_dataset('/data/imerg/', 'imerg', '2023-01-15')
    >>> ds_filled = gap_fill(ds, 'imerg', '2023-01-15', method='linear')
    """
    if product_type not in PRODUCT_CONFIG:
        raise ValueError(f"Unknown product type '{product_type}'")

    config = PRODUCT_CONFIG[product_type]
    resolution_hours = config['temporal_resolution_hours']

    if max_gap_hours is None:
        max_gap_hours = 2 * resolution_hours

    logger.info(f"Gap filling with method='{method}', max_gap={max_gap_hours}h")

    # Generate complete time series for the day
    date_dt = pd.to_datetime(date)
    complete_times = pd.date_range(
        start=date_dt,
        end=date_dt + timedelta(days=1),
        freq=f'{int(resolution_hours * 60)}min',
        inclusive='left'
    )

    # Identify existing times
    original_times = pd.DatetimeIndex(ds.time.values)

    # Create mask of original data
    is_original = np.isin(complete_times, original_times)

    # Reindex to complete time series
    ds_complete = ds.reindex(time=complete_times)

    # Apply interpolation method
    if method == 'zero':
        ds_filled = ds_complete.fillna(fill_value)
    elif method == 'previous':
        ds_filled = ds_complete.ffill(dim='time')
        ds_filled = ds_filled.fillna(fill_value)  # Fill leading NaNs
    elif method in ['linear', 'nearest']:
        ds_filled = ds_complete.interpolate_na(
            dim='time',
            method=method,
            fill_value='extrapolate'
        )
        # Handle remaining NaNs (e.g., at boundaries)
        ds_filled = ds_filled.fillna(fill_value)
    else:
        raise ValueError(
            f"Unknown interpolation method '{method}'. "
            "Use 'linear', 'nearest', 'zero', or 'previous'."
        )

    # Check for large gaps and replace with fill_value
    max_gap_steps = int(max_gap_hours / resolution_hours)
    if max_gap_steps > 0:
        # Identify large gaps
        gap_mask = _identify_large_gaps(~is_original, max_gap_steps)
        if gap_mask.any():
            logger.warning(
                f"Found {gap_mask.sum()} timesteps in gaps > {max_gap_hours}h, "
                f"filling with {fill_value}"
            )
            # Apply fill_value to large gaps for all data variables
            for var in ds_filled.data_vars:
                ds_filled[var] = xr.where(
                    xr.DataArray(gap_mask, dims=['time'], coords={'time': complete_times}),
                    fill_value,
                    ds_filled[var]
                )

    # Add gap-filled mask as a coordinate
    gap_filled_mask = xr.DataArray(
        ~is_original,
        dims=['time'],
        coords={'time': complete_times},
        attrs={
            'long_name': 'Gap-filled timestep mask',
            'description': 'True where timestep was filled/interpolated'
        }
    )
    ds_filled = ds_filled.assign_coords(gap_filled_mask=gap_filled_mask)

    # Add gap-filling metadata
    ds_filled.attrs['gap_fill_method'] = method
    ds_filled.attrs['gap_fill_max_gap_hours'] = max_gap_hours
    ds_filled.attrs['gap_fill_fill_value'] = fill_value
    ds_filled.attrs['gap_filled_count'] = int((~is_original).sum())

    n_filled = int((~is_original).sum())
    logger.info(f"Gap filled {n_filled} timesteps")

    return ds_filled


def _identify_large_gaps(is_missing: np.ndarray, max_gap_size: int) -> np.ndarray:
    """
    Identify timesteps that are part of gaps larger than max_gap_size.

    Parameters
    ----------
    is_missing : np.ndarray
        Boolean array where True indicates missing timestep.
    max_gap_size : int
        Maximum gap size (in timesteps) to allow.

    Returns
    -------
    np.ndarray
        Boolean array where True indicates part of a large gap.
    """
    large_gap_mask = np.zeros_like(is_missing, dtype=bool)

    # Find consecutive runs of missing data
    if not is_missing.any():
        return large_gap_mask

    changes = np.diff(np.concatenate([[0], is_missing.astype(int), [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for start, end in zip(starts, ends):
        gap_size = end - start
        if gap_size > max_gap_size:
            large_gap_mask[start:end] = True

    return large_gap_mask


def calculate_daily_accumulation(
    ds: xr.Dataset,
    product_type: str,
    precip_variable: Optional[str] = None,
    min_valid_fraction: float = 0.5
) -> xr.Dataset:
    """
    Calculate daily precipitation accumulation from rain intensity data.

    Converts rain rate (mm/hr) to daily accumulation (mm/day) by integrating
    over the time dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with precipitation rate data.
    product_type : str
        Type of precipitation product.
    precip_variable : str, optional
        Name of the precipitation variable. If None, uses default for product.
    min_valid_fraction : float, optional
        Minimum fraction of valid (non-NaN) timesteps required for a valid
        daily accumulation. Default is 0.5.

    Returns
    -------
    xr.Dataset
        Dataset with daily precipitation accumulation variable.
        Includes 'daily_precipitation' variable in mm/day.

    Notes
    -----
    The calculation assumes precipitation values are in mm/hr (rain rate).
    For instantaneous rain rate, the accumulation is:
        daily_accum = sum(rain_rate * temporal_resolution_hours)

    Examples
    --------
    >>> ds = load_dataset('/data/imerg/', 'imerg', '2023-01-15')
    >>> ds = gap_fill(ds, 'imerg', '2023-01-15')
    >>> daily = calculate_daily_accumulation(ds, 'imerg')
    """
    if product_type not in PRODUCT_CONFIG:
        raise ValueError(f"Unknown product type '{product_type}'")

    config = PRODUCT_CONFIG[product_type]
    resolution_hours = config['temporal_resolution_hours']

    if precip_variable is None:
        precip_variable = config['precip_variable']

    logger.info(
        f"Calculating daily accumulation for '{precip_variable}' "
        f"with resolution {resolution_hours}h"
    )

    # Get precipitation data
    if precip_variable not in ds.data_vars:
        raise ValueError(
            f"Variable '{precip_variable}' not found in dataset. "
            f"Available variables: {list(ds.data_vars)}"
        )

    precip = ds[precip_variable]

    # Count valid timesteps per pixel
    valid_count = (~np.isnan(precip)).sum(dim='time')
    total_count = len(ds.time)
    valid_fraction = valid_count / total_count

    # Calculate accumulation: rate (mm/hr) * time (hr)
    # Sum over all timesteps, multiplying by temporal resolution
    daily_accum = (precip * resolution_hours).sum(dim='time', skipna=True)

    # Mask pixels with insufficient valid data
    daily_accum = daily_accum.where(valid_fraction >= min_valid_fraction)

    # Create output dataset
    date_str = pd.to_datetime(ds.time.values[0]).strftime('%Y-%m-%d')

    output_ds = xr.Dataset(
        {
            'daily_precipitation': daily_accum,
            'valid_fraction': valid_fraction,
        },
        attrs={
            'title': f'Daily Precipitation Accumulation - {date_str}',
            'source': f'{product_type.upper()} precipitation product',
            'temporal_resolution_hours': resolution_hours,
            'min_valid_fraction': min_valid_fraction,
            'units': 'mm/day',
            'processing_date': datetime.now().isoformat(),
            'source_variable': precip_variable,
            'source_timesteps': total_count,
        }
    )

    # Copy original variable attributes
    if precip.attrs:
        output_ds['daily_precipitation'].attrs.update({
            'long_name': f"Daily accumulated {precip.attrs.get('long_name', 'precipitation')}",
            'original_units': precip.attrs.get('units', 'mm/hr'),
            'units': 'mm/day'
        })

    # Copy gap-filling metadata if present
    if 'gap_fill_method' in ds.attrs:
        output_ds.attrs['gap_fill_method'] = ds.attrs['gap_fill_method']
        output_ds.attrs['gap_filled_count'] = ds.attrs.get('gap_filled_count', 0)

    # Add gap-filled fraction if available
    if 'gap_filled_mask' in ds.coords:
        gap_filled_count = ds.coords['gap_filled_mask'].sum().item()
        output_ds.attrs['gap_filled_fraction'] = gap_filled_count / total_count

    logger.info(
        f"Daily accumulation calculated: "
        f"mean={float(daily_accum.mean()):.2f} mm/day, "
        f"max={float(daily_accum.max()):.2f} mm/day"
    )

    return output_ds


def save_output(
    ds: xr.Dataset,
    output_path: str,
    encoding: Optional[Dict] = None,
    overwrite: bool = True
) -> str:
    """
    Save dataset to NetCDF file with proper metadata.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to save.
    output_path : str
        Output file path. Should end with '.nc' or '.nc4'.
    encoding : dict, optional
        Variable encoding settings. If None, uses defaults with compression.
    overwrite : bool, optional
        Whether to overwrite existing file. Default is True.

    Returns
    -------
    str
        Path to saved file.

    Raises
    ------
    FileExistsError
        If file exists and overwrite is False.

    Examples
    --------
    >>> ds = calculate_daily_accumulation(ds_filled, 'imerg')
    >>> save_output(ds, '/data/output/daily_2023-01-15.nc')
    """
    import os

    if not overwrite and os.path.exists(output_path):
        raise FileExistsError(f"File already exists: {output_path}")

    # Create directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")

    # Default encoding with compression (if netCDF4 is available)
    if encoding is None:
        try:
            # Try with compression (requires netCDF4 engine)
            encoding = {}
            for var in ds.data_vars:
                encoding[var] = {
                    'zlib': True,
                    'complevel': 4,
                    'dtype': 'float32'
                }
            ds.to_netcdf(output_path, encoding=encoding, engine='netcdf4')
        except (ImportError, ValueError):
            # Fall back to simple encoding without compression
            encoding = {}
            for var in ds.data_vars:
                encoding[var] = {'dtype': 'float32'}
            ds.to_netcdf(output_path, encoding=encoding)
    else:
        ds.to_netcdf(output_path, encoding=encoding)

    logger.info(f"Saving output to: {output_path}")
    logger.info(f"Successfully saved: {output_path}")

    return output_path


def compute_daily_accumulation(
    input_path: str,
    product_type: str,
    date: str,
    output_path: Optional[str] = None,
    precip_variable: Optional[str] = None,
    gap_fill_method: str = 'linear',
    max_gap_hours: Optional[float] = None,
    max_missing_fraction: float = 0.2,
    min_valid_fraction: float = 0.5,
    save_result: bool = True
) -> xr.Dataset:
    """
    Compute daily precipitation accumulation for a single date.

    This is the main convenience function that executes the complete pipeline:
    load -> check completeness -> gap fill -> accumulate -> save.

    Parameters
    ----------
    input_path : str
        Path to the directory containing input data files.
    product_type : str
        Type of precipitation product ('imerg', 'gsmap', or 'radar').
    date : str
        Date string in 'YYYY-MM-DD' format.
    output_path : str, optional
        Output file path. If None and save_result is True, auto-generates path.
    precip_variable : str, optional
        Name of the precipitation variable. Uses product default if None.
    gap_fill_method : str, optional
        Gap filling method. Default is 'linear'.
    max_gap_hours : float, optional
        Maximum gap size to interpolate.
    max_missing_fraction : float, optional
        Maximum fraction of missing timesteps allowed. Default is 0.2.
    min_valid_fraction : float, optional
        Minimum valid fraction for daily accumulation. Default is 0.5.
    save_result : bool, optional
        Whether to save the result to file. Default is True.

    Returns
    -------
    xr.Dataset
        Daily accumulation dataset.

    Raises
    ------
    ValueError
        If completeness check fails and too much data is missing.

    Examples
    --------
    >>> daily = compute_daily_accumulation(
    ...     input_path='/data/imerg/',
    ...     product_type='imerg',
    ...     date='2023-01-15',
    ...     output_path='/data/output/daily_2023-01-15.nc'
    ... )
    """
    logger.info(f"=== Starting daily accumulation pipeline for {date} ===")

    # Step 1: Load dataset
    ds = load_dataset(input_path, product_type, date, precip_variable)

    # Step 2: Check completeness
    is_complete, report = check_timestep_completeness(
        ds, product_type, date, max_missing_fraction
    )

    if not is_complete:
        raise ValueError(
            f"Dataset completeness check failed for {date}. "
            f"Missing fraction: {report['missing_fraction']:.1%} "
            f"(max allowed: {max_missing_fraction:.1%}). "
            f"Missing timesteps: {report['missing_timesteps']}"
        )

    # Step 3: Gap fill
    ds_filled = gap_fill(
        ds, product_type, date,
        method=gap_fill_method,
        max_gap_hours=max_gap_hours
    )

    # Step 4: Calculate daily accumulation
    daily_ds = calculate_daily_accumulation(
        ds_filled, product_type,
        precip_variable=precip_variable,
        min_valid_fraction=min_valid_fraction
    )

    # Step 5: Save output
    if save_result:
        if output_path is None:
            date_str = pd.to_datetime(date).strftime('%Y%m%d')
            output_path = f"daily_accumulation_{product_type}_{date_str}.nc"

        save_output(daily_ds, output_path)

    logger.info(f"=== Pipeline completed for {date} ===")

    return daily_ds


class DailyAccumulationPipeline:
    """
    Pipeline class for computing daily precipitation accumulation.

    This class provides an object-oriented interface for processing
    precipitation data over date ranges with consistent settings.

    Parameters
    ----------
    product_type : str
        Type of precipitation product ('imerg', 'gsmap', or 'radar').
    input_path : str
        Path to the directory containing input data files.
    output_path : str, optional
        Path to output directory. If None, outputs to current directory.
    precip_variable : str, optional
        Name of the precipitation variable.
    gap_fill_method : str, optional
        Gap filling method. Default is 'linear'.
    max_gap_hours : float, optional
        Maximum gap size to interpolate.
    max_missing_fraction : float, optional
        Maximum fraction of missing timesteps allowed.
    min_valid_fraction : float, optional
        Minimum valid fraction for daily accumulation.
    log_level : int, optional
        Logging level. Default is logging.INFO.

    Attributes
    ----------
    config : dict
        Product configuration settings.
    processing_log : list
        Log of processed dates and their status.

    Examples
    --------
    >>> pipeline = DailyAccumulationPipeline(
    ...     product_type='imerg',
    ...     input_path='/data/imerg/',
    ...     output_path='/data/output/'
    ... )
    >>> results = pipeline.process_date_range('2023-01-01', '2023-01-31')
    >>> pipeline.print_summary()
    """

    def __init__(
        self,
        product_type: str,
        input_path: str,
        output_path: Optional[str] = None,
        precip_variable: Optional[str] = None,
        gap_fill_method: str = 'linear',
        max_gap_hours: Optional[float] = None,
        max_missing_fraction: float = 0.2,
        min_valid_fraction: float = 0.5,
        log_level: int = logging.INFO
    ):
        if product_type not in PRODUCT_CONFIG:
            raise ValueError(
                f"Unknown product type '{product_type}'. "
                f"Supported types: {list(PRODUCT_CONFIG.keys())}"
            )

        self.product_type = product_type
        self.input_path = input_path
        self.output_path = output_path or '.'
        self.precip_variable = precip_variable
        self.gap_fill_method = gap_fill_method
        self.max_gap_hours = max_gap_hours
        self.max_missing_fraction = max_missing_fraction
        self.min_valid_fraction = min_valid_fraction

        self.config = PRODUCT_CONFIG[product_type]
        self.processing_log: List[Dict] = []

        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def process_date(self, date: str) -> Optional[xr.Dataset]:
        """
        Process a single date.

        Parameters
        ----------
        date : str
            Date string in 'YYYY-MM-DD' format.

        Returns
        -------
        xr.Dataset or None
            Daily accumulation dataset, or None if processing failed.
        """
        import os

        date_str = pd.to_datetime(date).strftime('%Y%m%d')
        output_file = os.path.join(
            self.output_path,
            f"daily_accumulation_{self.product_type}_{date_str}.nc"
        )

        log_entry = {
            'date': date,
            'status': 'pending',
            'output_file': output_file,
            'error': None
        }

        try:
            result = compute_daily_accumulation(
                input_path=self.input_path,
                product_type=self.product_type,
                date=date,
                output_path=output_file,
                precip_variable=self.precip_variable,
                gap_fill_method=self.gap_fill_method,
                max_gap_hours=self.max_gap_hours,
                max_missing_fraction=self.max_missing_fraction,
                min_valid_fraction=self.min_valid_fraction,
                save_result=True
            )
            log_entry['status'] = 'success'
            self.processing_log.append(log_entry)
            return result

        except Exception as e:
            log_entry['status'] = 'failed'
            log_entry['error'] = str(e)
            self.processing_log.append(log_entry)
            logger.error(f"Failed to process {date}: {e}")
            return None

    def process_date_range(
        self,
        start_date: str,
        end_date: str,
        skip_existing: bool = True
    ) -> Dict[str, xr.Dataset]:
        """
        Process a range of dates.

        Parameters
        ----------
        start_date : str
            Start date in 'YYYY-MM-DD' format.
        end_date : str
            End date in 'YYYY-MM-DD' format (inclusive).
        skip_existing : bool, optional
            Whether to skip dates with existing output files. Default is True.

        Returns
        -------
        dict
            Dictionary mapping dates to their result datasets.
        """
        import os

        dates = pd.date_range(start_date, end_date, freq='D')
        results = {}

        logger.info(
            f"Processing {len(dates)} dates from {start_date} to {end_date}"
        )

        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            output_file = os.path.join(
                self.output_path,
                f"daily_accumulation_{self.product_type}_{date.strftime('%Y%m%d')}.nc"
            )

            if skip_existing and os.path.exists(output_file):
                logger.info(f"Skipping {date_str} - output exists")
                self.processing_log.append({
                    'date': date_str,
                    'status': 'skipped',
                    'output_file': output_file,
                    'error': None
                })
                continue

            result = self.process_date(date_str)
            if result is not None:
                results[date_str] = result

        return results

    def print_summary(self):
        """Print processing summary."""
        total = len(self.processing_log)
        success = sum(1 for e in self.processing_log if e['status'] == 'success')
        failed = sum(1 for e in self.processing_log if e['status'] == 'failed')
        skipped = sum(1 for e in self.processing_log if e['status'] == 'skipped')

        print("\n" + "=" * 50)
        print("DAILY ACCUMULATION PIPELINE SUMMARY")
        print("=" * 50)
        print(f"Product Type: {self.product_type}")
        print(f"Input Path:   {self.input_path}")
        print(f"Output Path:  {self.output_path}")
        print("-" * 50)
        print(f"Total Dates:  {total}")
        print(f"Successful:   {success}")
        print(f"Failed:       {failed}")
        print(f"Skipped:      {skipped}")
        print("=" * 50)

        if failed > 0:
            print("\nFailed Dates:")
            for entry in self.processing_log:
                if entry['status'] == 'failed':
                    print(f"  {entry['date']}: {entry['error']}")

    def get_processing_log(self) -> pd.DataFrame:
        """
        Get processing log as DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with processing log entries.
        """
        return pd.DataFrame(self.processing_log)


# Example script usage (can be run as: python -m pacha.L2_processing.daily_accumulation)
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compute daily precipitation accumulation from rain intensity data'
    )
    parser.add_argument(
        'input_path',
        help='Path to input data directory'
    )
    parser.add_argument(
        'product_type',
        choices=['imerg', 'gsmap', 'radar'],
        help='Precipitation product type'
    )
    parser.add_argument(
        '--start-date',
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        help='End date (YYYY-MM-DD). If not provided, processes single day.'
    )
    parser.add_argument(
        '--output-path',
        default='.',
        help='Output directory path (default: current directory)'
    )
    parser.add_argument(
        '--precip-variable',
        help='Name of precipitation variable in dataset'
    )
    parser.add_argument(
        '--gap-fill-method',
        default='linear',
        choices=['linear', 'nearest', 'zero', 'previous'],
        help='Gap filling method (default: linear)'
    )
    parser.add_argument(
        '--max-missing-fraction',
        type=float,
        default=0.2,
        help='Maximum allowed missing data fraction (default: 0.2)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create pipeline
    pipeline = DailyAccumulationPipeline(
        product_type=args.product_type,
        input_path=args.input_path,
        output_path=args.output_path,
        precip_variable=args.precip_variable,
        gap_fill_method=args.gap_fill_method,
        max_missing_fraction=args.max_missing_fraction,
        log_level=log_level
    )

    # Process dates
    end_date = args.end_date or args.start_date
    pipeline.process_date_range(args.start_date, end_date)

    # Print summary
    pipeline.print_summary()

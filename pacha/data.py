"""Data loading, processing, and transformation utilities.

This module provides functions for loading precipitation data from various
file formats and sources, as well as utilities for data preprocessing
and transformation.

Functions:
    load_timeseries: Load time series precipitation data from CSV files.
    load_netcdf: Load gridded precipitation data from NetCDF files.
    interpolate_to_points: Interpolate gridded data to point locations.
    aggregate_temporal: Aggregate precipitation data to different time scales.

Example:
    Loading precipitation data from a CSV file::

        from pacha.data import load_timeseries

        # Load hourly rainfall data
        precip = load_timeseries(
            "rainfall.csv",
            time_column="datetime",
            value_column="precipitation_mm",
            units="mm"
        )

        # Check data quality
        print(f"Records: {len(precip.values)}")
        print(f"Missing: {precip.values.isna().sum()}")

"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from pacha.core import GriddedData, TimeSeriesData


def load_timeseries(
    filepath: str | Path,
    time_column: str = "time",
    value_column: str = "value",
    units: str = "mm",
    source: str | None = None,
    parse_dates: bool = True,
    **kwargs: Any,
) -> TimeSeriesData:
    """Load time series precipitation data from a CSV file.

    Reads precipitation data from a CSV file and returns a TimeSeriesData
    object with properly parsed timestamps and validated values.

    Args:
        filepath: Path to the CSV file containing precipitation data.
        time_column: Name of the column containing timestamps.
            Defaults to 'time'.
        value_column: Name of the column containing precipitation values.
            Defaults to 'value'.
        units: Units of the precipitation values (e.g., 'mm', 'mm/h').
            Defaults to 'mm'.
        source: Optional source identifier. If None, uses the filename.
            Defaults to None.
        parse_dates: Whether to parse the time column as datetime.
            Defaults to True.
        **kwargs: Additional keyword arguments passed to pandas.read_csv().

    Returns:
        TimeSeriesData object containing the loaded precipitation data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are not found in the file.
        KeyError: If specified columns are not found in the file.

    Example:
        >>> # Load data with default column names
        >>> precip = load_timeseries("rainfall.csv")
        >>>
        >>> # Load data with custom column names
        >>> precip = load_timeseries(
        ...     "weather_data.csv",
        ...     time_column="datetime",
        ...     value_column="precip_mm",
        ...     units="mm/h"
        ... )
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read the CSV file
    df = pd.read_csv(filepath, **kwargs)

    # Validate columns exist
    if time_column not in df.columns:
        raise KeyError(
            f"Time column '{time_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if value_column not in df.columns:
        raise KeyError(
            f"Value column '{value_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Parse timestamps
    if parse_dates:
        times = pd.to_datetime(df[time_column])
    else:
        times = pd.DatetimeIndex(df[time_column])

    # Get values
    values = df[value_column].values.astype(np.float64)

    # Set source if not provided
    if source is None:
        source = filepath.stem

    return TimeSeriesData(
        values=values,
        times=times,
        units=units,
        source=source,
    )


def load_netcdf(
    filepath: str | Path,
    variable: str,
    lat_var: str = "latitude",
    lon_var: str = "longitude",
    time_var: str | None = "time",
    units: str | None = None,
    source: str | None = None,
) -> GriddedData:
    """Load gridded precipitation data from a NetCDF file.

    Reads precipitation data from a NetCDF file using xarray and returns
    a GriddedData object with spatial coordinates and optional time dimension.

    Args:
        filepath: Path to the NetCDF file containing precipitation data.
        variable: Name of the precipitation variable in the file.
        lat_var: Name of the latitude coordinate variable.
            Defaults to 'latitude'.
        lon_var: Name of the longitude coordinate variable.
            Defaults to 'longitude'.
        time_var: Name of the time coordinate variable, or None if
            the data has no time dimension. Defaults to 'time'.
        units: Units of the precipitation values. If None, attempts to
            read from file attributes. Defaults to None.
        source: Optional source identifier. If None, uses the filename.
            Defaults to None.

    Returns:
        GriddedData object containing the loaded precipitation field.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        KeyError: If the specified variable is not found in the file.
        ImportError: If xarray is not installed.

    Example:
        >>> # Load 2D precipitation field
        >>> precip = load_netcdf(
        ...     "radar_precip.nc",
        ...     variable="precipitation",
        ...     lat_var="lat",
        ...     lon_var="lon"
        ... )
        >>>
        >>> # Load 3D precipitation data (with time)
        >>> precip = load_netcdf(
        ...     "satellite_precip.nc",
        ...     variable="precip",
        ...     time_var="time"
        ... )
    """
    try:
        import xarray as xr
    except ImportError as e:
        raise ImportError(
            "xarray is required for loading NetCDF files. "
            "Install it with: pip install xarray netcdf4"
        ) from e

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Open the dataset
    ds = xr.open_dataset(filepath)

    # Get the variable
    if variable not in ds.data_vars:
        raise KeyError(
            f"Variable '{variable}' not found. "
            f"Available variables: {list(ds.data_vars)}"
        )

    da = ds[variable]

    # Get coordinates
    lat = ds[lat_var].values if lat_var in ds.coords else None
    lon = ds[lon_var].values if lon_var in ds.coords else None

    # Get time coordinate if present
    times = None
    if time_var is not None and time_var in ds.coords:
        times = pd.DatetimeIndex(ds[time_var].values)

    # Get units from attributes if not provided
    if units is None:
        units = da.attrs.get("units", "mm")

    # Set source if not provided
    if source is None:
        source = filepath.stem

    # Close the dataset
    ds.close()

    return GriddedData(
        values=da.values,
        latitude=lat,
        longitude=lon,
        times=times,
        units=units,
        source=source,
    )


def interpolate_to_points(
    gridded: GriddedData,
    target_lat: npt.NDArray[np.floating[Any]],
    target_lon: npt.NDArray[np.floating[Any]],
    method: str = "linear",
) -> npt.NDArray[np.floating[Any]]:
    """Interpolate gridded precipitation data to point locations.

    Uses scipy interpolation to extract precipitation values at specific
    geographic coordinates from a gridded dataset.

    Args:
        gridded: GriddedData object containing the precipitation field.
        target_lat: Array of target latitude coordinates.
        target_lon: Array of target longitude coordinates.
        method: Interpolation method ('linear', 'nearest', 'cubic').
            Defaults to 'linear'.

    Returns:
        Array of interpolated precipitation values at target locations.

    Raises:
        ValueError: If gridded data lacks coordinate information.
        ValueError: If interpolation method is not supported.

    Example:
        >>> from pacha.data import load_netcdf, interpolate_to_points
        >>> import numpy as np
        >>>
        >>> # Load gridded data
        >>> grid = load_netcdf("radar.nc", variable="precip")
        >>>
        >>> # Define gauge locations
        >>> gauge_lat = np.array([-34.5, -34.8, -35.0])
        >>> gauge_lon = np.array([-55.5, -55.8, -56.0])
        >>>
        >>> # Extract values at gauge locations
        >>> values = interpolate_to_points(grid, gauge_lat, gauge_lon)
    """
    from scipy.interpolate import RegularGridInterpolator

    if gridded.latitude is None or gridded.longitude is None:
        raise ValueError(
            "Cannot interpolate: gridded data lacks coordinate information"
        )

    valid_methods = ("linear", "nearest", "cubic")
    if method not in valid_methods:
        raise ValueError(
            f"Invalid interpolation method '{method}'. "
            f"Choose from: {valid_methods}"
        )

    # Handle 2D case
    if gridded.values.ndim == 2:
        interpolator = RegularGridInterpolator(
            (gridded.latitude, gridded.longitude),
            gridded.values,
            method=method,
            bounds_error=False,
            fill_value=np.nan,
        )
        points = np.column_stack([target_lat, target_lon])
        return interpolator(points)

    # Handle 3D case (time, lat, lon)
    if gridded.values.ndim == 3:
        n_times = gridded.values.shape[0]
        n_points = len(target_lat)
        result = np.empty((n_times, n_points))

        for t in range(n_times):
            interpolator = RegularGridInterpolator(
                (gridded.latitude, gridded.longitude),
                gridded.values[t],
                method=method,
                bounds_error=False,
                fill_value=np.nan,
            )
            points = np.column_stack([target_lat, target_lon])
            result[t] = interpolator(points)

        return result

    raise ValueError(
        f"Unsupported array dimensions: {gridded.values.ndim}. "
        "Expected 2D (lat, lon) or 3D (time, lat, lon)."
    )


def aggregate_temporal(
    data: TimeSeriesData,
    target_freq: str,
    method: str = "sum",
    min_valid_ratio: float = 0.8,
) -> TimeSeriesData:
    """Aggregate precipitation time series to a different temporal resolution.

    Resamples time series data to a coarser temporal resolution (e.g.,
    hourly to daily) with quality control based on data availability.

    Args:
        data: TimeSeriesData object to aggregate.
        target_freq: Target frequency string (e.g., 'D' for daily,
            'W' for weekly, 'M' for monthly).
        method: Aggregation method ('sum', 'mean', 'max').
            Defaults to 'sum'.
        min_valid_ratio: Minimum ratio of valid (non-NaN) values required
            to compute aggregated value. Values below this threshold result
            in NaN. Defaults to 0.8 (80%).

    Returns:
        TimeSeriesData object with aggregated values.

    Raises:
        ValueError: If data has no time information.
        ValueError: If method is not supported.

    Example:
        >>> from pacha.data import load_timeseries, aggregate_temporal
        >>>
        >>> # Load hourly data
        >>> hourly = load_timeseries("hourly_rain.csv")
        >>>
        >>> # Aggregate to daily totals
        >>> daily = aggregate_temporal(hourly, "D", method="sum")
        >>>
        >>> # Aggregate to monthly means
        >>> monthly = aggregate_temporal(hourly, "M", method="mean")
    """
    if data.times is None:
        raise ValueError("Cannot aggregate: time series has no time information")

    valid_methods = ("sum", "mean", "max")
    if method not in valid_methods:
        raise ValueError(
            f"Invalid aggregation method '{method}'. " f"Choose from: {valid_methods}"
        )

    # Create pandas Series for resampling
    series = data.to_series()

    # Get resampler
    resampler = series.resample(target_freq)

    # Compute aggregation with quality control
    def aggregate_with_qc(group: pd.Series) -> float:
        """Aggregate group with quality control."""
        valid_count = group.notna().sum()
        total_count = len(group)

        if total_count == 0:
            return np.nan

        valid_ratio = valid_count / total_count
        if valid_ratio < min_valid_ratio:
            return np.nan

        if method == "sum":
            return float(group.sum())
        elif method == "mean":
            return float(group.mean())
        else:  # max
            return float(group.max())

    aggregated = resampler.apply(aggregate_with_qc)

    return TimeSeriesData(
        values=aggregated.values,
        times=aggregated.index,
        units=data.units,
        source=data.source,
        metadata={
            **data.metadata,
            "aggregation_method": method,
            "aggregation_freq": target_freq,
            "min_valid_ratio": min_valid_ratio,
        },
    )


def create_synthetic_timeseries(
    start: str | datetime,
    end: str | datetime,
    freq: str = "H",
    mean_intensity: float = 1.0,
    wet_fraction: float = 0.1,
    seed: int | None = None,
) -> TimeSeriesData:
    """Create synthetic precipitation time series for testing.

    Generates a synthetic precipitation time series using an exponential
    distribution for wet periods and a specified wet/dry fraction.

    Args:
        start: Start date/time of the series.
        end: End date/time of the series.
        freq: Time frequency (e.g., 'H' for hourly, 'D' for daily).
            Defaults to 'H'.
        mean_intensity: Mean precipitation intensity during wet periods.
            Defaults to 1.0.
        wet_fraction: Fraction of time steps with precipitation.
            Defaults to 0.1 (10%).
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        TimeSeriesData object with synthetic precipitation values.

    Example:
        >>> from pacha.data import create_synthetic_timeseries
        >>>
        >>> # Create one year of hourly synthetic data
        >>> synthetic = create_synthetic_timeseries(
        ...     start="2024-01-01",
        ...     end="2024-12-31",
        ...     freq="h",
        ...     mean_intensity=2.0,
        ...     wet_fraction=0.05
        ... )
    """
    if seed is not None:
        np.random.seed(seed)

    # Create time index
    times = pd.date_range(start=start, end=end, freq=freq)
    n_times = len(times)

    # Generate wet/dry mask
    is_wet = np.random.random(n_times) < wet_fraction

    # Generate precipitation values
    values = np.zeros(n_times)
    n_wet = is_wet.sum()
    if n_wet > 0:
        values[is_wet] = np.random.exponential(scale=mean_intensity, size=n_wet)

    return TimeSeriesData(
        values=values,
        times=times,
        units="mm" if freq in ["D", "W", "M"] else "mm/h",
        source="synthetic",
        metadata={
            "mean_intensity": mean_intensity,
            "wet_fraction": wet_fraction,
            "seed": seed,
        },
    )


__all__ = [
    "load_timeseries",
    "load_netcdf",
    "interpolate_to_points",
    "aggregate_temporal",
    "create_synthetic_timeseries",
]

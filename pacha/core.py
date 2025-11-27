"""Core functionality and base classes for precipitation data handling.

This module provides the foundational classes and functions for working
with precipitation data in PACHA. It includes data structures for
representing precipitation time series, spatial grids, and metadata.

Classes:
    PrecipitationData: Base class for precipitation data representation.
    TimeSeriesData: Class for handling time series precipitation data.
    GriddedData: Class for handling gridded/spatial precipitation data.

Functions:
    validate_precipitation: Validate precipitation values for physical consistency.
    mask_invalid: Mask invalid or missing precipitation values.

Example:
    Creating and working with precipitation time series::

        from pacha.core import TimeSeriesData
        import numpy as np

        # Create sample precipitation data
        values = np.array([0.0, 1.2, 3.5, 0.8, 0.0])
        times = pd.date_range("2024-01-01", periods=5, freq="h")

        # Create TimeSeriesData object
        precip = TimeSeriesData(
            values=values,
            times=times,
            units="mm/h",
            source="rain_gauge"
        )

        # Access statistics
        print(f"Total: {precip.total()} mm")
        print(f"Max rate: {precip.max()} mm/h")

"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class PrecipitationData:
    """Base class for precipitation data representation.

    This class provides a common interface for all precipitation data types
    in PACHA. It handles data validation, unit conversion, and basic
    statistical operations.

    Attributes:
        values: Array of precipitation values.
        units: Units of the precipitation values (e.g., 'mm/h', 'mm').
        source: Source identifier for the precipitation data.
        metadata: Additional metadata dictionary.

    Example:
        Creating a basic precipitation data object::

            data = PrecipitationData(
                values=np.array([0.0, 1.5, 2.3]),
                units="mm/h",
                source="gauge_001"
            )
    """

    values: npt.NDArray[np.floating[Any]]
    units: str = "mm"
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate precipitation data after initialization.

        Raises:
            ValueError: If precipitation values contain negative values
                that are not marked as missing data.
        """
        self.values = np.asarray(self.values, dtype=np.float64)
        if not _validate_precipitation(self.values):
            raise ValueError(
                "Precipitation values must be non-negative. "
                "Use np.nan for missing values."
            )

    def total(self) -> float:
        """Calculate the total accumulated precipitation.

        Returns:
            Total precipitation sum, excluding NaN values.

        Example:
            >>> data = PrecipitationData(values=np.array([1.0, 2.0, 3.0]))
            >>> data.total()
            6.0
        """
        return float(np.nansum(self.values))

    def max(self) -> float:
        """Get the maximum precipitation value.

        Returns:
            Maximum precipitation value, excluding NaN values.

        Example:
            >>> data = PrecipitationData(values=np.array([1.0, 5.0, 2.0]))
            >>> data.max()
            5.0
        """
        return float(np.nanmax(self.values))

    def min(self) -> float:
        """Get the minimum precipitation value.

        Returns:
            Minimum precipitation value, excluding NaN values.

        Example:
            >>> data = PrecipitationData(values=np.array([1.0, 5.0, 2.0]))
            >>> data.min()
            1.0
        """
        return float(np.nanmin(self.values))

    def mean(self) -> float:
        """Calculate the mean precipitation value.

        Returns:
            Mean precipitation value, excluding NaN values.

        Example:
            >>> data = PrecipitationData(values=np.array([1.0, 2.0, 3.0]))
            >>> data.mean()
            2.0
        """
        return float(np.nanmean(self.values))

    def std(self) -> float:
        """Calculate the standard deviation of precipitation values.

        Returns:
            Standard deviation of precipitation values, excluding NaN values.

        Example:
            >>> values = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
            >>> data = PrecipitationData(values=values)
            >>> round(data.std(), 2)
            2.0
        """
        return float(np.nanstd(self.values))


@dataclass
class TimeSeriesData(PrecipitationData):
    """Class for handling time series precipitation data.

    Extends PrecipitationData with temporal information, providing
    functionality for time-based operations on precipitation data.

    Attributes:
        values: Array of precipitation values.
        times: DatetimeIndex or array of timestamps for each value.
        units: Units of the precipitation values.
        source: Source identifier for the precipitation data.
        metadata: Additional metadata dictionary.

    Example:
        Creating a time series of hourly precipitation::

            import pandas as pd
            import numpy as np

            times = pd.date_range("2024-01-01", periods=24, freq="h")
            values = np.random.exponential(scale=0.5, size=24)

            ts = TimeSeriesData(
                values=values,
                times=times,
                units="mm/h",
                source="automatic_gauge"
            )

            # Get daily totals
            daily = ts.resample("D")
    """

    times: pd.DatetimeIndex | list[datetime] | None = None

    def __post_init__(self) -> None:
        """Initialize time series data with validation.

        Raises:
            ValueError: If times and values have different lengths.
        """
        super().__post_init__()
        if self.times is not None:
            if not isinstance(self.times, pd.DatetimeIndex):
                self.times = pd.DatetimeIndex(self.times)
            if len(self.times) != len(self.values):
                raise ValueError(
                    f"Length of times ({len(self.times)}) must match "
                    f"length of values ({len(self.values)})"
                )

    def to_series(self) -> pd.Series:
        """Convert to a pandas Series with datetime index.

        Returns:
            Pandas Series with precipitation values and datetime index.

        Raises:
            ValueError: If times are not set.

        Example:
            >>> ts = TimeSeriesData(
            ...     values=np.array([1.0, 2.0]),
            ...     times=pd.date_range("2024-01-01", periods=2, freq="h")
            ... )
            >>> series = ts.to_series()
            >>> isinstance(series, pd.Series)
            True
        """
        if self.times is None:
            raise ValueError("Cannot convert to Series: times are not set")
        return pd.Series(self.values, index=self.times, name=self.source or "precip")

    def resample(self, freq: str, method: str = "sum") -> TimeSeriesData:
        """Resample the time series to a different frequency.

        Args:
            freq: Target frequency string (e.g., 'D' for daily, 'H' for hourly).
            method: Aggregation method ('sum', 'mean', 'max', 'min').
                Defaults to 'sum'.

        Returns:
            New TimeSeriesData object with resampled values.

        Raises:
            ValueError: If times are not set or method is invalid.

        Example:
            >>> ts = TimeSeriesData(
            ...     values=np.array([1.0, 2.0, 3.0, 4.0]),
            ...     times=pd.date_range("2024-01-01", periods=4, freq="6h")
            ... )
            >>> daily = ts.resample("D", method="sum")
            >>> daily.total()
            10.0
        """
        if self.times is None:
            raise ValueError("Cannot resample: times are not set")

        series = self.to_series()
        resampler = series.resample(freq)

        if method == "sum":
            resampled = resampler.sum()
        elif method == "mean":
            resampled = resampler.mean()
        elif method == "max":
            resampled = resampler.max()
        elif method == "min":
            resampled = resampler.min()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return TimeSeriesData(
            values=resampled.values,
            times=resampled.index,
            units=self.units,
            source=self.source,
            metadata=self.metadata.copy(),
        )


@dataclass
class GriddedData(PrecipitationData):
    """Class for handling gridded/spatial precipitation data.

    Extends PrecipitationData with spatial information for 2D or 3D
    precipitation fields from radar, satellite, or model data.

    Attributes:
        values: 2D or 3D array of precipitation values (y, x) or (time, y, x).
        latitude: Array of latitude values for each grid row.
        longitude: Array of longitude values for each grid column.
        times: Optional datetime index for temporal dimension.
        crs: Coordinate reference system string (e.g., 'EPSG:4326').
        units: Units of the precipitation values.
        source: Source identifier for the precipitation data.
        metadata: Additional metadata dictionary.

    Example:
        Creating a gridded precipitation field::

            import numpy as np

            # Create a 10x10 precipitation grid
            values = np.random.exponential(scale=1.0, size=(10, 10))
            lat = np.linspace(-35, -34, 10)
            lon = np.linspace(-56, -55, 10)

            grid = GriddedData(
                values=values,
                latitude=lat,
                longitude=lon,
                units="mm/h",
                source="radar"
            )
    """

    latitude: npt.NDArray[np.floating[Any]] | None = None
    longitude: npt.NDArray[np.floating[Any]] | None = None
    times: pd.DatetimeIndex | None = None
    crs: str = "EPSG:4326"

    def __post_init__(self) -> None:
        """Initialize gridded data with validation.

        Raises:
            ValueError: If latitude/longitude dimensions don't match values.
        """
        self.values = np.asarray(self.values, dtype=np.float64)
        if not _validate_precipitation(self.values):
            raise ValueError(
                "Precipitation values must be non-negative. "
                "Use np.nan for missing values."
            )

        if self.latitude is not None:
            self.latitude = np.asarray(self.latitude, dtype=np.float64)
        if self.longitude is not None:
            self.longitude = np.asarray(self.longitude, dtype=np.float64)

        # Validate dimensions
        if self.latitude is not None and self.longitude is not None:
            shape = self.values.shape
            if len(shape) == 2:
                if len(self.latitude) != shape[0]:
                    raise ValueError(
                        f"Latitude length ({len(self.latitude)}) must match "
                        f"first dimension of values ({shape[0]})"
                    )
                if len(self.longitude) != shape[1]:
                    raise ValueError(
                        f"Longitude length ({len(self.longitude)}) must match "
                        f"second dimension of values ({shape[1]})"
                    )
            elif len(shape) == 3:
                if len(self.latitude) != shape[1]:
                    raise ValueError(
                        f"Latitude length ({len(self.latitude)}) must match "
                        f"second dimension of values ({shape[1]})"
                    )
                if len(self.longitude) != shape[2]:
                    raise ValueError(
                        f"Longitude length ({len(self.longitude)}) must match "
                        f"third dimension of values ({shape[2]})"
                    )

    def spatial_mean(self) -> float:
        """Calculate the spatial mean of the precipitation field.

        Returns:
            Spatial mean precipitation value.

        Example:
            >>> grid = GriddedData(values=np.array([[1.0, 2.0], [3.0, 4.0]]))
            >>> grid.spatial_mean()
            2.5
        """
        return float(np.nanmean(self.values))

    def get_extent(self) -> tuple[float, float, float, float] | None:
        """Get the spatial extent of the gridded data.

        Returns:
            Tuple of (lon_min, lon_max, lat_min, lat_max) or None
            if coordinates are not set.

        Example:
            >>> grid = GriddedData(
            ...     values=np.zeros((2, 2)),
            ...     latitude=np.array([-35, -34]),
            ...     longitude=np.array([-56, -55])
            ... )
            >>> grid.get_extent()
            (-56.0, -55.0, -35.0, -34.0)
        """
        if self.latitude is None or self.longitude is None:
            return None
        return (
            float(np.min(self.longitude)),
            float(np.max(self.longitude)),
            float(np.min(self.latitude)),
            float(np.max(self.latitude)),
        )


def _validate_precipitation(values: npt.NDArray[np.floating[Any]]) -> bool:
    """Validate that precipitation values are physically consistent.

    Checks that all non-NaN values are non-negative, as negative
    precipitation is physically impossible.

    Args:
        values: Array of precipitation values to validate.

    Returns:
        True if all values are valid (non-negative or NaN),
        False otherwise.

    Example:
        >>> _validate_precipitation(np.array([0.0, 1.0, np.nan]))
        True
        >>> _validate_precipitation(np.array([0.0, -1.0, 2.0]))
        False
    """
    # Get non-NaN values
    valid_values = values[~np.isnan(values)]
    # Check for negative values
    return bool(np.all(valid_values >= 0))


def validate_precipitation(values: npt.NDArray[np.floating[Any]]) -> bool:
    """Validate that precipitation values are physically consistent.

    Public function to check that precipitation values are non-negative,
    as negative precipitation is physically impossible.

    Args:
        values: Array of precipitation values to validate.

    Returns:
        True if all values are valid (non-negative or NaN),
        False otherwise.

    Example:
        >>> from pacha.core import validate_precipitation
        >>> import numpy as np
        >>> validate_precipitation(np.array([0.0, 1.5, 2.3]))
        True
        >>> validate_precipitation(np.array([0.0, -1.0]))
        False
    """
    return _validate_precipitation(values)


def mask_invalid(
    values: npt.NDArray[np.floating[Any]],
    threshold: float | None = None,
    mask_zeros: bool = False,
) -> npt.NDArray[np.floating[Any]]:
    """Mask invalid or missing precipitation values.

    Replaces invalid values (negative, above threshold, or optionally zeros)
    with NaN for consistent handling of missing data.

    Args:
        values: Array of precipitation values.
        threshold: Optional maximum valid value. Values above this
            threshold are masked. Defaults to None.
        mask_zeros: If True, also mask zero values. Useful for
            log transformations. Defaults to False.

    Returns:
        Array with invalid values replaced by NaN.

    Example:
        >>> from pacha.core import mask_invalid
        >>> import numpy as np
        >>> values = np.array([-1.0, 0.0, 5.0, 1000.0])
        >>> mask_invalid(values, threshold=100.0)
        array([nan,  0.,  5., nan])
        >>> mask_invalid(values, threshold=100.0, mask_zeros=True)
        array([nan, nan,  5., nan])
    """
    result = values.copy().astype(np.float64)

    # Mask negative values
    result[result < 0] = np.nan

    # Mask values above threshold
    if threshold is not None:
        result[result > threshold] = np.nan

    # Optionally mask zeros
    if mask_zeros:
        result[result == 0] = np.nan

    return result


__all__ = [
    "PrecipitationData",
    "TimeSeriesData",
    "GriddedData",
    "validate_precipitation",
    "mask_invalid",
]

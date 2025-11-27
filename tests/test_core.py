"""Tests for pacha.core module."""

import numpy as np
import pandas as pd
import pytest

from pacha.core import (
    GriddedData,
    PrecipitationData,
    TimeSeriesData,
    mask_invalid,
    validate_precipitation,
)


class TestPrecipitationData:
    """Tests for the PrecipitationData class."""

    def test_create_basic(self) -> None:
        """Test creating a basic PrecipitationData object."""
        values = np.array([0.0, 1.0, 2.0, 3.0])
        data = PrecipitationData(values=values, units="mm")
        assert len(data.values) == 4
        assert data.units == "mm"

    def test_reject_negative_values(self) -> None:
        """Test that negative values raise an error."""
        values = np.array([0.0, -1.0, 2.0])
        with pytest.raises(ValueError, match="non-negative"):
            PrecipitationData(values=values)

    def test_allow_nan_values(self) -> None:
        """Test that NaN values are allowed."""
        values = np.array([0.0, np.nan, 2.0])
        data = PrecipitationData(values=values)
        assert np.isnan(data.values[1])

    def test_total(self) -> None:
        """Test total precipitation calculation."""
        values = np.array([1.0, 2.0, 3.0])
        data = PrecipitationData(values=values)
        assert data.total() == 6.0

    def test_total_with_nan(self) -> None:
        """Test total ignores NaN values."""
        values = np.array([1.0, np.nan, 3.0])
        data = PrecipitationData(values=values)
        assert data.total() == 4.0

    def test_max(self) -> None:
        """Test max value extraction."""
        values = np.array([1.0, 5.0, 2.0])
        data = PrecipitationData(values=values)
        assert data.max() == 5.0

    def test_min(self) -> None:
        """Test min value extraction."""
        values = np.array([1.0, 5.0, 2.0])
        data = PrecipitationData(values=values)
        assert data.min() == 1.0

    def test_mean(self) -> None:
        """Test mean calculation."""
        values = np.array([1.0, 2.0, 3.0])
        data = PrecipitationData(values=values)
        assert data.mean() == 2.0

    def test_std(self) -> None:
        """Test standard deviation calculation."""
        values = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        data = PrecipitationData(values=values)
        assert round(data.std(), 2) == 2.0


class TestTimeSeriesData:
    """Tests for the TimeSeriesData class."""

    def test_create_with_times(self) -> None:
        """Test creating TimeSeriesData with timestamps."""
        values = np.array([0.0, 1.0, 2.0])
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        ts = TimeSeriesData(values=values, times=times)
        assert len(ts.times) == 3

    def test_mismatched_lengths_raises(self) -> None:
        """Test that mismatched lengths raise an error."""
        values = np.array([0.0, 1.0, 2.0])
        times = pd.date_range("2024-01-01", periods=2, freq="h")
        with pytest.raises(ValueError, match="Length"):
            TimeSeriesData(values=values, times=times)

    def test_to_series(self) -> None:
        """Test conversion to pandas Series."""
        values = np.array([0.0, 1.0, 2.0])
        times = pd.date_range("2024-01-01", periods=3, freq="h")
        ts = TimeSeriesData(values=values, times=times, source="test")
        series = ts.to_series()
        assert isinstance(series, pd.Series)
        assert series.name == "test"

    def test_resample_sum(self) -> None:
        """Test resampling with sum aggregation."""
        values = np.array([1.0, 2.0, 3.0, 4.0])
        times = pd.date_range("2024-01-01", periods=4, freq="6h")
        ts = TimeSeriesData(values=values, times=times)
        daily = ts.resample("D", method="sum")
        assert daily.total() == 10.0


class TestGriddedData:
    """Tests for the GriddedData class."""

    def test_create_2d(self) -> None:
        """Test creating 2D GriddedData."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        lat = np.array([-35.0, -34.0])
        lon = np.array([-56.0, -55.0])
        grid = GriddedData(values=values, latitude=lat, longitude=lon)
        assert grid.values.shape == (2, 2)

    def test_spatial_mean(self) -> None:
        """Test spatial mean calculation."""
        values = np.array([[1.0, 2.0], [3.0, 4.0]])
        grid = GriddedData(values=values)
        assert grid.spatial_mean() == 2.5

    def test_get_extent(self) -> None:
        """Test getting spatial extent."""
        values = np.zeros((2, 2))
        lat = np.array([-35.0, -34.0])
        lon = np.array([-56.0, -55.0])
        grid = GriddedData(values=values, latitude=lat, longitude=lon)
        extent = grid.get_extent()
        assert extent == (-56.0, -55.0, -35.0, -34.0)

    def test_dimension_mismatch_raises(self) -> None:
        """Test that dimension mismatch raises an error."""
        values = np.zeros((2, 3))
        lat = np.array([-35.0])  # Wrong size
        lon = np.array([-56.0, -55.0, -54.0])
        with pytest.raises(ValueError, match="Latitude length"):
            GriddedData(values=values, latitude=lat, longitude=lon)


class TestValidatePrecipitation:
    """Tests for the validate_precipitation function."""

    def test_valid_values(self) -> None:
        """Test validation of valid values."""
        values = np.array([0.0, 1.5, 2.3])
        assert validate_precipitation(values) is True

    def test_invalid_negative(self) -> None:
        """Test detection of invalid negative values."""
        values = np.array([0.0, -1.0, 2.0])
        assert validate_precipitation(values) is False

    def test_nan_allowed(self) -> None:
        """Test that NaN values are considered valid."""
        values = np.array([0.0, np.nan, 2.0])
        assert validate_precipitation(values) is True


class TestMaskInvalid:
    """Tests for the mask_invalid function."""

    def test_mask_negative(self) -> None:
        """Test masking negative values."""
        values = np.array([-1.0, 0.0, 5.0])
        result = mask_invalid(values)
        assert np.isnan(result[0])
        assert result[1] == 0.0
        assert result[2] == 5.0

    def test_mask_threshold(self) -> None:
        """Test masking values above threshold."""
        values = np.array([0.0, 5.0, 1000.0])
        result = mask_invalid(values, threshold=100.0)
        assert result[0] == 0.0
        assert result[1] == 5.0
        assert np.isnan(result[2])

    def test_mask_zeros(self) -> None:
        """Test optionally masking zeros."""
        values = np.array([0.0, 1.0, 2.0])
        result = mask_invalid(values, mask_zeros=True)
        assert np.isnan(result[0])
        assert result[1] == 1.0
        assert result[2] == 2.0

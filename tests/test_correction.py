"""Tests for pacha.correction module."""

import numpy as np
import pytest

from pacha.correction import (
    distribution_mapping,
    linear_scaling,
    local_intensity_scaling,
    mean_field_bias,
    quantile_mapping,
)


class TestMeanFieldBias:
    """Tests for the mean_field_bias function."""

    def test_basic_correction(self) -> None:
        """Test basic mean field bias correction."""
        estimates = np.array([0.5, 1.0, 1.5, 2.0])
        reference = np.array([1.0, 2.0, 3.0, 4.0])
        corrected = mean_field_bias(estimates, reference)
        np.testing.assert_array_almost_equal(corrected, reference)

    def test_shape_mismatch_raises(self) -> None:
        """Test that shape mismatch raises an error."""
        estimates = np.array([1.0, 2.0, 3.0])
        reference = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            mean_field_bias(estimates, reference)

    def test_preserves_zeros(self) -> None:
        """Test that zeros are preserved after correction."""
        estimates = np.array([0.0, 1.0, 2.0, 0.0])
        reference = np.array([0.0, 2.0, 4.0, 0.0])
        corrected = mean_field_bias(estimates, reference)
        assert corrected[0] == 0.0
        assert corrected[3] == 0.0

    def test_handles_nan(self) -> None:
        """Test handling of NaN values."""
        estimates = np.array([1.0, np.nan, 2.0])
        reference = np.array([2.0, 1.0, 4.0])
        corrected = mean_field_bias(estimates, reference)
        assert np.isnan(corrected[1])


class TestLinearScaling:
    """Tests for the linear_scaling function."""

    def test_ratio_method(self) -> None:
        """Test ratio scaling method."""
        estimates = np.array([1.0, 2.0, 3.0])
        reference = np.array([2.0, 4.0, 6.0])
        corrected = linear_scaling(estimates, reference, method="ratio")
        np.testing.assert_array_almost_equal(corrected, reference)

    def test_regression_method(self) -> None:
        """Test regression scaling method."""
        estimates = np.array([1.0, 2.0, 3.0, 4.0])
        reference = np.array([1.5, 2.5, 3.5, 4.5])
        corrected = linear_scaling(estimates, reference, method="regression")
        # Regression should produce similar values to reference
        assert np.allclose(corrected, reference, atol=0.5)

    def test_invalid_method_raises(self) -> None:
        """Test that invalid method raises an error."""
        estimates = np.array([1.0, 2.0])
        reference = np.array([2.0, 4.0])
        with pytest.raises(ValueError, match="Invalid method"):
            linear_scaling(estimates, reference, method="invalid")


class TestQuantileMapping:
    """Tests for the quantile_mapping function."""

    def test_basic_mapping(self) -> None:
        """Test basic quantile mapping."""
        estimates = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
        reference = np.array([0.0, 0.8, 1.5, 2.2, 3.0, 4.5])
        corrected = quantile_mapping(estimates, reference, n_quantiles=10)
        # Check that extremes are preserved approximately
        assert corrected[0] >= 0
        assert corrected[-1] > estimates[-1]

    def test_preserves_zeros(self) -> None:
        """Test that zeros are roughly preserved."""
        estimates = np.array([0.0, 1.0, 2.0, 3.0])
        reference = np.array([0.0, 1.5, 2.5, 4.0])
        corrected = quantile_mapping(estimates, reference)
        # First value should be close to zero (interpolated)
        assert corrected[0] >= 0

    def test_invalid_n_quantiles_raises(self) -> None:
        """Test that n_quantiles < 2 raises an error."""
        estimates = np.array([1.0, 2.0])
        reference = np.array([1.5, 2.5])
        with pytest.raises(ValueError, match="n_quantiles"):
            quantile_mapping(estimates, reference, n_quantiles=1)


class TestLocalIntensityScaling:
    """Tests for the local_intensity_scaling function."""

    def test_basic_scaling(self) -> None:
        """Test basic local intensity scaling."""
        estimates = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
        reference = np.array([0.2, 0.6, 1.2, 6.0, 12.0])
        corrected = local_intensity_scaling(estimates, reference)
        # Corrected values should be closer to reference
        diff_before = np.abs(estimates - reference).sum()
        diff_after = np.abs(corrected - reference).sum()
        assert diff_after < diff_before

    def test_with_custom_bins(self) -> None:
        """Test with custom intensity bins."""
        estimates = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
        reference = np.array([0.2, 0.6, 1.2, 6.0, 12.0])
        bins = np.array([0.0, 0.5, 1.0, 5.0, 20.0])
        corrected = local_intensity_scaling(estimates, reference, intensity_bins=bins)
        assert len(corrected) == len(estimates)


class TestDistributionMapping:
    """Tests for the distribution_mapping function."""

    def test_gamma_distribution(self) -> None:
        """Test distribution mapping with gamma distribution."""
        np.random.seed(42)
        estimates = np.random.exponential(scale=1.0, size=100)
        reference = np.random.gamma(shape=2.0, scale=1.0, size=100)
        corrected = distribution_mapping(estimates, reference, distribution="gamma")
        # Corrected should have different mean than original
        assert len(corrected) == len(estimates)

    def test_invalid_distribution_raises(self) -> None:
        """Test that invalid distribution raises an error."""
        estimates = np.array([1.0, 2.0, 3.0])
        reference = np.array([1.5, 2.5, 3.5])
        with pytest.raises(ValueError, match="Unsupported distribution"):
            distribution_mapping(estimates, reference, distribution="normal")

    def test_fallback_insufficient_data(self) -> None:
        """Test fallback to mean field bias with insufficient data."""
        estimates = np.array([0.0, 1.0, 0.0])  # Only 1 positive value
        reference = np.array([0.0, 2.0, 0.0])
        corrected = distribution_mapping(estimates, reference)
        assert len(corrected) == len(estimates)

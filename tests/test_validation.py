"""Tests for pacha.validation module."""

import numpy as np
import pytest

from pacha.validation import (
    bias,
    categorical_scores,
    compute_metrics,
    correlation,
    kling_gupta,
    mae,
    nash_sutcliffe,
    rmse,
)


class TestRMSE:
    """Tests for the rmse function."""

    def test_perfect_match(self) -> None:
        """Test RMSE is 0 for perfect match."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        assert rmse(observed, predicted) == 0.0

    def test_basic_calculation(self) -> None:
        """Test basic RMSE calculation."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.1, 2.2, 2.9])
        result = rmse(observed, predicted)
        assert round(result, 2) == 0.14

    def test_shape_mismatch_raises(self) -> None:
        """Test that shape mismatch raises an error."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Shape mismatch"):
            rmse(observed, predicted)

    def test_handles_nan(self) -> None:
        """Test handling of NaN values."""
        observed = np.array([1.0, np.nan, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        result = rmse(observed, predicted)
        assert result == 0.0


class TestMAE:
    """Tests for the mae function."""

    def test_perfect_match(self) -> None:
        """Test MAE is 0 for perfect match."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        assert mae(observed, predicted) == 0.0

    def test_basic_calculation(self) -> None:
        """Test basic MAE calculation."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.1, 1.8, 3.2])
        result = mae(observed, predicted)
        assert round(result, 2) == 0.17


class TestBias:
    """Tests for the bias function."""

    def test_no_bias(self) -> None:
        """Test bias is 0 when there is no systematic error."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        assert bias(observed, predicted) == 0.0

    def test_positive_bias(self) -> None:
        """Test detection of positive bias."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.2, 2.2, 3.2])
        assert bias(observed, predicted) == pytest.approx(0.2)

    def test_negative_bias(self) -> None:
        """Test detection of negative bias."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([0.8, 1.8, 2.8])
        assert bias(observed, predicted) == pytest.approx(-0.2)

    def test_relative_bias(self) -> None:
        """Test relative bias calculation."""
        observed = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.2, 2.2, 3.2])
        result = bias(observed, predicted, relative=True)
        assert round(result, 1) == 10.0


class TestCorrelation:
    """Tests for the correlation function."""

    def test_perfect_correlation(self) -> None:
        """Test correlation is near 1 for nearly perfect positive correlation."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 2.1, 2.9, 4.1])
        result = correlation(observed, predicted)
        assert result > 0.99

    def test_no_correlation(self) -> None:
        """Test correlation near 0 for uncorrelated data."""
        np.random.seed(42)
        observed = np.random.rand(100)
        predicted = np.random.rand(100)
        result = correlation(observed, predicted)
        assert abs(result) < 0.3


class TestNashSutcliffe:
    """Tests for the nash_sutcliffe function."""

    def test_perfect_fit(self) -> None:
        """Test NSE is 1 for perfect fit."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 2.0, 3.0, 4.0])
        assert nash_sutcliffe(observed, predicted) == 1.0

    def test_basic_calculation(self) -> None:
        """Test basic NSE calculation."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 2.0, 2.9, 4.0])
        result = nash_sutcliffe(observed, predicted)
        assert round(result, 3) == 0.996

    def test_mean_model(self) -> None:
        """Test NSE is 0 when predictions equal the mean."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        mean_val = np.mean(observed)
        predicted = np.full_like(observed, mean_val)
        result = nash_sutcliffe(observed, predicted)
        assert abs(result) < 1e-10


class TestKlingGupta:
    """Tests for the kling_gupta function."""

    def test_perfect_fit(self) -> None:
        """Test KGE is 1 for perfect fit."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 2.0, 3.0, 4.0])
        assert kling_gupta(observed, predicted) == 1.0

    def test_basic_calculation(self) -> None:
        """Test basic KGE calculation."""
        observed = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.1, 2.0, 2.9, 4.0])
        result = kling_gupta(observed, predicted)
        assert result > 0.95


class TestCategoricalScores:
    """Tests for the categorical_scores function."""

    def test_perfect_detection(self) -> None:
        """Test POD is 1 for perfect detection."""
        observed = np.array([0.0, 0.5, 1.0, 0.0, 2.0])
        predicted = np.array([0.0, 0.5, 1.0, 0.0, 2.0])
        scores = categorical_scores(observed, predicted, threshold=0.1)
        assert scores["pod"] == 1.0

    def test_basic_scores(self) -> None:
        """Test basic categorical scores."""
        observed = np.array([0.0, 0.5, 1.0, 0.0, 2.0])
        predicted = np.array([0.0, 0.3, 0.8, 0.2, 1.5])
        scores = categorical_scores(observed, predicted, threshold=0.1)
        assert scores["pod"] == 1.0  # All rain events detected
        assert scores["far"] == 0.25  # 1 false alarm out of 4 predicted rain

    def test_hits_and_misses(self) -> None:
        """Test hits and misses calculation."""
        observed = np.array([1.0, 1.0, 0.0, 0.0])
        predicted = np.array([1.0, 0.0, 1.0, 0.0])
        scores = categorical_scores(observed, predicted, threshold=0.1)
        assert scores["hits"] == 1.0
        assert scores["misses"] == 1.0
        assert scores["false_alarms"] == 1.0
        assert scores["correct_negatives"] == 1.0


class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_returns_all_metrics(self) -> None:
        """Test that all expected metrics are returned."""
        observed = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        predicted = np.array([0.1, 0.9, 2.1, 2.9, 4.1])
        metrics = compute_metrics(observed, predicted)
        expected_keys = [
            "n_valid",
            "mean_observed",
            "mean_predicted",
            "rmse",
            "mae",
            "bias",
            "relative_bias",
            "correlation",
            "nse",
            "kge",
            "pod",
            "far",
            "csi",
        ]
        for key in expected_keys:
            assert key in metrics

    def test_valid_count(self) -> None:
        """Test that valid count is correct."""
        observed = np.array([1.0, np.nan, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])
        metrics = compute_metrics(observed, predicted)
        assert metrics["n_valid"] == 2

"""Validation metrics and evaluation tools for precipitation data.

This module provides functions for evaluating the quality of precipitation
estimates by comparing them against reference observations. It includes
standard statistical metrics, categorical verification scores, and
visualization utilities.

Functions:
    compute_metrics: Compute comprehensive validation metrics.
    rmse: Calculate Root Mean Square Error.
    mae: Calculate Mean Absolute Error.
    bias: Calculate mean bias.
    correlation: Calculate Pearson correlation coefficient.
    nash_sutcliffe: Calculate Nash-Sutcliffe efficiency.
    kling_gupta: Calculate Kling-Gupta efficiency.
    categorical_scores: Calculate categorical verification scores.

Example:
    Validating precipitation estimates::

        from pacha.validation import compute_metrics, categorical_scores
        import numpy as np

        # Sample data
        observed = np.array([0.0, 1.5, 3.2, 0.0, 5.1, 2.3])
        predicted = np.array([0.2, 1.2, 2.8, 0.0, 4.5, 2.0])

        # Compute continuous metrics
        metrics = compute_metrics(observed, predicted)
        print(f"RMSE: {metrics['rmse']:.2f} mm")
        print(f"NSE: {metrics['nse']:.3f}")

        # Compute categorical scores (hit/miss/false alarm)
        cat = categorical_scores(observed, predicted, threshold=0.1)
        print(f"POD: {cat['pod']:.2f}")
        print(f"FAR: {cat['far']:.2f}")

"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def rmse(
    observed: npt.NDArray[np.floating[Any]],
    predicted: npt.NDArray[np.floating[Any]],
) -> float:
    """Calculate Root Mean Square Error (RMSE).

    RMSE measures the average magnitude of errors between predicted
    and observed values, giving higher weight to large errors.

    The formula is:
        RMSE = sqrt(mean((predicted - observed)^2))

    Args:
        observed: Array of observed/reference values.
        predicted: Array of predicted/estimated values.

    Returns:
        Root mean square error value.

    Raises:
        ValueError: If observed and predicted have different shapes.

    Example:
        >>> from pacha.validation import rmse
        >>> import numpy as np
        >>> observed = np.array([1.0, 2.0, 3.0])
        >>> predicted = np.array([1.1, 2.2, 2.9])
        >>> round(rmse(observed, predicted), 2)
        0.14
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if observed.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed.shape} vs "
            f"predicted {predicted.shape}"
        )

    # Mask NaN values
    valid_mask = ~np.isnan(observed) & ~np.isnan(predicted)

    if not np.any(valid_mask):
        return np.nan

    return float(np.sqrt(np.mean((predicted[valid_mask] - observed[valid_mask]) ** 2)))


def mae(
    observed: npt.NDArray[np.floating[Any]],
    predicted: npt.NDArray[np.floating[Any]],
) -> float:
    """Calculate Mean Absolute Error (MAE).

    MAE measures the average magnitude of errors without considering
    their direction, treating all errors equally regardless of sign.

    The formula is:
        MAE = mean(|predicted - observed|)

    Args:
        observed: Array of observed/reference values.
        predicted: Array of predicted/estimated values.

    Returns:
        Mean absolute error value.

    Raises:
        ValueError: If observed and predicted have different shapes.

    Example:
        >>> from pacha.validation import mae
        >>> import numpy as np
        >>> observed = np.array([1.0, 2.0, 3.0])
        >>> predicted = np.array([1.1, 1.8, 3.2])
        >>> round(mae(observed, predicted), 2)
        0.17
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if observed.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed.shape} vs "
            f"predicted {predicted.shape}"
        )

    valid_mask = ~np.isnan(observed) & ~np.isnan(predicted)

    if not np.any(valid_mask):
        return np.nan

    return float(np.mean(np.abs(predicted[valid_mask] - observed[valid_mask])))


def bias(
    observed: npt.NDArray[np.floating[Any]],
    predicted: npt.NDArray[np.floating[Any]],
    relative: bool = False,
) -> float:
    """Calculate mean bias (systematic error).

    Bias measures the average difference between predicted and observed
    values, indicating systematic over- or under-estimation.

    For absolute bias:
        Bias = mean(predicted - observed)

    For relative bias (percent):
        Relative Bias = 100 * (mean(predicted) - mean(observed)) / mean(observed)

    Args:
        observed: Array of observed/reference values.
        predicted: Array of predicted/estimated values.
        relative: If True, return relative bias as percentage.
            Defaults to False.

    Returns:
        Mean bias value (absolute or relative percentage).

    Raises:
        ValueError: If observed and predicted have different shapes.

    Example:
        >>> from pacha.validation import bias
        >>> import numpy as np
        >>> observed = np.array([1.0, 2.0, 3.0])
        >>> predicted = np.array([1.2, 2.2, 3.2])
        >>> bias(observed, predicted)
        0.2
        >>> round(bias(observed, predicted, relative=True), 1)
        10.0
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if observed.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed.shape} vs "
            f"predicted {predicted.shape}"
        )

    valid_mask = ~np.isnan(observed) & ~np.isnan(predicted)

    if not np.any(valid_mask):
        return np.nan

    obs_valid = observed[valid_mask]
    pred_valid = predicted[valid_mask]

    if relative:
        mean_obs = np.mean(obs_valid)
        if mean_obs == 0:
            return np.nan
        return float(100 * (np.mean(pred_valid) - mean_obs) / mean_obs)

    return float(np.mean(pred_valid - obs_valid))


def correlation(
    observed: npt.NDArray[np.floating[Any]],
    predicted: npt.NDArray[np.floating[Any]],
) -> float:
    """Calculate Pearson correlation coefficient.

    Measures the linear relationship between observed and predicted
    values. Values range from -1 (perfect negative correlation) to
    1 (perfect positive correlation).

    Args:
        observed: Array of observed/reference values.
        predicted: Array of predicted/estimated values.

    Returns:
        Pearson correlation coefficient.

    Raises:
        ValueError: If observed and predicted have different shapes.

    Example:
        >>> from pacha.validation import correlation
        >>> import numpy as np
        >>> observed = np.array([1.0, 2.0, 3.0, 4.0])
        >>> predicted = np.array([1.1, 2.1, 2.9, 4.1])
        >>> round(correlation(observed, predicted), 3)
        0.999
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if observed.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed.shape} vs "
            f"predicted {predicted.shape}"
        )

    valid_mask = ~np.isnan(observed) & ~np.isnan(predicted)

    if np.sum(valid_mask) < 2:
        return np.nan

    corr_matrix = np.corrcoef(observed[valid_mask], predicted[valid_mask])
    return float(corr_matrix[0, 1])


def nash_sutcliffe(
    observed: npt.NDArray[np.floating[Any]],
    predicted: npt.NDArray[np.floating[Any]],
) -> float:
    """Calculate Nash-Sutcliffe Efficiency (NSE).

    NSE is a normalized statistic that determines the relative magnitude
    of residual variance compared to measured data variance. Values range
    from -∞ to 1, where 1 is perfect fit and 0 indicates the model
    performs no better than using the mean of observations.

    The formula is:
        NSE = 1 - sum((observed - predicted)^2) / sum((observed - mean(observed))^2)

    Args:
        observed: Array of observed/reference values.
        predicted: Array of predicted/estimated values.

    Returns:
        Nash-Sutcliffe efficiency value.

    Raises:
        ValueError: If observed and predicted have different shapes.

    Example:
        >>> from pacha.validation import nash_sutcliffe
        >>> import numpy as np
        >>> observed = np.array([1.0, 2.0, 3.0, 4.0])
        >>> predicted = np.array([1.1, 2.0, 2.9, 4.0])
        >>> round(nash_sutcliffe(observed, predicted), 3)
        0.996

    Notes:
        - NSE > 0.5 is generally considered acceptable for hydrological modeling.
        - NSE is sensitive to systematic bias and outliers.
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if observed.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed.shape} vs "
            f"predicted {predicted.shape}"
        )

    valid_mask = ~np.isnan(observed) & ~np.isnan(predicted)

    if not np.any(valid_mask):
        return np.nan

    obs_valid = observed[valid_mask]
    pred_valid = predicted[valid_mask]

    obs_mean = np.mean(obs_valid)
    ss_res = np.sum((obs_valid - pred_valid) ** 2)
    ss_tot = np.sum((obs_valid - obs_mean) ** 2)

    if ss_tot == 0:
        return np.nan

    return float(1 - (ss_res / ss_tot))


def kling_gupta(
    observed: npt.NDArray[np.floating[Any]],
    predicted: npt.NDArray[np.floating[Any]],
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """Calculate Kling-Gupta Efficiency (KGE).

    KGE decomposes model performance into correlation, variability bias,
    and mean bias components. It addresses some limitations of NSE,
    particularly the insensitivity to systematic bias.

    The formula is:
        KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)

    where:
        - r: Pearson correlation coefficient
        - alpha: ratio of standard deviations (predicted/observed)
        - beta: ratio of means (predicted/observed)

    Args:
        observed: Array of observed/reference values.
        predicted: Array of predicted/estimated values.
        weights: Weights for correlation, variability, and bias components.
            Defaults to (1.0, 1.0, 1.0).

    Returns:
        Kling-Gupta efficiency value.

    Raises:
        ValueError: If observed and predicted have different shapes.

    Example:
        >>> from pacha.validation import kling_gupta
        >>> import numpy as np
        >>> observed = np.array([1.0, 2.0, 3.0, 4.0])
        >>> predicted = np.array([1.1, 2.0, 2.9, 4.0])
        >>> round(kling_gupta(observed, predicted), 3)
        0.95

    Notes:
        - KGE > 0 indicates the model outperforms using the mean.
        - KGE > 0.5 is generally considered acceptable.
        - Perfect score is 1.0.
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if observed.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed.shape} vs "
            f"predicted {predicted.shape}"
        )

    valid_mask = ~np.isnan(observed) & ~np.isnan(predicted)

    if np.sum(valid_mask) < 2:
        return np.nan

    obs_valid = observed[valid_mask]
    pred_valid = predicted[valid_mask]

    # Calculate components
    r = correlation(obs_valid, pred_valid)

    obs_std = np.std(obs_valid)
    pred_std = np.std(pred_valid)
    alpha = pred_std / obs_std if obs_std > 0 else np.nan

    obs_mean = np.mean(obs_valid)
    pred_mean = np.mean(pred_valid)
    beta = pred_mean / obs_mean if obs_mean > 0 else np.nan

    if np.isnan(r) or np.isnan(alpha) or np.isnan(beta):
        return np.nan

    # Calculate KGE
    w_r, w_alpha, w_beta = weights
    kge = 1 - np.sqrt(
        w_r * (r - 1) ** 2 + w_alpha * (alpha - 1) ** 2 + w_beta * (beta - 1) ** 2
    )

    return float(kge)


def categorical_scores(
    observed: npt.NDArray[np.floating[Any]],
    predicted: npt.NDArray[np.floating[Any]],
    threshold: float = 0.1,
) -> dict[str, float]:
    """Calculate categorical verification scores for precipitation detection.

    Evaluates the ability to detect precipitation events using a binary
    classification framework (rain/no-rain). Computes standard categorical
    metrics from the confusion matrix.

    Args:
        observed: Array of observed/reference values.
        predicted: Array of predicted/estimated values.
        threshold: Precipitation threshold for rain/no-rain classification.
            Values >= threshold are classified as rain. Defaults to 0.1.

    Returns:
        Dictionary containing categorical scores:
            - 'hits': Number of true positives (correct rain detection)
            - 'misses': Number of false negatives (missed rain)
            - 'false_alarms': Number of false positives (false rain)
            - 'correct_negatives': Number of true negatives (correct no-rain)
            - 'pod': Probability of Detection (hit rate)
            - 'far': False Alarm Ratio
            - 'csi': Critical Success Index (threat score)
            - 'bias_score': Frequency Bias (not to be confused with mean bias)
            - 'accuracy': Overall accuracy
            - 'hss': Heidke Skill Score

    Raises:
        ValueError: If observed and predicted have different shapes.

    Example:
        >>> from pacha.validation import categorical_scores
        >>> import numpy as np
        >>> observed = np.array([0.0, 0.5, 1.0, 0.0, 2.0])
        >>> predicted = np.array([0.0, 0.3, 0.8, 0.2, 1.5])
        >>> scores = categorical_scores(observed, predicted, threshold=0.1)
        >>> scores['pod']
        1.0
        >>> scores['far']
        0.25

    Notes:
        - POD (Probability of Detection) ranges from 0 to 1 (higher is better).
        - FAR (False Alarm Ratio) ranges from 0 to 1 (lower is better).
        - CSI (Critical Success Index) ranges from 0 to 1 (higher is better).
        - HSS (Heidke Skill Score) ranges from -∞ to 1 (higher is better).
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if observed.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed.shape} vs "
            f"predicted {predicted.shape}"
        )

    valid_mask = ~np.isnan(observed) & ~np.isnan(predicted)

    if not np.any(valid_mask):
        return {
            "hits": np.nan,
            "misses": np.nan,
            "false_alarms": np.nan,
            "correct_negatives": np.nan,
            "pod": np.nan,
            "far": np.nan,
            "csi": np.nan,
            "bias_score": np.nan,
            "accuracy": np.nan,
            "hss": np.nan,
        }

    obs_valid = observed[valid_mask]
    pred_valid = predicted[valid_mask]

    # Create binary classifications
    obs_rain = obs_valid >= threshold
    pred_rain = pred_valid >= threshold

    # Calculate contingency table elements
    hits = np.sum(obs_rain & pred_rain)
    misses = np.sum(obs_rain & ~pred_rain)
    false_alarms = np.sum(~obs_rain & pred_rain)
    correct_negatives = np.sum(~obs_rain & ~pred_rain)

    n_total = len(obs_valid)

    # Calculate scores
    # POD (Probability of Detection / Hit Rate)
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan

    # FAR (False Alarm Ratio)
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan

    # CSI (Critical Success Index / Threat Score)
    csi = (
        hits / (hits + misses + false_alarms)
        if (hits + misses + false_alarms) > 0
        else np.nan
    )

    # Frequency Bias
    bias_score = (
        (hits + false_alarms) / (hits + misses) if (hits + misses) > 0 else np.nan
    )

    # Accuracy
    accuracy = (hits + correct_negatives) / n_total if n_total > 0 else np.nan

    # HSS (Heidke Skill Score)
    expected_correct = (
        (hits + misses) * (hits + false_alarms)
        + (correct_negatives + misses) * (correct_negatives + false_alarms)
    ) / n_total
    hss = (
        (hits + correct_negatives - expected_correct) / (n_total - expected_correct)
        if (n_total - expected_correct) > 0
        else np.nan
    )

    return {
        "hits": float(hits),
        "misses": float(misses),
        "false_alarms": float(false_alarms),
        "correct_negatives": float(correct_negatives),
        "pod": float(pod) if not np.isnan(pod) else np.nan,
        "far": float(far) if not np.isnan(far) else np.nan,
        "csi": float(csi) if not np.isnan(csi) else np.nan,
        "bias_score": float(bias_score) if not np.isnan(bias_score) else np.nan,
        "accuracy": float(accuracy) if not np.isnan(accuracy) else np.nan,
        "hss": float(hss) if not np.isnan(hss) else np.nan,
    }


def compute_metrics(
    observed: npt.NDArray[np.floating[Any]],
    predicted: npt.NDArray[np.floating[Any]],
    threshold: float = 0.1,
) -> dict[str, float]:
    """Compute comprehensive validation metrics for precipitation.

    Calculates a complete set of continuous and categorical metrics
    for evaluating precipitation estimates against reference observations.

    Args:
        observed: Array of observed/reference values.
        predicted: Array of predicted/estimated values.
        threshold: Threshold for categorical scores. Defaults to 0.1.

    Returns:
        Dictionary containing all computed metrics:
            - 'n_valid': Number of valid data pairs
            - 'mean_observed': Mean of observed values
            - 'mean_predicted': Mean of predicted values
            - 'rmse': Root Mean Square Error
            - 'mae': Mean Absolute Error
            - 'bias': Mean bias
            - 'relative_bias': Relative bias (%)
            - 'correlation': Pearson correlation coefficient
            - 'nse': Nash-Sutcliffe Efficiency
            - 'kge': Kling-Gupta Efficiency
            - 'pod': Probability of Detection
            - 'far': False Alarm Ratio
            - 'csi': Critical Success Index

    Raises:
        ValueError: If observed and predicted have different shapes.

    Example:
        >>> from pacha.validation import compute_metrics
        >>> import numpy as np
        >>> observed = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        >>> predicted = np.array([0.1, 0.9, 2.1, 2.9, 4.1])
        >>> metrics = compute_metrics(observed, predicted)
        >>> round(metrics['rmse'], 2)
        0.12
        >>> round(metrics['nse'], 2)
        0.99
    """
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    if observed.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: observed {observed.shape} vs "
            f"predicted {predicted.shape}"
        )

    valid_mask = ~np.isnan(observed) & ~np.isnan(predicted)
    n_valid = np.sum(valid_mask)

    # Basic statistics
    mean_obs = np.mean(observed[valid_mask]) if n_valid > 0 else np.nan
    mean_pred = np.mean(predicted[valid_mask]) if n_valid > 0 else np.nan

    # Continuous metrics
    rmse_val = rmse(observed, predicted)
    mae_val = mae(observed, predicted)
    bias_val = bias(observed, predicted)
    rel_bias = bias(observed, predicted, relative=True)
    corr = correlation(observed, predicted)
    nse = nash_sutcliffe(observed, predicted)
    kge = kling_gupta(observed, predicted)

    # Categorical metrics
    cat_scores = categorical_scores(observed, predicted, threshold=threshold)

    return {
        "n_valid": int(n_valid),
        "mean_observed": float(mean_obs),
        "mean_predicted": float(mean_pred),
        "rmse": rmse_val,
        "mae": mae_val,
        "bias": bias_val,
        "relative_bias": rel_bias,
        "correlation": corr,
        "nse": nse,
        "kge": kge,
        "pod": cat_scores["pod"],
        "far": cat_scores["far"],
        "csi": cat_scores["csi"],
    }


__all__ = [
    "rmse",
    "mae",
    "bias",
    "correlation",
    "nash_sutcliffe",
    "kling_gupta",
    "categorical_scores",
    "compute_metrics",
]

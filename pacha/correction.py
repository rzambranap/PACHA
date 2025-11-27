"""Precipitation bias correction and adjustment methods.

This module provides various methods for correcting systematic biases
in precipitation estimates. It includes techniques commonly used in
hydrology and meteorology for adjusting radar, satellite, and model
precipitation data using reference observations.

Functions:
    mean_field_bias: Apply mean field bias correction.
    quantile_mapping: Apply quantile mapping bias correction.
    linear_scaling: Apply linear scaling correction.
    local_intensity_scaling: Apply local intensity scaling correction.
    distribution_mapping: Apply distribution-based bias correction.

Example:
    Correcting satellite precipitation using gauge data::

        from pacha.correction import mean_field_bias, quantile_mapping
        import numpy as np

        # Sample satellite estimates and gauge observations
        satellite = np.array([0.5, 1.2, 2.0, 0.3, 1.8])
        gauge = np.array([0.8, 1.0, 2.5, 0.5, 1.5])

        # Apply mean field bias correction
        corrected_mfb = mean_field_bias(satellite, gauge)

        # Apply quantile mapping
        corrected_qm = quantile_mapping(satellite, gauge)

"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats


def mean_field_bias(
    estimates: npt.NDArray[np.floating[Any]],
    reference: npt.NDArray[np.floating[Any]],
    threshold: float = 0.0,
) -> npt.NDArray[np.floating[Any]]:
    """Apply mean field bias correction to precipitation estimates.

    Adjusts precipitation estimates by multiplying with the ratio of
    mean reference to mean estimate values. This is one of the simplest
    and most widely used bias correction methods.

    The correction factor (G) is calculated as:
        G = mean(reference) / mean(estimates)
        corrected = estimates * G

    Args:
        estimates: Array of precipitation estimates to be corrected.
        reference: Array of reference precipitation values (e.g., from gauges).
        threshold: Minimum precipitation value to include in bias calculation.
            Values below this threshold are treated as no-rain and excluded
            from the ratio calculation. Defaults to 0.0.

    Returns:
        Array of bias-corrected precipitation estimates.

    Raises:
        ValueError: If estimates and reference have different shapes.
        ValueError: If mean of estimates is zero (cannot compute ratio).

    Example:
        >>> from pacha.correction import mean_field_bias
        >>> import numpy as np
        >>>
        >>> # Satellite underestimates by ~50%
        >>> satellite = np.array([0.5, 1.0, 1.5, 2.0])
        >>> gauge = np.array([1.0, 2.0, 3.0, 4.0])
        >>>
        >>> corrected = mean_field_bias(satellite, gauge)
        >>> np.allclose(corrected, gauge)
        True

    Notes:
        - This method assumes a spatially and temporally constant bias.
        - Works best for large-scale, systematic biases.
        - May not perform well when bias varies with intensity.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if estimates.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: estimates {estimates.shape} vs "
            f"reference {reference.shape}"
        )

    # Create mask for valid values (both above threshold and not NaN)
    valid_mask = (
        (estimates > threshold)
        & (reference > threshold)
        & ~np.isnan(estimates)
        & ~np.isnan(reference)
    )

    if not np.any(valid_mask):
        # No valid pairs, return original estimates
        return estimates.copy()

    mean_estimate = np.mean(estimates[valid_mask])
    mean_reference = np.mean(reference[valid_mask])

    if mean_estimate == 0:
        raise ValueError(
            "Mean of estimates is zero. Cannot compute bias correction factor."
        )

    # Calculate correction factor
    correction_factor = mean_reference / mean_estimate

    # Apply correction
    corrected = estimates * correction_factor

    return corrected


def linear_scaling(
    estimates: npt.NDArray[np.floating[Any]],
    reference: npt.NDArray[np.floating[Any]],
    method: str = "ratio",
) -> npt.NDArray[np.floating[Any]]:
    """Apply linear scaling correction to precipitation estimates.

    Adjusts precipitation estimates using a linear relationship derived
    from the comparison between estimates and reference data.

    Two methods are available:
        - 'ratio': Multiplicative scaling using mean ratio (same as mean_field_bias)
        - 'regression': Linear regression scaling (y = a*x + b)

    Args:
        estimates: Array of precipitation estimates to be corrected.
        reference: Array of reference precipitation values.
        method: Scaling method ('ratio' or 'regression').
            Defaults to 'ratio'.

    Returns:
        Array of corrected precipitation estimates.

    Raises:
        ValueError: If estimates and reference have different shapes.
        ValueError: If method is not 'ratio' or 'regression'.

    Example:
        >>> from pacha.correction import linear_scaling
        >>> import numpy as np
        >>>
        >>> estimates = np.array([1.0, 2.0, 3.0, 4.0])
        >>> reference = np.array([1.5, 2.5, 3.5, 4.5])
        >>>
        >>> # Using regression method
        >>> corrected = linear_scaling(estimates, reference, method="regression")

    Notes:
        - The ratio method preserves zeros in the estimates.
        - The regression method may produce negative values which are clipped to zero.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if estimates.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: estimates {estimates.shape} vs "
            f"reference {reference.shape}"
        )

    if method == "ratio":
        return mean_field_bias(estimates, reference, threshold=0.0)

    elif method == "regression":
        # Get valid pairs
        valid_mask = ~np.isnan(estimates) & ~np.isnan(reference)

        if not np.any(valid_mask):
            return estimates.copy()

        # Fit linear regression
        slope, intercept, _, _, _ = stats.linregress(
            estimates[valid_mask], reference[valid_mask]
        )

        # Apply correction
        corrected = slope * estimates + intercept

        # Clip negative values to zero
        corrected = np.maximum(corrected, 0.0)

        return corrected

    else:
        raise ValueError(
            f"Invalid method '{method}'. Choose from: 'ratio', 'regression'"
        )


def quantile_mapping(
    estimates: npt.NDArray[np.floating[Any]],
    reference: npt.NDArray[np.floating[Any]],
    n_quantiles: int = 100,
    extrapolation: str = "constant",
) -> npt.NDArray[np.floating[Any]]:
    """Apply quantile mapping bias correction.

    Quantile mapping corrects the distribution of estimates to match the
    distribution of reference data. Each estimate value is mapped to
    the reference value with the same cumulative probability.

    Args:
        estimates: Array of precipitation estimates to be corrected.
        reference: Array of reference precipitation values used for training.
        n_quantiles: Number of quantiles to use for the mapping.
            Higher values provide finer resolution but require more data.
            Defaults to 100.
        extrapolation: Method for handling values outside the training range.
            Options: 'constant' (use nearest quantile), 'linear' (extrapolate).
            Defaults to 'constant'.

    Returns:
        Array of bias-corrected precipitation estimates.

    Raises:
        ValueError: If n_quantiles is less than 2.
        ValueError: If extrapolation method is not supported.

    Example:
        >>> from pacha.correction import quantile_mapping
        >>> import numpy as np
        >>>
        >>> # Training data
        >>> estimates_train = np.array([0, 0.5, 1.0, 1.5, 2.0, 3.0])
        >>> reference_train = np.array([0, 0.8, 1.5, 2.2, 3.0, 4.5])
        >>>
        >>> # Apply correction
        >>> corrected = quantile_mapping(estimates_train, reference_train)

    Notes:
        - Works best when training data is representative of all conditions.
        - Preserves the rank order of estimates.
        - Can handle non-linear biases that vary with intensity.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")

    valid_extrapolations = ("constant", "linear")
    if extrapolation not in valid_extrapolations:
        raise ValueError(
            f"Invalid extrapolation '{extrapolation}'. "
            f"Choose from: {valid_extrapolations}"
        )

    # Remove NaN values for quantile calculation
    est_valid = estimates[~np.isnan(estimates)]
    ref_valid = reference[~np.isnan(reference)]

    if len(est_valid) == 0 or len(ref_valid) == 0:
        return estimates.copy()

    # Calculate quantiles
    quantile_levels = np.linspace(0, 1, n_quantiles)
    est_quantiles = np.nanquantile(est_valid, quantile_levels)
    ref_quantiles = np.nanquantile(ref_valid, quantile_levels)

    # Apply mapping using interpolation
    corrected = np.interp(
        estimates,
        est_quantiles,
        ref_quantiles,
        left=ref_quantiles[0] if extrapolation == "constant" else np.nan,
        right=ref_quantiles[-1] if extrapolation == "constant" else np.nan,
    )

    # Handle linear extrapolation if needed
    if extrapolation == "linear":
        below_mask = estimates < est_quantiles[0]
        above_mask = estimates > est_quantiles[-1]

        if np.any(below_mask) and len(est_quantiles) >= 2:
            slope = (ref_quantiles[1] - ref_quantiles[0]) / (
                est_quantiles[1] - est_quantiles[0] + 1e-10
            )
            corrected[below_mask] = ref_quantiles[0] + slope * (
                estimates[below_mask] - est_quantiles[0]
            )

        if np.any(above_mask) and len(est_quantiles) >= 2:
            slope = (ref_quantiles[-1] - ref_quantiles[-2]) / (
                est_quantiles[-1] - est_quantiles[-2] + 1e-10
            )
            corrected[above_mask] = ref_quantiles[-1] + slope * (
                estimates[above_mask] - est_quantiles[-1]
            )

    # Ensure non-negative values
    corrected = np.maximum(corrected, 0.0)

    # Preserve NaN values from original
    corrected[np.isnan(estimates)] = np.nan

    return corrected


def local_intensity_scaling(
    estimates: npt.NDArray[np.floating[Any]],
    reference: npt.NDArray[np.floating[Any]],
    intensity_bins: npt.NDArray[np.floating[Any]] | None = None,
    n_bins: int = 10,
) -> npt.NDArray[np.floating[Any]]:
    """Apply local intensity scaling correction.

    This method applies different correction factors for different
    precipitation intensity ranges, allowing for intensity-dependent
    bias correction.

    Args:
        estimates: Array of precipitation estimates to be corrected.
        reference: Array of reference precipitation values.
        intensity_bins: Array of bin edges for intensity classification.
            If None, bins are automatically determined. Defaults to None.
        n_bins: Number of intensity bins if intensity_bins is None.
            Defaults to 10.

    Returns:
        Array of bias-corrected precipitation estimates.

    Raises:
        ValueError: If estimates and reference have different shapes.

    Example:
        >>> from pacha.correction import local_intensity_scaling
        >>> import numpy as np
        >>>
        >>> estimates = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
        >>> reference = np.array([0.2, 0.6, 1.2, 6.0, 12.0])
        >>>
        >>> # Automatic bins
        >>> corrected = local_intensity_scaling(estimates, reference)
        >>>
        >>> # Custom bins
        >>> bins = np.array([0, 0.5, 1.0, 5.0, 20.0])
        >>> corrected = local_intensity_scaling(
        ...     estimates, reference, intensity_bins=bins
        ... )

    Notes:
        - More robust than single-factor correction for intensity-dependent biases.
        - Requires sufficient data in each intensity bin for reliable correction.
        - Bins with insufficient data use the global correction factor.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    if estimates.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch: estimates {estimates.shape} vs "
            f"reference {reference.shape}"
        )

    # Create mask for valid values
    valid_mask = ~np.isnan(estimates) & ~np.isnan(reference)

    if not np.any(valid_mask):
        return estimates.copy()

    # Determine intensity bins
    if intensity_bins is None:
        # Use quantile-based bins from estimates
        quantile_levels = np.linspace(0, 1, n_bins + 1)
        intensity_bins = np.nanquantile(estimates[valid_mask], quantile_levels)
        # Ensure unique bins
        intensity_bins = np.unique(intensity_bins)

    # Calculate global correction factor as fallback
    global_factor = np.nanmean(reference[valid_mask]) / (
        np.nanmean(estimates[valid_mask]) + 1e-10
    )

    # Initialize corrected array
    corrected = estimates.copy()

    # Apply bin-specific correction
    for i in range(len(intensity_bins) - 1):
        bin_low = intensity_bins[i]
        bin_high = intensity_bins[i + 1]

        # Include upper bound in last bin
        if i == len(intensity_bins) - 2:
            bin_mask = (estimates >= bin_low) & (estimates <= bin_high)
        else:
            bin_mask = (estimates >= bin_low) & (estimates < bin_high)

        combined_mask = bin_mask & valid_mask

        if np.sum(combined_mask) >= 3:  # Minimum samples for reliable factor
            bin_mean_est = np.mean(estimates[combined_mask])
            bin_mean_ref = np.mean(reference[combined_mask])

            if bin_mean_est > 0:
                bin_factor = bin_mean_ref / bin_mean_est
            else:
                bin_factor = global_factor
        else:
            bin_factor = global_factor

        corrected[bin_mask] = estimates[bin_mask] * bin_factor

    return corrected


def distribution_mapping(
    estimates: npt.NDArray[np.floating[Any]],
    reference: npt.NDArray[np.floating[Any]],
    distribution: str = "gamma",
) -> npt.NDArray[np.floating[Any]]:
    """Apply distribution-based bias correction.

    Fits parametric distributions to both estimates and reference data,
    then maps estimates to reference distribution using the cumulative
    distribution functions.

    Args:
        estimates: Array of precipitation estimates to be corrected.
        reference: Array of reference precipitation values.
        distribution: Distribution type to fit ('gamma', 'exponential', 'weibull').
            Defaults to 'gamma'.

    Returns:
        Array of bias-corrected precipitation estimates.

    Raises:
        ValueError: If distribution type is not supported.

    Example:
        >>> from pacha.correction import distribution_mapping
        >>> import numpy as np
        >>>
        >>> estimates = np.random.exponential(scale=1.0, size=100)
        >>> reference = np.random.gamma(shape=2.0, scale=1.0, size=100)
        >>>
        >>> corrected = distribution_mapping(estimates, reference, distribution="gamma")

    Notes:
        - Gamma distribution is most commonly used for precipitation.
        - Only fits to non-zero values; zeros are preserved.
        - Requires sufficient data for reliable distribution fitting.
    """
    estimates = np.asarray(estimates, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    supported_distributions = ("gamma", "exponential", "weibull")
    if distribution not in supported_distributions:
        raise ValueError(
            f"Unsupported distribution '{distribution}'. "
            f"Choose from: {supported_distributions}"
        )

    # Get non-zero, non-NaN values
    est_positive = estimates[(estimates > 0) & ~np.isnan(estimates)]
    ref_positive = reference[(reference > 0) & ~np.isnan(reference)]

    if len(est_positive) < 5 or len(ref_positive) < 5:
        # Not enough data for distribution fitting, fall back to mean field bias
        return mean_field_bias(estimates, reference)

    # Fit distributions
    if distribution == "gamma":
        est_params = stats.gamma.fit(est_positive, floc=0)
        ref_params = stats.gamma.fit(ref_positive, floc=0)
        est_cdf = lambda x: stats.gamma.cdf(x, *est_params)  # noqa: E731
        ref_ppf = lambda p: stats.gamma.ppf(p, *ref_params)  # noqa: E731

    elif distribution == "exponential":
        est_params = stats.expon.fit(est_positive, floc=0)
        ref_params = stats.expon.fit(ref_positive, floc=0)
        est_cdf = lambda x: stats.expon.cdf(x, *est_params)  # noqa: E731
        ref_ppf = lambda p: stats.expon.ppf(p, *ref_params)  # noqa: E731

    else:  # weibull
        est_params = stats.weibull_min.fit(est_positive, floc=0)
        ref_params = stats.weibull_min.fit(ref_positive, floc=0)
        est_cdf = lambda x: stats.weibull_min.cdf(x, *est_params)  # noqa: E731
        ref_ppf = lambda p: stats.weibull_min.ppf(p, *ref_params)  # noqa: E731

    # Apply correction
    corrected = estimates.copy()

    # Only correct positive values
    positive_mask = (estimates > 0) & ~np.isnan(estimates)

    if np.any(positive_mask):
        # Get CDF values for estimates
        cdf_values = est_cdf(estimates[positive_mask])

        # Clip CDF values to avoid numerical issues at extremes
        cdf_values = np.clip(cdf_values, 1e-10, 1 - 1e-10)

        # Map to reference distribution
        corrected[positive_mask] = ref_ppf(cdf_values)

    # Ensure non-negative values
    corrected = np.maximum(corrected, 0.0)

    return corrected


__all__ = [
    "mean_field_bias",
    "linear_scaling",
    "quantile_mapping",
    "local_intensity_scaling",
    "distribution_mapping",
]

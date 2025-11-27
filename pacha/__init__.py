"""PACHA - Precipitation Analysis & Correction for Hydrological Applications.

PACHA is a Python package for analyzing, correcting, and validating
precipitation data from various sources including rain gauges, weather
radars, satellites, and commercial microwave links.

The package provides tools for:
    - Loading and processing precipitation data from multiple sources
    - Applying bias correction and adjustment methods
    - Validating precipitation estimates against reference data
    - Computing statistical metrics for evaluation

Modules:
    core: Core functionality and base classes for precipitation handling.
    data: Data loading, processing, and transformation utilities.
    correction: Precipitation bias correction and adjustment methods.
    validation: Validation metrics and evaluation tools.

Example:
    Basic usage of PACHA for loading and correcting precipitation data::

        import pacha

        # Load precipitation data
        precip_data = pacha.data.load_timeseries("rainfall.csv")

        # Apply mean field bias correction
        corrected = pacha.correction.mean_field_bias(
            estimates=precip_data,
            reference=reference_data
        )

        # Validate the corrected estimates
        metrics = pacha.validation.compute_metrics(
            observed=reference_data,
            predicted=corrected
        )
        print(f"RMSE: {metrics['rmse']:.2f} mm")

Attributes:
    __version__: Package version string.
    __author__: Package author information.

"""

__version__ = "0.1.0"
__author__ = "Rodrigo Zambrana"

from pacha import core, correction, data, validation

__all__ = [
    "__version__",
    "__author__",
    "core",
    "data",
    "correction",
    "validation",
]

"""
L2 Processing module for PACHA.

This module contains functions for Level 2 data processing, which involves
deriving geophysical variables from L1 source data at the same resolution
and location.

Submodules
----------
radar
    Radar reflectivity to rain rate conversions.
daily_accumulation
    Daily precipitation accumulation from rain intensity data.
"""

from pacha.L2_processing.daily_accumulation import (  # noqa: F401
    PRODUCT_CONFIG,
    DailyAccumulationPipeline,
    calculate_daily_accumulation,
    check_timestep_completeness,
    compute_daily_accumulation,
    gap_fill,
    load_dataset,
    normalize_dataset,
    save_output,
)

__all__ = [
    'PRODUCT_CONFIG',
    'DailyAccumulationPipeline',
    'calculate_daily_accumulation',
    'check_timestep_completeness',
    'compute_daily_accumulation',
    'gap_fill',
    'load_dataset',
    'normalize_dataset',
    'save_output',
]

"""
Merging module for PACHA.

This module contains algorithms for data fusion and merging of precipitation
datasets from multiple sources, including quantile matching and bias correction
methods.
"""

from .spp_correction import (
    ClusteredCorrector,
    calculate_features_for_clustering,
    subdivide_dataset,
)

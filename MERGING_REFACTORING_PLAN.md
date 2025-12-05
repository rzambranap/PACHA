# PACHA Merging Module Refactoring Plan

## Overview

This document outlines a comprehensive plan to refactor the `pacha/merging` module to improve robustness, maintainability, and extensibility. The refactoring will also integrate K-Means classification capabilities to enable spatially-aware data fusion.

---

## Current State Analysis

### Existing Structure
```
pacha/merging/
├── __init__.py          # Empty module docstring
└── quantile_matching.py # All merging functionality (~566 lines)
```

### Current Implementation Issues

1. **Single File Monolith**: All functionality is in `quantile_matching.py`
2. **Tight Coupling**: The `Fuser` class handles multiple fusion methods with method-specific initialization
3. **Limited Extensibility**: Adding new fusion methods requires modifying the `Fuser` class internals
4. **No Spatial Classification**: No support for region-based or cluster-based fusion parameters
5. **No Abstract Base Classes**: No clear interface for fusion methods
6. **Limited Error Handling**: Minimal validation and error messages
7. **Missing Type Hints**: No type annotations for better IDE support and documentation
8. **No K-Means Integration**: Classification must be done externally (as seen in notebooks)

---

## Refactoring Goals

1. **Modularity**: Separate concerns into distinct modules
2. **Extensibility**: Use design patterns (Strategy/Factory) for easy addition of new methods
3. **Type Safety**: Add comprehensive type hints
4. **Validation**: Robust input validation and meaningful error messages
5. **Testing**: Enable easier unit testing of individual components
6. **K-Means Integration**: Native support for spatially-classified fusion
7. **Documentation**: Improved docstrings and examples

---

## Proposed New Structure

```
pacha/merging/
├── __init__.py              # Public API exports
├── base.py                  # Abstract base classes and interfaces
├── quantile_matching.py     # Quantile matching implementation
├── linear_regression.py     # Linear regression implementation  
├── classification.py        # K-Means and spatial classification
├── fuser.py                 # Unified Fuser class (orchestrator)
├── transfer_functions.py    # Transfer/fitting functions
├── utils.py                 # Helper functions (echelon, rescale)
└── validators.py            # Input validation utilities
```

---

## Detailed TODO List

### Phase 1: Foundation & Code Organization (Priority: High)

- [ ] **1.1 Create base module (`base.py`)**
  - [ ] Define `BaseFusionMethod` abstract base class with methods:
    - `fit(ds, **kwargs) -> dict` - Calculate fusion parameters
    - `transform(ds, params) -> xr.Dataset` - Apply fusion parameters
    - `fit_transform(ds, **kwargs) -> xr.Dataset` - Fit and transform in one step
    - `get_params() -> dict` - Get current parameters
    - `set_params(**params)` - Set parameters
  - [ ] Define `FusionResult` dataclass for standardized output

- [ ] **1.2 Create utils module (`utils.py`)**
  - [ ] Move `echelon_down()` function
  - [ ] Move `echelon_up()` function
  - [ ] Move `rescale_qs_to_calc()` function
  - [ ] Add unit tests for each utility function

- [ ] **1.3 Create transfer functions module (`transfer_functions.py`)**
  - [ ] Define `quadratic_transfer(x, a, b, c)` function
  - [ ] Define `linear_transfer(x, a)` function
  - [ ] Define `polynomial_transfer(x, *coeffs)` generic function
  - [ ] Add registry pattern for custom transfer functions
  - [ ] Add unit tests

- [ ] **1.4 Create validators module (`validators.py`)**
  - [ ] `validate_dataset_fields(ds, n_fields=2)` - Validate field count
  - [ ] `validate_quantiles(quantiles)` - Validate quantile array
  - [ ] `validate_reference_variable(ds, ref_var)` - Validate reference exists
  - [ ] `validate_fusion_params(params)` - Validate parameter dict
  - [ ] Add meaningful error messages and custom exceptions

### Phase 2: Refactor Existing Methods (Priority: High)

- [ ] **2.1 Refactor Quantile Matching (`quantile_matching.py`)**
  - [ ] Create `QuantileMatchingMethod` class inheriting from `BaseFusionMethod`
  - [ ] Implement `fit()` method with proper validation
  - [ ] Implement `transform()` method
  - [ ] Implement `fit_transform()` method
  - [ ] Keep `fit_qs()` function as utility
  - [ ] Keep `match_quantiles()` as standalone function for backward compatibility
  - [ ] Add type hints throughout
  - [ ] Add comprehensive docstrings
  - [ ] Add unit tests

- [ ] **2.2 Create Quantile Matching By Parts (`quantile_matching.py` or separate)**
  - [ ] Create `QuantileMatchingByPartsMethod` class
  - [ ] Implement two-part fitting with smooth transition
  - [ ] Add configurable cutoff quantile
  - [ ] Add unit tests

- [ ] **2.3 Create Linear Regression Module (`linear_regression.py`)**
  - [ ] Create `LinearRegressionMethod` class inheriting from `BaseFusionMethod`
  - [ ] Implement `fit()` method
  - [ ] Implement `transform()` method
  - [ ] Add unit tests

### Phase 3: K-Means Classification Integration (Priority: High)

- [ ] **3.1 Create classification module (`classification.py`)**
  - [ ] Define `SpatialClassifier` base class with methods:
    - `fit(ds) -> self` - Fit classifier to data
    - `predict(ds) -> np.ndarray` - Predict cluster labels
    - `get_cluster_masks() -> dict[int, xr.DataArray]` - Get masks for each cluster
  - [ ] Implement `KMeansClassifier` class:
    - [ ] Initialize with `n_clusters`, `features`, `random_state`
    - [ ] Support time-averaged features extraction
    - [ ] Support pixel-wise classification
    - [ ] Support custom feature engineering
  - [ ] Implement `LandSeaClassifier` class (refactor from `divide_dataset_land_sea`)
  - [ ] Implement `GridClassifier` for regular grid subdivision
  - [ ] Add serialization support (save/load models as pkl)
  - [ ] Add unit tests

- [ ] **3.2 Define feature extraction utilities**
  - [ ] `extract_temporal_features(ds)` - Mean, std, max, min over time
  - [ ] `extract_spatial_features(ds)` - Local statistics
  - [ ] `extract_combined_features(ds)` - Both temporal and spatial
  - [ ] Support PCA dimensionality reduction
  - [ ] Add unit tests

- [ ] **3.3 Integration with Fuser**
  - [ ] Add `classifier` parameter to `Fuser` class
  - [ ] Implement `calculate_fusing_params_by_cluster()`
  - [ ] Implement `apply_params_by_cluster()`
  - [ ] Support cluster-specific fusion methods

### Phase 4: Unified Fuser Class (Priority: Medium)

- [ ] **4.1 Create new Fuser (`fuser.py`)**
  - [ ] Implement Factory pattern for fusion method creation
  - [ ] Support method registration for custom methods
  - [ ] Constructor parameters:
    - `method`: str or `BaseFusionMethod` instance
    - `classifier`: Optional `SpatialClassifier`
    - `reference_variable`: str
    - `sep_land_sea`: bool (backward compatibility)
    - `**method_params`: Method-specific parameters
  - [ ] Methods:
    - `fit(ds)` - Calculate fusion parameters
    - `transform(ds)` - Apply fusion parameters
    - `fit_transform(ds)` - Combined fit and transform
    - `export_params()` - Export parameters as dict/JSON
    - `import_params()` - Import parameters
    - `save(filepath)` - Save model to file
    - `load(filepath)` - Load model from file
  - [ ] Backward compatibility with existing `Fuser` API
  - [ ] Add comprehensive unit tests

- [ ] **4.2 Method Registry**
  - [ ] Create `METHOD_REGISTRY` dict mapping names to classes
  - [ ] Support registration of custom methods:
    ```python
    Fuser.register_method('custom', CustomFusionMethod)
    ```
  - [ ] Built-in methods:
    - `'quantile_matching'`
    - `'quantile_matching_by_parts'`
    - `'linear_regression'`

### Phase 5: Enhanced Features (Priority: Medium)

- [ ] **5.1 Add visualization utilities**
  - [ ] `plot_quantile_comparison(ds, params)` - Compare source vs matched quantiles
  - [ ] `plot_transfer_function(params)` - Visualize the transfer function
  - [ ] `plot_cluster_map(classifier)` - Show spatial classification
  - [ ] `plot_fusion_diagnostics(fuser, ds)` - Comprehensive diagnostics

- [ ] **5.2 Add statistical diagnostics**
  - [ ] `calculate_fusion_metrics(ds_original, ds_fused, ds_reference)`
  - [ ] Support for: RMSE, Correlation, KGE, Bias, etc.
  - [ ] Per-cluster metrics

- [ ] **5.3 Add parameter persistence**
  - [ ] JSON export/import for fusion parameters
  - [ ] Pickle support for full model (including classifier)
  - [ ] Version tracking for model compatibility

### Phase 6: Testing & Documentation (Priority: High)

- [ ] **6.1 Unit Tests**
  - [ ] Test each module independently
  - [ ] Test edge cases (empty data, NaN handling, single timestep)
  - [ ] Test backward compatibility
  - [ ] Achieve >80% code coverage

- [ ] **6.2 Integration Tests**
  - [ ] Test complete fusion workflow
  - [ ] Test K-Means + Quantile Matching pipeline
  - [ ] Test land/sea separation
  - [ ] Test with real-world-like synthetic data

- [ ] **6.3 Documentation**
  - [ ] Update module docstrings
  - [ ] Add examples to each class/function
  - [ ] Create tutorial notebook for new features
  - [ ] Update TESTING_TODO.md with new tests

### Phase 7: Cleanup & Migration (Priority: Low)

- [ ] **7.1 Update `__init__.py`**
  - [ ] Export public API
  - [ ] Add `__all__` list
  - [ ] Maintain backward compatibility imports

- [ ] **7.2 Deprecation Warnings**
  - [ ] Add deprecation warnings to old patterns
  - [ ] Document migration path

- [ ] **7.3 Update notebooks**
  - [ ] Update `spp_correction_notebook.ipynb` to use new API
  - [ ] Create migration examples

---

## Implementation Guidelines

### Type Hints Example
```python
from typing import Optional, Callable, Dict, Any, Union, List
import numpy as np
import xarray as xr

def fit_qs(
    qs: xr.Dataset,
    fit_func: Callable[..., np.ndarray],
    char_pcp: str = 'radar'
) -> np.ndarray:
    """Fit a function to quantile data."""
    ...
```

### Base Class Example
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import xarray as xr

@dataclass
class FusionResult:
    """Container for fusion results."""
    matched_data: xr.Dataset
    parameters: Dict[str, Any]
    diagnostics: Optional[Dict[str, Any]] = None

class BaseFusionMethod(ABC):
    """Abstract base class for fusion methods."""
    
    @abstractmethod
    def fit(self, ds: xr.Dataset, **kwargs) -> Dict[str, Any]:
        """Calculate fusion parameters from training data."""
        pass
    
    @abstractmethod
    def transform(self, ds: xr.Dataset, params: Dict[str, Any]) -> xr.Dataset:
        """Apply fusion parameters to data."""
        pass
    
    def fit_transform(self, ds: xr.Dataset, **kwargs) -> FusionResult:
        """Fit parameters and transform in one step."""
        params = self.fit(ds, **kwargs)
        matched = self.transform(ds, params)
        return FusionResult(matched_data=matched, parameters=params)
```

### K-Means Classifier Example
```python
from sklearn.cluster import KMeans
import numpy as np
import xarray as xr

class KMeansClassifier:
    """K-Means spatial classifier for fusion."""
    
    def __init__(
        self,
        n_clusters: int = 3,
        features: List[str] = ['mean', 'std'],
        random_state: Optional[int] = None
    ):
        self.n_clusters = n_clusters
        self.features = features
        self.random_state = random_state
        self._kmeans = None
        self._labels = None
    
    def fit(self, ds: xr.Dataset) -> 'KMeansClassifier':
        """Fit K-Means to extracted features."""
        feature_matrix = self._extract_features(ds)
        self._kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        )
        self._labels = self._kmeans.fit_predict(feature_matrix)
        return self
    
    def get_cluster_masks(self, ds: xr.Dataset) -> Dict[int, xr.DataArray]:
        """Get boolean masks for each cluster."""
        masks = {}
        labels_2d = self._labels.reshape(ds.lat.size, ds.lon.size)
        for i in range(self.n_clusters):
            masks[i] = xr.DataArray(
                labels_2d == i,
                dims=['lat', 'lon'],
                coords={'lat': ds.lat, 'lon': ds.lon}
            )
        return masks
```

---

## Backward Compatibility

To ensure backward compatibility:

1. Keep `match_quantiles()` function with same signature
2. Keep `Fuser` class with same initialization
3. Add deprecation warnings for old patterns
4. Provide migration guide in documentation

### Deprecation Example
```python
import warnings

class Fuser:
    def __init__(self, fusing_params):
        # Check for old-style params
        if 'sep_land_sea' in fusing_params:
            warnings.warn(
                "'sep_land_sea' is deprecated, use classifier=LandSeaClassifier() instead",
                DeprecationWarning,
                stacklevel=2
            )
        # Continue with initialization...
```

---

## Dependencies

### Existing
- numpy
- scipy
- xarray

### New (for K-Means integration)
- scikit-learn (sklearn) - Already used in notebooks

### Optional
- global-land-mask (for land/sea classification)

---

## Timeline Estimate

| Phase | Effort | Priority |
|-------|--------|----------|
| Phase 1: Foundation | 2-3 days | High |
| Phase 2: Refactor Methods | 2-3 days | High |
| Phase 3: K-Means Integration | 3-4 days | High |
| Phase 4: Unified Fuser | 2-3 days | Medium |
| Phase 5: Enhanced Features | 2-3 days | Medium |
| Phase 6: Testing & Docs | 3-4 days | High |
| Phase 7: Cleanup | 1-2 days | Low |

**Total Estimated Effort**: 15-22 days

---

## Success Criteria

1. All existing tests pass
2. New tests achieve >80% coverage on merging module
3. K-Means classification works end-to-end
4. Notebooks can be updated to use new API
5. No breaking changes to public API (deprecations allowed)
6. Documentation is complete and up-to-date

---

## Notes

- The K-Means model (`notebooks/kmean_models/k_means_model_20240801_110831.pkl`) suggests this feature is already in use externally
- The `spp_correction_notebook.ipynb` imports sklearn's KMeans, confirming the need for integration
- Consider making sklearn an optional dependency with graceful fallback

---

## Questions for Discussion

1. Should we support multiple classifiers simultaneously (e.g., land/sea + k-means)?
2. What features should be extracted by default for K-Means classification?
3. Should we persist fitted models as pickle or use a custom format?
4. What level of backward compatibility is required?
5. Should visualization utilities be in the merging module or moved to `visualisation/`?

---

*Last Updated: December 2024*
*Author: PACHA Development Team*

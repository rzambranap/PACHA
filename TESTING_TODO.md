# PACHA Testing To-Do List

This document outlines the testing requirements to achieve comprehensive test coverage across the PACHA package.

## Current Testing Status

### Existing Tests
- [x] `tests/test_pacha.py` - Basic package import tests
- [x] `tests/test_generate_wiki.py` - Wiki generation script tests

### Testing Infrastructure
- [x] pytest configured
- [x] unittest support available
- [ ] Test coverage reporting (pytest-cov)
- [ ] Continuous Integration testing workflow

---

## Module Testing Requirements

### 1. Core Module (`pacha/core.py`)
- [ ] Core functionality tests (once implemented)
- [ ] Base class tests

### 2. Utils Package (`pacha/utils/`)

#### 2.1 Geospatial Utilities (`utils/geospatial.py`)
- [ ] **BBox Class Tests**
  - [ ] `__init__` - Test initialization with valid coordinates
  - [ ] `to_shape_compliant` - Test conversion to shapefile format
  - [ ] Test with edge cases (zero extent, crossing dateline, etc.)
  
- [ ] **Bounding Box Functions**
  - [ ] `get_data_in_bbox` - Test extraction from xarray with different lat orderings
  - [ ] `get_bbox_from_xarray` - Test bounding box extraction from xarray data
  - [ ] `get_bbox_from_gdf` - Test with GeoDataFrame inputs
  - [ ] `get_bbox_from_df` - Test with pandas DataFrame inputs
  - [ ] `get_surrounding_bbox_from_bboxs` - Test combining multiple bounding boxes
  - [ ] `get_extent_from_bbox` - Test conversion to matplotlib extent format
  
- [ ] **Land Mask Functions**
  - [ ] `add_land_mask` - Test adding land mask to xarray data
  - [ ] `divide_dataset_land_sea` - Test splitting dataset by land/sea
  
- [ ] **Grid Operations**
  - [ ] `subdivide` - Test grid subdivision
  - [ ] `calc_ds_center` - Test center calculation
  
- [ ] **Point Extraction Functions**
  - [ ] `get_val_coord` - Test value extraction at coordinates
  - [ ] `ds_to_df_at_point` - Test time series extraction

#### 2.2 Metrics (`utils/metrics.py`)
- [ ] **Statistical Metrics**
  - [ ] `kge` - Test Kling-Gupta Efficiency calculation
  - [ ] `psnr` - Test Peak Signal-to-Noise Ratio
  - [ ] `corr2_coeff` - Test 2D correlation
  - [ ] `pearsonr_2D` - Test Pearson correlation with 2D arrays
  
- [ ] **Contingency Score Functions**
  - [ ] `contingency_dataframe_timewise` - Test time-wise contingency
  - [ ] `calc_pixel_wise_contingency` - Test pixel-wise POD/FAR
  - [ ] `calc_area_wise_contingency` - Test area-wise POD/FAR
  - [ ] `calc_contingency` - Test general contingency calculation
  
- [ ] **Aggregation Functions**
  - [ ] `count_pixels_over_threshold` - Test pixel counting
  - [ ] `average_over_domain` - Test domain averaging
  - [ ] `calc_rel_bias` - Test relative bias calculation
  - [ ] `calculate_relative_bias` - Test array-based relative bias
  - [ ] `calc_corr_count_above_threshold` - Test correlation calculation
  - [ ] `kge_ds` - Test KGE for datasets
  
- [ ] **Coordinate Functions**
  - [ ] `get_val_coord` - Test value extraction
  - [ ] `get_coords_station` - Test station coordinate extraction
  
- [ ] **Time Series Functions**
  - [ ] `rad_sat_psnrs` - Test radar-satellite correlation time series

#### 2.3 Scores (`utils/scores.py`)
- [ ] **Basic Score Functions**
  - [ ] `calculate_correlation` - Test with/without NaN handling
  - [ ] `calculate_rmse` - Test RMSE calculation
  - [ ] `calculate_relative_bias` - Test relative bias
  - [ ] `calculate_kge` - Test KGE calculation
  - [ ] `calculate_r2score` - Test R-squared calculation
  - [ ] `calculate_contingency_scores` - Test POD/POND/FAR with various thresholds
  
- [ ] **Dataset Score Functions**
  - [ ] `score_rbias` - Test relative bias scoring
  - [ ] `score_rmse` - Test RMSE scoring
  - [ ] `score_kge` - Test KGE scoring
  - [ ] `score_contingency` - Test contingency scoring
  - [ ] `score_r2` - Test R-squared scoring
  - [ ] `score_correlation` - Test correlation scoring
  - [ ] `score_coeff_var_corr` - Test CV correlation
  - [ ] `score_support_size_corr` - Test support size correlation
  - [ ] `score_spatial_avg_corr` - Test spatial average correlation
  
- [ ] **Helper Functions**
  - [ ] `check_only_n_variables` - Test variable count validation
  - [ ] `apply_function_to_multi_input_dataset` - Test multi-variable application
  - [ ] `apply_function_to_dual_input` - Test dual-variable application
  - [ ] `get_remaining_field` - Test field extraction
  - [ ] `calculate_CV_for_ds` - Test coefficient of variation
  - [ ] `change_deepest_keys` - Test key remapping

#### 2.4 Temporal Utilities (`utils/temporal.py`)
- [ ] Review and add tests for temporal functions

#### 2.5 File Utilities (`utils/file_utils.py`)
- [ ] Review and add tests for file operations

#### 2.6 Instruments (`utils/instruments.py`)
- [ ] Review and add tests for instrument configurations

#### 2.7 ML Utilities (`utils/ml_utils.py`)
- [ ] Review and add tests for machine learning utilities

#### 2.8 Visualization (`utils/visualisation.py`)
- [ ] Review and add tests for visualization functions

### 3. Data Sources Package (`pacha/data_sources/`)

#### 3.1 Satellite Loaders (`data_sources/satellite_loaders.py`)
- [ ] `open_mf_imerg` - Test multi-file IMERG loading
- [ ] Test GSMaP loading functions
- [ ] Test error handling for missing files
- [ ] Test coordinate handling and subsetting

#### 3.2 Radar Loaders (`data_sources/radar_loaders.py`)
- [ ] Test radar data loading functions
- [ ] Test format handling
- [ ] Test coordinate transformations

#### 3.3 Gauge Loaders
- [ ] `gauges_funceme_csv.py` - Test FUNCEME CSV parsing
- [ ] `gauges_meteofrance.py` - Test Météo-France format parsing

#### 3.4 CML Readers (`data_sources/cml_readers.py`)
- [ ] Test Commercial Microwave Link data loading

#### 3.5 PyART Loaders (`data_sources/pyart_loaders.py`)
- [ ] Test PyART integration (optional dependency)

### 4. Merging Package (`pacha/merging/`)

#### 4.1 Quantile Matching (`merging/quantile_matching.py`)
- [ ] **Core Functions**
  - [ ] `fit_qs` - Test quantile fitting
  - [ ] `echelon_down` - Test smooth step-down function
  - [ ] `echelon_up` - Test smooth step-up function
  - [ ] `rescale_qs_to_calc` - Test quantile rescaling
  - [ ] `match_quantiles` - Test full quantile matching
  
- [ ] **Fuser Class**
  - [ ] Test initialization with different methods
  - [ ] `calculate_fusing_params` - Test parameter calculation
  - [ ] `apply_params` - Test parameter application
  - [ ] `export_matching_ds` - Test matched dataset export
  - [ ] `define_fields` - Test field definition
  - [ ] Test quantile matching method
  - [ ] Test quantile matching by parts method
  - [ ] Test linear regression method
  - [ ] Test land/sea separation option

### 5. L1 Processing Package (`pacha/L1_processing/`)

#### 5.1 Common (`L1_processing/common.py`)
- [ ] Review and add tests for common L1 functions

#### 5.2 Radar (`L1_processing/radar.py`)
- [ ] Test radar georeferencing functions
- [ ] Test format standardization

### 6. L2 Processing Package (`pacha/L2_processing/`)

#### 6.1 Radar (`L2_processing/radar.py`)
- [ ] Test reflectivity to rain rate conversion
- [ ] Test derived variable calculations

### 7. L3 Processing Package (`pacha/L3_processing/`)

#### 7.1 Regrid (`L3_processing/regrid.py`)
- [ ] Test regridding functions
- [ ] Test coordinate transformations

#### 7.2 Spatial Interpolation (`L3_processing/spatial_interpolation.py`)
- [ ] Test interpolation methods
- [ ] Test with different grid configurations

### 8. Analysis Package (`pacha/analysis/`)

#### 8.1 Data Extraction (`analysis/data_extraction.py`)
- [ ] Test data extraction functions
- [ ] Test filtering and selection

### 9. Visualization Package (`pacha/visualisation/`)

#### 9.1 Geospatial (`visualisation/geospatial.py`)
- [ ] Test map generation (can use image comparison or smoke tests)
- [ ] Test projection handling

#### 9.2 Statistical (`visualisation/statistical.py`)
- [ ] Test statistical plot generation

---

## Integration Tests

### Data Pipeline Integration
- [ ] Test complete L1 → L2 → L3 pipeline
- [ ] Test data loading → processing → visualization workflow
- [ ] Test fusion workflow end-to-end

### Cross-Module Integration
- [ ] Test utils integration with data_sources
- [ ] Test merging integration with analysis
- [ ] Test visualization with processed data

---

## Test Categories

### Unit Tests (Priority: High)
- [ ] All public functions in utils/
- [ ] All public functions in merging/
- [ ] All public functions in analysis/

### Integration Tests (Priority: Medium)
- [ ] Data loading workflows
- [ ] Processing pipelines
- [ ] Fusion workflows

### Edge Case Tests (Priority: Medium)
- [ ] Empty datasets
- [ ] NaN/missing data handling
- [ ] Boundary conditions
- [ ] Invalid input handling

### Performance Tests (Priority: Low)
- [ ] Large dataset handling
- [ ] Memory usage monitoring
- [ ] Processing time benchmarks

---

## Testing Infrastructure Improvements

### Immediate Priorities
- [ ] Add pytest-cov for coverage reporting
- [ ] Create pytest.ini or pyproject.toml configuration
- [ ] Add test fixtures for common test data
- [ ] Create mock data generators for testing

### CI/CD Integration
- [ ] Create GitHub Actions workflow for automated testing
- [ ] Add coverage badge to README
- [ ] Configure test reporting

### Test Data Management
- [ ] Create small synthetic test datasets
- [ ] Document test data requirements
- [ ] Add test data download scripts if needed

---

## Coverage Goals

| Module | Current Coverage | Target Coverage |
|--------|-----------------|-----------------|
| `pacha/utils/` | Low | 80% |
| `pacha/merging/` | None | 80% |
| `pacha/analysis/` | None | 70% |
| `pacha/data_sources/` | None | 60% |
| `pacha/L1_processing/` | None | 60% |
| `pacha/L2_processing/` | None | 60% |
| `pacha/L3_processing/` | None | 60% |
| `pacha/visualisation/` | None | 50% |
| **Overall** | Low | 70% |

---

## Priority Order for Implementation

### Phase 1: Core Utilities (Highest Priority)
1. `utils/scores.py` - Basic statistical functions
2. `utils/metrics.py` - Performance metrics
3. `utils/geospatial.py` - Geospatial operations

### Phase 2: Data Fusion
4. `merging/quantile_matching.py` - Core fusion functionality

### Phase 3: Processing Pipelines
5. `L3_processing/` - Regridding and interpolation
6. `L2_processing/` - Derived variables
7. `L1_processing/` - Georeferencing

### Phase 4: Data Sources
8. `data_sources/satellite_loaders.py`
9. `data_sources/radar_loaders.py`
10. Other data loaders

### Phase 5: Analysis and Visualization
11. `analysis/data_extraction.py`
12. `visualisation/` modules

---

## Testing Best Practices

1. **Test Naming**: Use descriptive names like `test_kge_with_perfect_correlation()`
2. **Arrange-Act-Assert**: Structure tests clearly
3. **Mock External Dependencies**: Use mocking for file I/O and external libraries
4. **Parametrize Tests**: Use pytest parametrize for multiple test cases
5. **Documentation**: Add docstrings to test functions explaining what they test
6. **Isolation**: Each test should be independent

---

## Notes

- Some modules require optional dependencies (e.g., `global_land_mask`, `geopandas`, `pyart`)
- Tests requiring optional dependencies should be marked with appropriate pytest markers
- Consider using `pytest.importorskip()` for optional dependency tests
- Visualization tests may require `matplotlib.use('Agg')` for headless testing

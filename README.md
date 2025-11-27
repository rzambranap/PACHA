# PACHA

**P**recipitation **A**nalysis & **C**orrection for **H**ydrological **A**pplications

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

PACHA is a Python package for analyzing, correcting, and validating precipitation data from various sources including rain gauges, weather radars, satellites, and commercial microwave links.

## Features

- **Data Handling**: Load and process precipitation data from multiple file formats (CSV, NetCDF)
- **Bias Correction**: Multiple methods for correcting systematic biases in precipitation estimates
  - Mean Field Bias
  - Linear Scaling
  - Quantile Mapping
  - Local Intensity Scaling
  - Distribution Mapping
- **Validation**: Comprehensive metrics for evaluating precipitation estimates
  - Continuous metrics (RMSE, MAE, NSE, KGE)
  - Categorical scores (POD, FAR, CSI, HSS)

## Installation

```bash
# Clone the repository
git clone https://github.com/rzambranap/PACHA.git
cd PACHA

# Install in development mode
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
import pacha

# Create sample data
estimates = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
reference = np.array([0.8, 1.5, 2.5, 3.5, 5.0])

# Apply mean field bias correction
corrected = pacha.correction.mean_field_bias(estimates, reference)

# Validate the corrected estimates
metrics = pacha.validation.compute_metrics(reference, corrected)
print(f"RMSE: {metrics['rmse']:.2f} mm")
print(f"NSE: {metrics['nse']:.3f}")
```

## Modules

### `pacha.core`
Core functionality and base classes for precipitation data handling.

```python
from pacha.core import PrecipitationData, TimeSeriesData, GriddedData

# Create precipitation time series
import pandas as pd
times = pd.date_range("2024-01-01", periods=24, freq="h")
values = np.random.exponential(scale=1.0, size=24)

ts = TimeSeriesData(values=values, times=times, units="mm/h")
print(f"Total: {ts.total():.1f} mm")
```

### `pacha.data`
Data loading, processing, and transformation utilities.

```python
from pacha.data import load_timeseries, create_synthetic_timeseries

# Load data from CSV
precip = load_timeseries("rainfall.csv", time_column="datetime", value_column="precip")

# Create synthetic data for testing
synthetic = create_synthetic_timeseries(
    start="2024-01-01",
    end="2024-12-31",
    freq="h",
    mean_intensity=2.0,
    wet_fraction=0.05
)
```

### `pacha.correction`
Precipitation bias correction and adjustment methods.

```python
from pacha.correction import (
    mean_field_bias,
    quantile_mapping,
    local_intensity_scaling
)

# Apply different correction methods
corrected_mfb = mean_field_bias(estimates, reference)
corrected_qm = quantile_mapping(estimates, reference, n_quantiles=100)
corrected_lis = local_intensity_scaling(estimates, reference, n_bins=10)
```

### `pacha.validation`
Validation metrics and evaluation tools.

```python
from pacha.validation import (
    compute_metrics,
    rmse, mae, bias,
    nash_sutcliffe, kling_gupta,
    categorical_scores
)

# Compute all metrics at once
metrics = compute_metrics(observed, predicted)

# Or compute individual metrics
rmse_val = rmse(observed, predicted)
nse = nash_sutcliffe(observed, predicted)
kge = kling_gupta(observed, predicted)

# Categorical verification
cat_scores = categorical_scores(observed, predicted, threshold=0.1)
print(f"POD: {cat_scores['pod']:.2f}")
print(f"FAR: {cat_scores['far']:.2f}")
```

## Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- xarray >= 0.19.0

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check pacha/ tests/

# Type checking
mypy pacha/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Rodrigo Zambrana (rodrizp@gmail.com)
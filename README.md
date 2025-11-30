# PACHA

**Precipitation Analysis & Correction for Hydrological Applications**

A Python package for precipitation data fusion and analysis, combining data from satellites, weather radars, rain gauges, and commercial microwave links for improved precipitation estimation.

## Features

- **Data Loading**: Support for multiple precipitation data sources:
  - Satellite products (IMERG, GSMaP)
  - Weather radar data (various formats via xradar and Py-ART)
  - Rain gauge networks (FUNCEME, Météo-France formats)
  - Commercial Microwave Links (CML)

- **Data Processing**: Multi-level processing pipeline:
  - L1: Georeferencing and format standardization
  - L2: Derived geophysical variables (reflectivity to rain rate)
  - L3: Spatial interpolation and regridding

- **Data Fusion**: Quantile matching and bias correction algorithms for merging multiple data sources

- **Analysis Tools**: Statistical metrics and scoring functions for validation

- **Visualization**: Geospatial mapping and statistical plotting utilities

## Installation

### Basic Installation

```bash
pip install -e .
pip install ipykernel
pip install haversine
pip install global_land_mask
```


### Full Installation (with all optional dependencies)

```bash
pip install -e ".[full]"
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Conda Environment (Recommended)

For radar data support, it's recommended to use a conda environment:

```bash
conda create --name pacha python=3.10
conda activate pacha
conda install -c conda-forge xradar cartopy geopandas
pip install -e .
```

## Package Structure

```
pacha/
├── L1_processing/     # Level 1 processing (georeferencing)
├── L2_processing/     # Level 2 processing (derived variables)
├── L3_processing/     # Level 3 processing (regridding, interpolation)
├── analysis/          # Data analysis utilities
├── data_sources/      # Data loaders for various sources
├── merging/           # Data fusion algorithms
├── utils/             # General utilities
└── visualisation/     # Visualization tools
```

## Data Level Conventions

Based loosely on NASA data level conventions:

- **L0**: Raw data files in their original formats
- **L1R/L1S**: Easy-to-read files with geo-referenced coordinates
- **L2A/L2S**: Derived geophysical variables at L1 resolution
- **L3**: Variables mapped to uniform space-time grid
- **L4**: Merged products with fusion metadata

## Quick Start

```python
import pacha
from pacha.utils.geospatial import BBox, get_data_in_bbox
from pacha.data_sources.satellite_loaders import open_mf_imerg

# Define region of interest
bbox = BBox(min_lat=-6.0, min_lon=-40.0, max_lat=-4.0, max_lon=-38.0)

# Load satellite data
satellite_data = open_mf_imerg(file_paths, field='precipitationCal')

# Extract region
regional_data = get_data_in_bbox(satellite_data, bbox)
```

## Documentation

### Wiki

The package documentation is automatically generated from docstrings and published to the [GitHub Wiki](../../wiki). The wiki is updated automatically when changes are pushed to the `main` branch.

To generate the wiki documentation locally:

```bash
python scripts/generate_wiki.py
```

This creates Markdown files in the `wiki/` directory.

### Interactive Documentation

To browse the package documentation interactively:

```bash
python -m pydoc -b
```

Then navigate to the `pacha` package.

## Testing

Run unit tests with:

```bash
python -m unittest discover tests/
```

Or with pytest:

```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure they pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Authors

- Rodrigo Zambrana (rodrizp@gmail.com)

# TODO: Package Installation for PACHA

This document provides a to-do list for installing packages that are imported in the PACHA modules.

## Core Dependencies (Required for Basic Functionality)

The following packages are listed in `requirements.txt` and `setup.py` as core dependencies:

- [x] **pandas** - Data manipulation and analysis
- [x] **matplotlib** - Plotting and visualization
- [x] **numpy** - Numerical operations
- [x] **xarray** - n-dimensional labeled arrays
- [x] **scipy** - Scientific computing (interpolation, optimization)

### Installation:
```bash
pip install pandas matplotlib numpy xarray scipy
```

Or simply:
```bash
pip install -r requirements.txt
```

---

## Full Installation Dependencies (Optional - for Complete Functionality)

The following packages are imported in specific modules but require the "full" installation option. These are needed for advanced features like radar data processing, geospatial analysis, and land/sea masking.

### Radar Data Processing

| Package | Used In | Purpose |
|---------|---------|---------|
| [ ] **xradar** | `L1_processing/radar.py`, `data_sources/radar_loaders.py` | Reading and processing radar data files (IRIS, GAMIC, Rainbow formats) |
| [ ] **pyart** | `data_sources/pyart_loaders.py` | Py-ART radar data loading and processing |

### Gauge Data Processing

| Package | Used In | Purpose |
|---------|---------|---------|
| [ ] **lat-lon-parser** | `data_sources/gauges_meteofrance.py` | Parsing latitude/longitude coordinate strings |

### Geospatial Utilities

| Package | Used In | Purpose |
|---------|---------|---------|
| [ ] **haversine** | `utils/instruments.py` | Calculate distances between geographic coordinates |
| [ ] **global-land-mask** | `utils/geospatial.py` | Land/sea masking for geospatial data |

### Visualization

| Package | Used In | Purpose |
|---------|---------|---------|
| [ ] **cartopy** | `visualisation/geospatial.py` | Map projections and geospatial plotting |
| [ ] **pyproj** | `visualisation/geospatial.py` | Coordinate transformations |
| [ ] **shapely** | `visualisation/geospatial.py` | Geometric operations |
| [ ] **geopandas** | `visualisation/geospatial.py` | Geospatial data frames |

### Additional Full Installation Dependencies

| Package | Purpose |
|---------|---------|
| [ ] **cython** | Performance optimization for some packages |

### Installation:
```bash
pip install -e ".[full]"
```

Or install individual packages:
```bash
pip install xradar pyart cartopy geopandas haversine global-land-mask lat-lon-parser pyproj shapely cython
```

---

## Development Dependencies

The following packages are needed for development, testing, and code quality:

| Package | Purpose |
|---------|---------|
| [ ] **setuptools** | Package building and installation |
| [ ] **notebook** | Jupyter notebook support |
| [ ] **black** | Code formatting |
| [ ] **flake8** | Linting |
| [ ] **pre-commit** | Git hooks for code quality |
| [ ] **pytest** | Testing framework |

### Installation:
```bash
pip install -e ".[dev]"
```

---

## Conda Installation (Recommended for Radar Support)

For radar data support, using conda is recommended due to complex dependencies:

```bash
# Create environment
conda create --name pacha python="3.10"
conda activate pacha

# Install packages that work better with conda
conda install -c conda-forge xradar cartopy geopandas

# Install PACHA
pip install -e .
```

---

## Quick Reference: Installation Options

| Installation Type | Command | What You Get |
|-------------------|---------|--------------|
| Basic | `pip install -e .` | Core functionality (pandas, numpy, xarray, scipy, matplotlib) |
| Full | `pip install -e ".[full]"` | All features including radar, geospatial, visualization |
| Development | `pip install -e ".[dev]"` | Testing and development tools |
| Full + Dev | `pip install -e ".[full,dev]"` | Everything |

---

## Module-Package Dependency Map

Below is a complete mapping of which modules depend on which packages:

### Always Required (Core)
- All modules use: `numpy`, `pandas`, `xarray`, `scipy`, `matplotlib`

### Conditionally Required (Full Installation)

| Module | Optional Package Required |
|--------|---------------------------|
| `L1_processing/radar.py` | xradar |
| `data_sources/radar_loaders.py` | xradar |
| `data_sources/pyart_loaders.py` | pyart |
| `data_sources/gauges_meteofrance.py` | lat-lon-parser |
| `utils/instruments.py` | haversine |
| `utils/geospatial.py` | global-land-mask |
| `visualisation/geospatial.py` | cartopy, pyproj, shapely, geopandas |

---

## Notes

1. **Import Errors**: If you encounter import errors when using specific functionality, check which optional package is needed from the table above.

2. **Minimal Installation**: For basic precipitation data analysis without radar support, the core installation is sufficient.

3. **Radar Data**: Working with radar data (IRIS, GAMIC, Rainbow formats) requires `xradar`. Working with Py-ART compatible formats requires `pyart`.

4. **Mapping**: Creating publication-quality maps requires `cartopy`, `geopandas`, and related packages.

5. **Platform Compatibility**: Some packages (especially `cartopy` and `pyart`) can be tricky to install on certain platforms. Using conda often resolves these issues.

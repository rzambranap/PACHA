"""
Setup script for PACHA package.

PACHA - Precipitation Analysis & Correction for Hydrological Applications
"""

from setuptools import setup, find_packages

setup(
    name='pacha',
    version='0.0.1',
    description='Precipitation Analysis & Correction for Hydrological Applications - '
                'Package to fuse data from Satellites, weather radars and commercial '
                'microwave links for more precise precipitation estimations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Rodrigo Zambrana',
    author_email='rodrizp@gmail.com',
    url='https://github.com/rzambranap/PACHA',
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    install_requires=[
        'pandas',
        'matplotlib',
        'numpy',
        'xarray',
        'scipy',
    ],
    extras_require={
        'full': [
            'cython',
            'pyart',
            'xradar',
            'cartopy',
            'geopandas',
            'haversine',
            'global-land-mask',
            'lat-lon-parser',
            'pyproj',
            'shapely',
        ],
        'dev': [
            'setuptools',
            'notebook',
            'black',
            'flake8',
            'pre-commit',
            'pytest',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Hydrology",
    ],
    python_requires='>=3.8',
)

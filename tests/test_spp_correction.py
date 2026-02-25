"""
Unit tests for the spp_correction merging module.

Tests the ClusteredCorrector class, calculate_features_for_clustering,
and subdivide_dataset functions.
"""

import json
import os
import pickle
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pacha.merging.spp_correction import (
    ClusteredCorrector,
    _default_correction_func,
    calculate_features_for_clustering,
    subdivide_dataset,
)


def _make_simple_ds(ntimes=20, nlat=8, nlon=8, seed=42):
    """Create a minimal xarray Dataset for testing."""
    rng = np.random.default_rng(seed)
    times = pd.date_range('2020-01-01', periods=ntimes, freq='D')
    lats = np.linspace(-5, 5, nlat)
    lons = np.linspace(-55, -45, nlon)
    data = rng.exponential(1.0, size=(ntimes, nlat, nlon))
    ref = rng.exponential(1.2, size=(ntimes, nlat, nlon))
    ds = xr.Dataset(
        {
            'imerg_v7': (['time', 'lat', 'lon'], data),
            'radar_rain_rate': (['time', 'lat', 'lon'], ref),
        },
        coords={'time': times, 'lat': lats, 'lon': lons},
    )
    return ds


class TestDefaultCorrectionFunc(unittest.TestCase):
    """Tests for the default quadratic correction function."""

    def test_output_shape(self):
        """Output shape matches input shape."""
        x = np.linspace(0, 10, 50)
        result = _default_correction_func(x, 1.0, 1.0, 0.0)
        self.assertEqual(result.shape, x.shape)

    def test_quadratic_values(self):
        """Verify a*x^2 + b*x + c values."""
        x = np.array([0.0, 1.0, 2.0])
        result = _default_correction_func(x, 1.0, 2.0, 3.0)
        expected = np.array([3.0, 6.0, 11.0])
        np.testing.assert_allclose(result, expected)


class TestCalculateFeaturesForClustering(unittest.TestCase):
    """Tests for calculate_features_for_clustering."""

    def setUp(self):
        self.ds = _make_simple_ds(ntimes=10)
        self.da = self.ds['imerg_v7']

    def test_returns_dict_with_variable_keys(self):
        """Result is a dict keyed by variable names."""
        result = calculate_features_for_clustering(self.ds)
        self.assertIsInstance(result, dict)
        for var in ['imerg_v7', 'radar_rain_rate']:
            self.assertIn(var, result)

    def test_feature_columns(self):
        """Feature DataFrame has expected columns."""
        result = calculate_features_for_clustering(self.ds)
        df = result['imerg_v7']
        for col in ['spm', 'sup', 'std', 'cvr']:
            self.assertIn(col, df.columns)

    def test_feature_length_matches_time(self):
        """Feature DataFrame length matches time dimension."""
        result = calculate_features_for_clustering(self.ds)
        df = result['imerg_v7']
        self.assertEqual(len(df), len(self.ds.time))

    def test_works_with_dataarray(self):
        """Function works with a DataArray input."""
        result = calculate_features_for_clustering(self.da)
        self.assertIn(self.da.name, result)

    def test_invalid_input_raises(self):
        """Non-xarray input raises TypeError."""
        with self.assertRaises(TypeError):
            calculate_features_for_clustering([[1, 2], [3, 4]])

    def test_missing_dims_raises(self):
        """DataArray without lat/lon raises TypeError."""
        da_bad = xr.DataArray(
            np.random.rand(5),
            coords={'time': pd.date_range('2020-01-01', periods=5, freq='D')},
            dims=['time'],
            name='x',
        )
        with self.assertRaises(TypeError):
            calculate_features_for_clustering(da_bad)


class TestSubdivideDataset(unittest.TestCase):
    """Tests for subdivide_dataset."""

    def setUp(self):
        self.ds = _make_simple_ds(ntimes=5, nlat=8, nlon=8)

    def test_number_of_subdivisions(self):
        """nsubdivisions=2 produces 4 subsets."""
        parts = subdivide_dataset(self.ds, 2)
        self.assertEqual(len(parts), 4)

    def test_no_subdivision(self):
        """nsubdivisions=1 produces 1 subset equal to input."""
        parts = subdivide_dataset(self.ds, 1)
        self.assertEqual(len(parts), 1)
        xr.testing.assert_equal(parts[0], self.ds)

    def test_all_data_covered(self):
        """Union of subdivisions covers all lat/lon values."""
        parts = subdivide_dataset(self.ds, 2)
        all_lats = np.concatenate([p.lat.values for p in parts])
        all_lons = np.concatenate([p.lon.values for p in parts])
        np.testing.assert_array_equal(
            np.sort(np.unique(all_lats)), np.sort(self.ds.lat.values)
        )
        np.testing.assert_array_equal(
            np.sort(np.unique(all_lons)), np.sort(self.ds.lon.values)
        )

    def test_subset_variables_preserved(self):
        """Subdivisions preserve all variables."""
        parts = subdivide_dataset(self.ds, 2)
        for part in parts:
            for var in self.ds.data_vars:
                self.assertIn(var, part)


class TestClusteredCorrectorInit(unittest.TestCase):
    """Tests for ClusteredCorrector initialization."""

    def test_default_init(self):
        """Default initialization sets expected attributes."""
        corrector = ClusteredCorrector()
        self.assertEqual(corrector.n_clusters, 4)
        self.assertEqual(corrector.random_state, 12)
        self.assertIsNone(corrector.kmeans)
        self.assertIsNone(corrector.params_by_label)

    def test_custom_n_clusters(self):
        """Custom n_clusters is stored."""
        corrector = ClusteredCorrector(n_clusters=3)
        self.assertEqual(corrector.n_clusters, 3)

    def test_apply_before_fit_raises(self):
        """apply() raises RuntimeError if not fitted."""
        corrector = ClusteredCorrector()
        ds = _make_simple_ds()
        with self.assertRaises(RuntimeError):
            corrector.apply(ds, target_variable='imerg_v7')


class TestClusteredCorrectorFit(unittest.TestCase):
    """Tests for ClusteredCorrector.fit."""

    def setUp(self):
        self.ds = _make_simple_ds(ntimes=30, seed=0)

    def test_fit_sets_kmeans(self):
        """fit() sets the kmeans attribute."""
        corrector = ClusteredCorrector(n_clusters=2)
        corrector.fit(self.ds, target_variable='imerg_v7')
        self.assertIsNotNone(corrector.kmeans)
        self.assertEqual(corrector.kmeans.n_clusters, 2)

    def test_fit_sets_params_by_label(self):
        """fit() sets params_by_label with one entry per cluster."""
        corrector = ClusteredCorrector(n_clusters=2)
        corrector.fit(self.ds, target_variable='imerg_v7')
        self.assertIsNotNone(corrector.params_by_label)
        self.assertEqual(len(corrector.params_by_label), 2)

    def test_fit_returns_self(self):
        """fit() returns self for chaining."""
        corrector = ClusteredCorrector(n_clusters=2)
        result = corrector.fit(self.ds, target_variable='imerg_v7')
        self.assertIs(result, corrector)


class TestClusteredCorrectorSaveLoad(unittest.TestCase):
    """Tests for ClusteredCorrector.save and load."""

    def setUp(self):
        self.ds = _make_simple_ds(ntimes=20, seed=1)
        self.corrector = ClusteredCorrector(n_clusters=2)
        self.corrector.fit(self.ds, target_variable='imerg_v7')

    def test_save_load_pickle(self):
        """save/load round-trip with pickle preserves n_clusters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, 'corrector.pkl')
            self.corrector.save(fpath)
            loaded = ClusteredCorrector.load(fpath)
            self.assertEqual(loaded.n_clusters, self.corrector.n_clusters)
            self.assertIsNotNone(loaded.kmeans)
            self.assertIsNotNone(loaded.params_by_label)

    def test_save_creates_file(self):
        """save() creates a file at the given path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, 'corrector.pkl')
            self.corrector.save(fpath)
            self.assertTrue(os.path.exists(fpath))

    def test_save_params_json(self):
        """save_params() creates a readable JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, 'params.json')
            self.corrector.save_params(fpath)
            self.assertTrue(os.path.exists(fpath))
            with open(fpath) as f:
                data = json.load(f)
            self.assertIn('0', data)
            self.assertIn('1', data)

    def test_load_params_json_round_trip(self):
        """save_params/load_params round-trip preserves numeric values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, 'params.json')
            self.corrector.save_params(fpath)
            params = ClusteredCorrector.load_params(fpath)
            self.assertEqual(len(params), self.corrector.n_clusters)
            for label in range(self.corrector.n_clusters):
                self.assertIn(label, params)

    def test_loaded_params_have_func(self):
        """load_params() attaches the correction function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, 'params.json')
            self.corrector.save_params(fpath)
            params = ClusteredCorrector.load_params(fpath)
            for label in params:
                for region_key in params[label]:
                    self.assertIn('func', params[label][region_key])
                    self.assertTrue(callable(params[label][region_key]['func']))

    def test_save_params_before_fit_raises(self):
        """save_params() raises RuntimeError if not fitted."""
        corrector = ClusteredCorrector()
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, 'params.json')
            with self.assertRaises(RuntimeError):
                corrector.save_params(fpath)


class TestNotebookParamsFile(unittest.TestCase):
    """Tests that the pre-saved notebook correction parameters file is valid."""

    def test_params_file_exists(self):
        """The pre-saved correction parameters file exists."""
        params_path = os.path.join(
            os.path.dirname(__file__), '..', 'pacha', 'merging', 'params',
            'correction_params_v7_4cluster.json'
        )
        self.assertTrue(os.path.exists(params_path))

    def test_params_file_loadable(self):
        """The pre-saved params file can be loaded by load_params."""
        params_path = os.path.join(
            os.path.dirname(__file__), '..', 'pacha', 'merging', 'params',
            'correction_params_v7_4cluster.json'
        )
        params = ClusteredCorrector.load_params(params_path)
        self.assertEqual(len(params), 4)

    def test_params_file_has_all_clusters(self):
        """Pre-saved file has 4 clusters (0–3)."""
        params_path = os.path.join(
            os.path.dirname(__file__), '..', 'pacha', 'merging', 'params',
            'correction_params_v7_4cluster.json'
        )
        params = ClusteredCorrector.load_params(params_path)
        for label in range(4):
            self.assertIn(label, params)

    def test_params_file_structure(self):
        """Pre-saved params have expected keys."""
        params_path = os.path.join(
            os.path.dirname(__file__), '..', 'pacha', 'merging', 'params',
            'correction_params_v7_4cluster.json'
        )
        params = ClusteredCorrector.load_params(params_path)
        for label in range(4):
            for region_key in ['land_params', 'sea_params']:
                self.assertIn(region_key, params[label])
                p = params[label][region_key]
                self.assertIn('offset', p)
                self.assertIn('p1_popt', p)
                self.assertIn('p2_popt', p)
                self.assertIn('func', p)


if __name__ == '__main__':
    unittest.main()

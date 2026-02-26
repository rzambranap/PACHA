"""
Tests for scripts/run_spp_correction.py.

Tests cover argument parsing, input validation, and end-to-end execution
using a temporary dataset and a temporary corrector pickle.
"""

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Make scripts/ importable when running tests from the repo root.
_REPO_ROOT = Path(__file__).parent.parent
_SCRIPTS_DIR = _REPO_ROOT / 'scripts'
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import run_spp_correction  # noqa: E402  (import after path manipulation)
from pacha.merging.spp_correction import ClusteredCorrector  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_ds(ntimes=20, nlat=8, nlon=8, seed=42):
    """Create a minimal xarray Dataset for testing."""
    rng = np.random.default_rng(seed)
    times = pd.date_range('2024-01-01', periods=ntimes, freq='D')
    lats = np.linspace(-5, 5, nlat)
    lons = np.linspace(-55, -45, nlon)
    data = rng.exponential(1.0, size=(ntimes, nlat, nlon))
    ref = rng.exponential(1.2, size=(ntimes, nlat, nlon))
    return xr.Dataset(
        {
            'imerg_v7': (['time', 'lat', 'lon'], data),
            'radar_rain_rate': (['time', 'lat', 'lon'], ref),
        },
        coords={'time': times, 'lat': lats, 'lon': lons},
    )


def _make_trained_corrector(ds):
    """Return a fitted ClusteredCorrector (n_clusters=2 for speed)."""
    corrector = ClusteredCorrector(n_clusters=2)
    corrector.fit(ds, target_variable='imerg_v7')
    return corrector


# ---------------------------------------------------------------------------
# Argument-parsing tests
# ---------------------------------------------------------------------------

class TestParseArgs(unittest.TestCase):
    """Tests for run_spp_correction.parse_args()."""

    def _base_args(self):
        return [
            '--input-folder', '/some/in',
            '--corrector-pkl', '/some/corrector.pkl',
            '--start-date', '2024-01-01',
            '--end-date', '2024-01-31',
            '--target-variable', 'imerg_v7',
            '--output-folder', '/some/out',
        ]

    def test_required_args_parsed(self):
        """All required arguments are parsed correctly."""
        args = run_spp_correction.parse_args(self._base_args())
        self.assertEqual(args.input_folder, '/some/in')
        self.assertEqual(args.corrector_pkl, '/some/corrector.pkl')
        self.assertEqual(args.start_date, '2024-01-01')
        self.assertEqual(args.end_date, '2024-01-31')
        self.assertEqual(args.target_variable, 'imerg_v7')
        self.assertEqual(args.output_folder, '/some/out')

    def test_default_optional_args(self):
        """Optional arguments have correct defaults."""
        args = run_spp_correction.parse_args(self._base_args())
        self.assertIsNone(args.params_json)
        self.assertEqual(args.input_glob, '*.nc')
        self.assertEqual(args.nsubdivisions, 2)
        self.assertIsNone(args.output_filename)
        self.assertFalse(args.overwrite)

    def test_optional_args_override(self):
        """Optional arguments can be overridden."""
        argv = self._base_args() + [
            '--params-json', '/some/params.json',
            '--input-glob', '*.nc4',
            '--nsubdivisions', '1',
            '--output-filename', 'out.nc',
            '--overwrite',
        ]
        args = run_spp_correction.parse_args(argv)
        self.assertEqual(args.params_json, '/some/params.json')
        self.assertEqual(args.input_glob, '*.nc4')
        self.assertEqual(args.nsubdivisions, 1)
        self.assertEqual(args.output_filename, 'out.nc')
        self.assertTrue(args.overwrite)

    def test_missing_required_arg_raises(self):
        """Missing required argument causes SystemExit."""
        # Drop --target-variable
        argv = [a for a in self._base_args() if a != 'imerg_v7' and a != '--target-variable']
        with self.assertRaises(SystemExit):
            run_spp_correction.parse_args(argv)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestMainValidation(unittest.TestCase):
    """Tests that main() exits gracefully on bad inputs."""

    def test_missing_input_folder_exits(self):
        """Non-existent input folder causes SystemExit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, 'c.pkl')
            # Create a dummy pkl so that check doesn't fail first
            corrector = ClusteredCorrector(n_clusters=2)
            corrector.fit(_make_simple_ds(), target_variable='imerg_v7')
            corrector.save(pkl_path)

            argv = [
                '--input-folder', os.path.join(tmpdir, 'nonexistent'),
                '--corrector-pkl', pkl_path,
                '--start-date', '2024-01-01',
                '--end-date', '2024-01-31',
                '--target-variable', 'imerg_v7',
                '--output-folder', tmpdir,
            ]
            with self.assertRaises(SystemExit):
                run_spp_correction.main(argv)

    def test_missing_corrector_pkl_exits(self):
        """Non-existent corrector pickle causes SystemExit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                '--input-folder', tmpdir,
                '--corrector-pkl', os.path.join(tmpdir, 'nonexistent.pkl'),
                '--start-date', '2024-01-01',
                '--end-date', '2024-01-31',
                '--target-variable', 'imerg_v7',
                '--output-folder', tmpdir,
            ]
            with self.assertRaises(SystemExit):
                run_spp_correction.main(argv)

    def test_no_matching_files_exits(self):
        """No files matching input-glob causes SystemExit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = _make_simple_ds()
            corrector = _make_trained_corrector(ds)
            pkl_path = os.path.join(tmpdir, 'corrector.pkl')
            corrector.save(pkl_path)

            argv = [
                '--input-folder', tmpdir,
                '--corrector-pkl', pkl_path,
                '--input-glob', '*.nc',
                '--start-date', '2024-01-01',
                '--end-date', '2024-01-31',
                '--target-variable', 'imerg_v7',
                '--output-folder', tmpdir,
            ]
            with self.assertRaises(SystemExit):
                run_spp_correction.main(argv)

    def test_overwrite_flag_prevents_clobber(self):
        """Without --overwrite, existing output file causes SystemExit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            in_dir = os.path.join(tmpdir, 'in')
            out_dir = os.path.join(tmpdir, 'out')
            os.makedirs(in_dir)
            os.makedirs(out_dir)

            ds = _make_simple_ds()
            corrector = _make_trained_corrector(ds)
            pkl_path = os.path.join(tmpdir, 'corrector.pkl')
            corrector.save(pkl_path)
            nc_path = os.path.join(in_dir, 'data.nc')
            ds.to_netcdf(nc_path)

            # Pre-create the output file
            expected_out = os.path.join(out_dir, 'spp_corrected_2024-01-01_2024-01-20.nc')
            open(expected_out, 'w').close()

            argv = [
                '--input-folder', in_dir,
                '--corrector-pkl', pkl_path,
                '--start-date', '2024-01-01',
                '--end-date', '2024-01-20',
                '--target-variable', 'imerg_v7',
                '--output-folder', out_dir,
            ]
            with self.assertRaises(SystemExit):
                run_spp_correction.main(argv)


# ---------------------------------------------------------------------------
# End-to-end test
# ---------------------------------------------------------------------------

class TestMainEndToEnd(unittest.TestCase):
    """End-to-end tests that produce an output NetCDF file."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.in_dir = os.path.join(self.tmpdir, 'in')
        self.out_dir = os.path.join(self.tmpdir, 'out')
        os.makedirs(self.in_dir)

        self.ds = _make_simple_ds(ntimes=20)
        self.corrector = _make_trained_corrector(self.ds)

        self.pkl_path = os.path.join(self.tmpdir, 'corrector.pkl')
        self.corrector.save(self.pkl_path)

        self.nc_path = os.path.join(self.in_dir, 'data.nc')
        self.ds.to_netcdf(self.nc_path)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _base_argv(self):
        return [
            '--input-folder', self.in_dir,
            '--corrector-pkl', self.pkl_path,
            '--start-date', '2024-01-01',
            '--end-date', '2024-01-20',
            '--target-variable', 'imerg_v7',
            '--output-folder', self.out_dir,
            '--nsubdivisions', '1',
        ]

    def test_output_file_created(self):
        """main() creates an output NetCDF file."""
        run_spp_correction.main(self._base_argv())
        expected = os.path.join(self.out_dir, 'spp_corrected_2024-01-01_2024-01-20.nc')
        self.assertTrue(os.path.exists(expected))

    def test_output_filename_override(self):
        """Custom --output-filename is used."""
        argv = self._base_argv() + ['--output-filename', 'custom.nc']
        run_spp_correction.main(argv)
        expected = os.path.join(self.out_dir, 'custom.nc')
        self.assertTrue(os.path.exists(expected))

    def test_output_folder_created(self):
        """Output folder is created if it does not exist."""
        new_out = os.path.join(self.tmpdir, 'new_out', 'nested')
        argv = self._base_argv()
        argv[argv.index(self.out_dir)] = new_out
        run_spp_correction.main(argv)
        self.assertTrue(os.path.isdir(new_out))

    def test_overwrite_replaces_file(self):
        """--overwrite allows replacing an existing output file."""
        # First run
        run_spp_correction.main(self._base_argv())
        expected = os.path.join(self.out_dir, 'spp_corrected_2024-01-01_2024-01-20.nc')
        mtime_first = os.path.getmtime(expected)

        # Second run with --overwrite
        import time
        time.sleep(0.1)
        run_spp_correction.main(self._base_argv() + ['--overwrite'])
        mtime_second = os.path.getmtime(expected)
        self.assertGreater(mtime_second, mtime_first)

    def test_output_is_valid_netcdf(self):
        """The output file is a valid NetCDF that can be opened with xarray."""
        run_spp_correction.main(self._base_argv())
        expected = os.path.join(self.out_dir, 'spp_corrected_2024-01-01_2024-01-20.nc')
        ds_out = xr.open_dataset(expected)
        self.assertGreater(ds_out.time.size, 0)
        ds_out.close()

    def test_params_json_override(self):
        """--params-json overrides corrector params without error."""
        json_path = os.path.join(self.tmpdir, 'params.json')
        self.corrector.save_params(json_path)

        argv = self._base_argv() + ['--params-json', json_path]
        run_spp_correction.main(argv)
        expected = os.path.join(self.out_dir, 'spp_corrected_2024-01-01_2024-01-20.nc')
        self.assertTrue(os.path.exists(expected))


if __name__ == '__main__':
    unittest.main()

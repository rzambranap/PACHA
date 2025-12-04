"""
Unit tests for daily_accumulation module.

This module tests the daily precipitation accumulation pipeline functions.
"""

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from pacha.L2_processing.daily_accumulation import (
    PRODUCT_CONFIG,
    DailyAccumulationPipeline,
    _identify_large_gaps,
    calculate_daily_accumulation,
    check_timestep_completeness,
    gap_fill,
    normalize_dataset,
    save_output,
)


class TestProductConfig(unittest.TestCase):
    """Tests for product configuration."""

    def test_product_config_exists(self):
        """Test that product configuration is properly defined."""
        self.assertIn('imerg', PRODUCT_CONFIG)
        self.assertIn('gsmap', PRODUCT_CONFIG)
        self.assertIn('radar', PRODUCT_CONFIG)

    def test_imerg_config(self):
        """Test IMERG configuration values."""
        config = PRODUCT_CONFIG['imerg']
        self.assertEqual(config['timesteps_per_day'], 48)
        self.assertEqual(config['temporal_resolution_hours'], 0.5)
        self.assertIn('precip_variable', config)

    def test_gsmap_config(self):
        """Test GSMaP configuration values."""
        config = PRODUCT_CONFIG['gsmap']
        self.assertEqual(config['timesteps_per_day'], 24)
        self.assertEqual(config['temporal_resolution_hours'], 1.0)

    def test_radar_config(self):
        """Test radar configuration values."""
        config = PRODUCT_CONFIG['radar']
        self.assertEqual(config['timesteps_per_day'], 144)  # 10-min resolution


class TestNormalizeDataset(unittest.TestCase):
    """Tests for normalize_dataset function."""

    def test_normalize_basic_dataset(self):
        """Test normalization of a basic dataset."""
        # Create test dataset
        times = pd.date_range('2023-01-15', periods=24, freq='h')
        lats = np.linspace(-10, 10, 20)
        lons = np.linspace(-50, -30, 40)

        ds = xr.Dataset(
            {'precipitation': (['time', 'lat', 'lon'],
                               np.random.rand(24, 20, 40))},
            coords={'time': times, 'lat': lats, 'lon': lons}
        )

        result = normalize_dataset(ds, 'gsmap')

        # Check dimensions are in expected order
        self.assertEqual(list(result.dims.keys())[:3], ['time', 'lat', 'lon'])
        # Check data is sorted by time
        time_diff = np.diff(result.time.values)
        self.assertTrue(all(t >= np.timedelta64(0) for t in time_diff))

    def test_normalize_renames_coordinates(self):
        """Test that latitude/longitude are renamed to lat/lon."""
        times = pd.date_range('2023-01-15', periods=6, freq='h')
        lats = np.linspace(-10, 10, 5)
        lons = np.linspace(-50, -30, 10)

        ds = xr.Dataset(
            {'precipitation': (['time', 'latitude', 'longitude'],
                               np.random.rand(6, 5, 10))},
            coords={'time': times, 'latitude': lats, 'longitude': lons}
        )

        result = normalize_dataset(ds, 'gsmap')

        self.assertIn('lat', result.dims)
        self.assertIn('lon', result.dims)


class TestCheckTimestepCompleteness(unittest.TestCase):
    """Tests for check_timestep_completeness function."""

    def test_complete_dataset(self):
        """Test completeness check with all timesteps present."""
        times = pd.date_range('2023-01-15', periods=24, freq='h')
        ds = xr.Dataset(
            {'precipitation': (['time'], np.random.rand(24))},
            coords={'time': times}
        )

        is_complete, report = check_timestep_completeness(
            ds, 'gsmap', '2023-01-15'
        )

        self.assertTrue(is_complete)
        self.assertEqual(report['expected_timesteps'], 24)
        self.assertEqual(report['actual_timesteps'], 24)
        self.assertEqual(report['missing_timesteps'], 0)
        self.assertEqual(report['missing_fraction'], 0.0)

    def test_incomplete_dataset(self):
        """Test completeness check with missing timesteps."""
        # Create dataset with only 12 hours (50% missing)
        times = pd.date_range('2023-01-15', periods=12, freq='h')
        ds = xr.Dataset(
            {'precipitation': (['time'], np.random.rand(12))},
            coords={'time': times}
        )

        is_complete, report = check_timestep_completeness(
            ds, 'gsmap', '2023-01-15', max_missing_fraction=0.2
        )

        self.assertFalse(is_complete)
        self.assertEqual(report['expected_timesteps'], 24)
        self.assertEqual(report['actual_timesteps'], 12)
        self.assertEqual(report['missing_timesteps'], 12)
        self.assertEqual(report['missing_fraction'], 0.5)

    def test_invalid_product_type(self):
        """Test that invalid product type raises ValueError."""
        ds = xr.Dataset({'precipitation': (['time'], [1, 2, 3])})

        with self.assertRaises(ValueError):
            check_timestep_completeness(ds, 'invalid_product', '2023-01-15')


class TestIdentifyLargeGaps(unittest.TestCase):
    """Tests for _identify_large_gaps function."""

    def test_no_gaps(self):
        """Test with no missing data."""
        is_missing = np.array([False, False, False, False, False])
        result = _identify_large_gaps(is_missing, max_gap_size=2)
        self.assertTrue(all(~result))

    def test_small_gap(self):
        """Test with gap smaller than threshold."""
        is_missing = np.array([False, True, False, False, False])
        result = _identify_large_gaps(is_missing, max_gap_size=2)
        self.assertTrue(all(~result))  # Small gaps not flagged

    def test_large_gap(self):
        """Test with gap larger than threshold."""
        is_missing = np.array([False, True, True, True, False])
        result = _identify_large_gaps(is_missing, max_gap_size=2)
        expected = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(result, expected)

    def test_multiple_gaps(self):
        """Test with multiple gaps of different sizes."""
        is_missing = np.array([
            False, True, False,  # Small gap (1)
            False, True, True, True, True, False  # Large gap (4)
        ])
        result = _identify_large_gaps(is_missing, max_gap_size=2)
        expected = np.array([
            False, False, False,  # Small gap not flagged
            False, True, True, True, True, False  # Large gap flagged
        ])
        np.testing.assert_array_equal(result, expected)


class TestGapFill(unittest.TestCase):
    """Tests for gap_fill function."""

    def test_gap_fill_zero(self):
        """Test gap filling with zeros."""
        # Create dataset with 12 hourly timesteps (missing half)
        times = pd.date_range('2023-01-15 00:00', periods=12, freq='2h')
        ds = xr.Dataset(
            {'precipitation': (['time', 'lat', 'lon'],
                               np.ones((12, 5, 5)))},
            coords={
                'time': times,
                'lat': np.arange(5),
                'lon': np.arange(5)
            }
        )

        result = gap_fill(ds, 'gsmap', '2023-01-15', method='zero')

        # Should now have 24 timesteps
        self.assertEqual(len(result.time), 24)
        # Check gap-filling metadata
        self.assertEqual(result.attrs['gap_fill_method'], 'zero')
        self.assertIn('gap_filled_mask', result.coords)

    def test_gap_fill_linear(self):
        """Test gap filling with linear interpolation."""
        # Create dataset with known gaps
        times = pd.date_range('2023-01-15 00:00', periods=12, freq='2h')
        data = np.arange(12)[:, np.newaxis, np.newaxis] * np.ones((1, 3, 3))

        ds = xr.Dataset(
            {'precipitation': (['time', 'lat', 'lon'], data)},
            coords={
                'time': times,
                'lat': np.arange(3),
                'lon': np.arange(3)
            }
        )

        result = gap_fill(ds, 'gsmap', '2023-01-15', method='linear')

        self.assertEqual(len(result.time), 24)
        self.assertEqual(result.attrs['gap_fill_method'], 'linear')

    def test_gap_fill_missing_first_timestep(self):
        """Test gap filling when first timestep is missing (edge case)."""
        # Create dataset missing first hour (starts at 01:00)
        times = pd.date_range('2023-01-15 01:00', periods=23, freq='h')
        data = np.ones((23, 3, 3))

        ds = xr.Dataset(
            {'precipitationCal': (['time', 'lat', 'lon'], data)},
            coords={
                'time': times,
                'lat': np.arange(3),
                'lon': np.arange(3)
            }
        )

        result = gap_fill(ds, 'gsmap', '2023-01-15', method='linear')

        # Should have all 24 timesteps
        self.assertEqual(len(result.time), 24)
        # First timestep should be filled (not NaN)
        first_val = result['precipitationCal'].isel(time=0).values
        self.assertFalse(np.any(np.isnan(first_val)))

    def test_gap_fill_missing_last_timestep(self):
        """Test gap filling when last timestep is missing (edge case)."""
        # Create dataset missing last hour (ends at 22:00)
        times = pd.date_range('2023-01-15 00:00', periods=23, freq='h')
        data = np.ones((23, 3, 3))

        ds = xr.Dataset(
            {'precipitationCal': (['time', 'lat', 'lon'], data)},
            coords={
                'time': times,
                'lat': np.arange(3),
                'lon': np.arange(3)
            }
        )

        result = gap_fill(ds, 'gsmap', '2023-01-15', method='linear')

        # Should have all 24 timesteps
        self.assertEqual(len(result.time), 24)
        # Last timestep should be filled (not NaN)
        last_val = result['precipitationCal'].isel(time=-1).values
        self.assertFalse(np.any(np.isnan(last_val)))

    def test_gap_fill_missing_both_ends(self):
        """Test gap filling when both first and last timesteps are missing."""
        # Create dataset missing both ends
        times = pd.date_range('2023-01-15 01:00', periods=22, freq='h')
        data = np.ones((22, 3, 3))

        ds = xr.Dataset(
            {'precipitationCal': (['time', 'lat', 'lon'], data)},
            coords={
                'time': times,
                'lat': np.arange(3),
                'lon': np.arange(3)
            }
        )

        result = gap_fill(ds, 'gsmap', '2023-01-15', method='linear')

        # Should have all 24 timesteps
        self.assertEqual(len(result.time), 24)
        # Both ends should be filled (not NaN)
        first_val = result['precipitationCal'].isel(time=0).values
        last_val = result['precipitationCal'].isel(time=-1).values
        self.assertFalse(np.any(np.isnan(first_val)))
        self.assertFalse(np.any(np.isnan(last_val)))

    def test_invalid_method(self):
        """Test that invalid interpolation method raises ValueError."""
        times = pd.date_range('2023-01-15', periods=24, freq='h')
        ds = xr.Dataset(
            {'precipitation': (['time'], np.random.rand(24))},
            coords={'time': times}
        )

        with self.assertRaises(ValueError):
            gap_fill(ds, 'gsmap', '2023-01-15', method='invalid_method')


class TestCalculateDailyAccumulation(unittest.TestCase):
    """Tests for calculate_daily_accumulation function."""

    def test_accumulation_calculation(self):
        """Test daily accumulation calculation."""
        # Create dataset with constant rain rate of 1 mm/hr for 24 hours
        times = pd.date_range('2023-01-15', periods=24, freq='h')
        rain_rate = np.ones((24, 5, 5))  # 1 mm/hr

        ds = xr.Dataset(
            {'precipitationCal': (['time', 'lat', 'lon'], rain_rate)},
            coords={
                'time': times,
                'lat': np.arange(5),
                'lon': np.arange(5)
            }
        )

        result = calculate_daily_accumulation(ds, 'gsmap')

        # 1 mm/hr * 1 hr * 24 timesteps = 24 mm/day
        expected = 24.0
        np.testing.assert_allclose(
            result['daily_precipitation'].values,
            np.full((5, 5), expected),
            rtol=1e-5
        )

    def test_accumulation_with_nan(self):
        """Test accumulation handles NaN values."""
        times = pd.date_range('2023-01-15', periods=24, freq='h')
        rain_rate = np.ones((24, 5, 5))
        rain_rate[0:12, :, :] = np.nan  # Half missing

        ds = xr.Dataset(
            {'precipitationCal': (['time', 'lat', 'lon'], rain_rate)},
            coords={
                'time': times,
                'lat': np.arange(5),
                'lon': np.arange(5)
            }
        )

        result = calculate_daily_accumulation(
            ds, 'gsmap', min_valid_fraction=0.5
        )

        # Valid fraction is 0.5, so data should be valid
        self.assertFalse(np.all(np.isnan(result['daily_precipitation'].values)))

    def test_accumulation_metadata(self):
        """Test that output contains proper metadata."""
        times = pd.date_range('2023-01-15', periods=24, freq='h')
        ds = xr.Dataset(
            {'precipitationCal': (['time', 'lat', 'lon'],
                                  np.random.rand(24, 3, 3))},
            coords={
                'time': times,
                'lat': np.arange(3),
                'lon': np.arange(3)
            }
        )

        result = calculate_daily_accumulation(ds, 'gsmap')

        self.assertIn('daily_precipitation', result.data_vars)
        self.assertIn('valid_fraction', result.data_vars)
        self.assertIn('title', result.attrs)
        self.assertIn('source', result.attrs)
        self.assertIn('units', result.attrs)

    def test_invalid_variable(self):
        """Test that missing variable raises ValueError."""
        times = pd.date_range('2023-01-15', periods=24, freq='h')
        ds = xr.Dataset(
            {'wrong_variable': (['time'], np.random.rand(24))},
            coords={'time': times}
        )

        with self.assertRaises(ValueError):
            calculate_daily_accumulation(
                ds, 'gsmap', precip_variable='nonexistent'
            )


class TestSaveOutput(unittest.TestCase):
    """Tests for save_output function."""

    def test_save_output(self):
        """Test saving output to file."""
        import os
        import tempfile

        ds = xr.Dataset(
            {'precipitation': (['lat', 'lon'], np.random.rand(5, 5))},
            coords={'lat': np.arange(5), 'lon': np.arange(5)}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'test_output.nc')
            result = save_output(ds, output_path)

            self.assertEqual(result, output_path)
            self.assertTrue(os.path.exists(output_path))

            # Verify file can be loaded
            loaded = xr.open_dataset(output_path)
            self.assertIn('precipitation', loaded.data_vars)

    def test_save_creates_directory(self):
        """Test that save_output creates parent directory."""
        import os
        import tempfile

        ds = xr.Dataset(
            {'precipitation': (['lat', 'lon'], np.random.rand(3, 3))},
            coords={'lat': np.arange(3), 'lon': np.arange(3)}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'subdir', 'test.nc')
            save_output(ds, output_path)

            self.assertTrue(os.path.exists(output_path))

    def test_no_overwrite(self):
        """Test that overwrite=False raises error for existing file."""
        import os
        import tempfile

        ds = xr.Dataset({'data': (['x'], [1, 2, 3])})

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, 'existing.nc')
            ds.to_netcdf(output_path)

            with self.assertRaises(FileExistsError):
                save_output(ds, output_path, overwrite=False)


class TestDailyAccumulationPipeline(unittest.TestCase):
    """Tests for DailyAccumulationPipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = DailyAccumulationPipeline(
            product_type='imerg',
            input_path='/data/imerg/',
            output_path='/data/output/'
        )

        self.assertEqual(pipeline.product_type, 'imerg')
        self.assertEqual(pipeline.input_path, '/data/imerg/')
        self.assertEqual(pipeline.output_path, '/data/output/')
        self.assertEqual(pipeline.config, PRODUCT_CONFIG['imerg'])

    def test_invalid_product_type(self):
        """Test that invalid product type raises ValueError."""
        with self.assertRaises(ValueError):
            DailyAccumulationPipeline(
                product_type='invalid',
                input_path='/data/'
            )

    def test_get_processing_log(self):
        """Test getting processing log as DataFrame."""
        pipeline = DailyAccumulationPipeline(
            product_type='gsmap',
            input_path='/data/'
        )

        # Add some log entries manually
        pipeline.processing_log.append({
            'date': '2023-01-15',
            'status': 'success',
            'output_file': '/data/out.nc',
            'error': None
        })

        log_df = pipeline.get_processing_log()

        self.assertIsInstance(log_df, pd.DataFrame)
        self.assertEqual(len(log_df), 1)
        self.assertIn('date', log_df.columns)
        self.assertIn('status', log_df.columns)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""

    def test_full_pipeline_simulation(self):
        """Test a simulated full pipeline run."""
        # Create complete test dataset
        times = pd.date_range('2023-01-15', periods=24, freq='h')
        lats = np.linspace(-5, 5, 10)
        lons = np.linspace(-45, -35, 10)

        rain_rate = np.random.exponential(scale=0.5, size=(24, 10, 10))

        ds = xr.Dataset(
            {'precipitationCal': (['time', 'lat', 'lon'], rain_rate)},
            coords={
                'time': times,
                'lat': lats,
                'lon': lons
            }
        )

        # Normalize
        ds_norm = normalize_dataset(ds, 'gsmap')

        # Check completeness
        is_complete, report = check_timestep_completeness(
            ds_norm, 'gsmap', '2023-01-15'
        )
        self.assertTrue(is_complete)

        # Gap fill (should be no-op for complete data)
        ds_filled = gap_fill(ds_norm, 'gsmap', '2023-01-15', method='linear')

        # Calculate daily accumulation
        daily = calculate_daily_accumulation(ds_filled, 'gsmap')

        # Verify output
        self.assertIn('daily_precipitation', daily.data_vars)
        self.assertIn('valid_fraction', daily.data_vars)

        # All valid fractions should be 1.0
        np.testing.assert_allclose(
            daily['valid_fraction'].values,
            np.ones((10, 10)),
            rtol=1e-5
        )


if __name__ == '__main__':
    unittest.main()

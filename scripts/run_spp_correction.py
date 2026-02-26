#!/usr/bin/env python
"""
SPP correction workflow script.

Applies a trained ClusteredCorrector to a set of NetCDF files and writes
corrected output to a NetCDF file.

Example usage::

    python scripts/run_spp_correction.py \\
        --input-folder /path/to/inputs \\
        --corrector-pkl /path/to/corrector.pkl \\
        --start-date 2024-01-01 \\
        --end-date 2024-01-31 \\
        --target-variable imerg_v7 \\
        --output-folder /path/to/outputs
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args(argv=None):
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv : list of str, optional
        Argument list. Defaults to sys.argv[1:].

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            'Apply SPP correction workflow using a trained ClusteredCorrector pickle.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        '--input-folder',
        required=True,
        help='Path to folder containing input NetCDF files.',
    )
    parser.add_argument(
        '--corrector-pkl',
        required=True,
        help='Path to a fully trained corrector pickle (produced via ClusteredCorrector.save()).',
    )
    parser.add_argument(
        '--start-date',
        required=True,
        help='Start date in YYYY-MM-DD format (inclusive).',
    )
    parser.add_argument(
        '--end-date',
        required=True,
        help='End date in YYYY-MM-DD format (inclusive).',
    )
    parser.add_argument(
        '--target-variable',
        required=True,
        help='Name of the variable to correct (e.g. imerg_v7).',
    )
    parser.add_argument(
        '--output-folder',
        required=True,
        help='Path to write corrected output NetCDF file.',
    )

    # Optional arguments
    parser.add_argument(
        '--params-json',
        default=None,
        help=(
            'Optional path to params JSON. If provided, params are loaded via '
            'ClusteredCorrector.load_params() and override corrector.params_by_label.'
        ),
    )
    parser.add_argument(
        '--input-glob',
        default='*.nc',
        help='Glob pattern for matching input files within input-folder.',
    )
    parser.add_argument(
        '--nsubdivisions',
        type=int,
        default=2,
        help='Number of spatial subdivisions along each axis for correction.',
    )
    parser.add_argument(
        '--output-filename',
        default=None,
        help=(
            'Output filename (without folder path). '
            'Defaults to spp_corrected_<start-date>_<end-date>.nc.'
        ),
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite the output file if it already exists.',
    )

    return parser.parse_args(argv)


def main(argv=None):
    """
    Entry point for the SPP correction workflow.

    Parameters
    ----------
    argv : list of str, optional
        Argument list. Defaults to sys.argv[1:].
    """
    import xarray as xr
    from pacha.merging.spp_correction import ClusteredCorrector

    args = parse_args(argv)

    # --- Validate inputs ---
    input_folder = Path(args.input_folder)
    if not input_folder.is_dir():
        print(f'ERROR: --input-folder does not exist or is not a directory: {input_folder}',
              file=sys.stderr)
        sys.exit(1)

    corrector_pkl = Path(args.corrector_pkl)
    if not corrector_pkl.is_file():
        print(f'ERROR: --corrector-pkl does not exist: {corrector_pkl}', file=sys.stderr)
        sys.exit(1)

    if args.params_json is not None:
        params_json = Path(args.params_json)
        if not params_json.is_file():
            print(f'ERROR: --params-json does not exist: {params_json}', file=sys.stderr)
            sys.exit(1)
    else:
        params_json = None

    # --- Discover input files ---
    input_files = sorted(input_folder.glob(args.input_glob))
    if not input_files:
        print(
            f'ERROR: No files matching "{args.input_glob}" found in {input_folder}',
            file=sys.stderr,
        )
        sys.exit(1)
    print(f'Found {len(input_files)} input file(s) matching "{args.input_glob}".')

    # --- Open dataset ---
    # Open individual files and concatenate to avoid requiring dask.
    # Files are explicitly closed after loading to free resources.
    print('Opening input dataset...')
    if len(input_files) == 1:
        ds = xr.open_dataset(str(input_files[0]), mask_and_scale=True).load()
    else:
        datasets = [xr.open_dataset(str(f), mask_and_scale=True) for f in input_files]
        try:
            ds = xr.concat(datasets, dim='time').load()
        finally:
            for _d in datasets:
                _d.close()

    # --- Subset by time range ---
    # xarray supports slicing with string dates when the time coordinate uses
    # numpy datetime64 (the default for NetCDF files without cftime).
    print(f'Subsetting time range: {args.start_date} to {args.end_date}...')
    ds = ds.sel(time=slice(args.start_date, args.end_date))
    if ds.time.size == 0:
        print(
            f'ERROR: No time steps found in the range [{args.start_date}, {args.end_date}].',
            file=sys.stderr,
        )
        sys.exit(1)
    print(f'Dataset has {ds.time.size} time step(s) after subsetting.')

    # --- Validate target variable ---
    if args.target_variable not in ds.data_vars:
        print(
            f'ERROR: --target-variable "{args.target_variable}" not found in dataset. '
            f'Available variables: {list(ds.data_vars)}',
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Load corrector ---
    print(f'Loading corrector from {corrector_pkl}...')
    corrector = ClusteredCorrector.load(str(corrector_pkl))

    # --- Optionally override params ---
    if params_json is not None:
        print(f'Overriding corrector params from {params_json}...')
        loaded_params = ClusteredCorrector.load_params(str(params_json))
        corrector.params_by_label = loaded_params

    # --- Apply correction ---
    print(
        f'Applying correction (target_variable={args.target_variable!r}, '
        f'nsubdivisions={args.nsubdivisions})...'
    )
    corrected_ds = corrector.apply(
        ds,
        target_variable=args.target_variable,
        nsubdivisions=args.nsubdivisions,
    )
    print('Correction complete.')

    # --- Determine output path ---
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.output_filename:
        output_filename = args.output_filename
    else:
        output_filename = f'spp_corrected_{args.start_date}_{args.end_date}.nc'

    output_path = output_folder / output_filename

    if output_path.exists() and not args.overwrite:
        print(
            f'ERROR: Output file already exists: {output_path}. '
            'Use --overwrite to replace it.',
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Write output ---
    print(f'Writing corrected dataset to {output_path}...')
    corrected_ds.to_netcdf(str(output_path))
    print(f'Done. Output written to: {output_path}')


if __name__ == '__main__':
    main()

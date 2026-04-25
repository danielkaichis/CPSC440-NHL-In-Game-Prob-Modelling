import os

import arviz as az
import pandas as pd


DEFAULT_TRAIN_FILE_PATHS = [
    'nhl_raw_data/2023_2024_part1.json',
    'nhl_raw_data/2023_2024_part2.json',
    'nhl_raw_data/2024_2025_part1.json',
    'nhl_raw_data/2024_2025_part2.json',
    'nhl_raw_data/2025_2026_part1.json',
]

DEFAULT_STATE_MAP = {
    0: '3v3',
    1: '4v4',
    2: '5v5',
    3: 'away_PP_1',
    4: 'away_PP_2',
    5: 'away_empty_net',
    6: 'home_PP_1',
    7: 'home_PP_2',
    8: 'home_empty_net',
    9: 'special',
}

def load_and_concat_event_files(file_paths, load_single_file_fn):
    """Load and concatenate per-file event tables."""
    all_dfs = []

    for filepath in file_paths:
        if os.path.exists(filepath):
            print(f"Processing {filepath}...")
            all_dfs.append(load_single_file_fn(filepath))
        else:
            print(f"Warning: Could not find {filepath}. Check your directory.")

    if not all_dfs:
        raise FileNotFoundError("No training files were found. Cannot build event table.")

    return pd.concat(all_dfs, ignore_index=True)


def filter_positive_durations(df):
    """Drop zero-duration rows so event exposure is valid for Poisson likelihoods."""
    return df[df['duration_seconds'] > 0].copy()


def add_state_codes(df, state_map, state_col='manpower_state', output_col='state_code'):
    """Map string manpower states to integer codes with a fallback to 'special'."""
    inv_state_map = {v: k for k, v in state_map.items()}
    default_code = inv_state_map.get('special', 9)

    df = df.copy()
    df[output_col] = df[state_col].map(inv_state_map).fillna(default_code).astype(int)
    return df


def save_trace_to_netcdf(trace, output_path, success_msg=None):
    """Persist posterior trace to disk and print a success message."""
    az.to_netcdf(trace, output_path)
    if success_msg:
        print(success_msg)

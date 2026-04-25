import json
import pandas as pd
import numpy as np
import arviz as az

from .team_load_data import load_team_data
from .team_advi import run_team_advi
from .team_monte_carlo import TeamMonteCarlo
from utils.persistence_utils import save_20_day_window_results
from utils.penalty_utils import estimate_home_penalty_share
from utils.training_utils import DEFAULT_STATE_MAP, add_state_codes, filter_positive_durations
from utils.evaluation_utils import (
    DEFAULT_CHECKPOINTS,
    evaluate_checkpoints,
)

def _read_game_ids(json_files):
    game_ids = set()
    for file_path in json_files:
        with open(file_path, 'r') as f:
            for game in json.load(f):
                game_ids.add(game.get('id'))
    return game_ids

def _posterior_to_team_priors(trace, min_sd=1e-3):
    off = trace.posterior['off_stars']
    deff = trace.posterior['def_stars']

    # Feed posterior means/SDs forward as priors for the next sliding window.
    return {
        'off_mu': off.mean(dim=("chain", "draw")).values,
        'off_sd': np.clip(off.std(dim=("chain", "draw")).values, min_sd, None),
        'def_mu': deff.mean(dim=("chain", "draw")).values,
        'def_sd': np.clip(deff.std(dim=("chain", "draw")).values, min_sd, None),
    }

def _trace_path(i):
    return f"advi_traces/sliding_window_trace_{i:03d}.nc"

def fit_sliding_windows(
    train_files,
    test_files,
    state_map,
    update_days=20,
    pretrain_iter=40000,
    update_iter=10000,
):
    """
    Fit one ADVI trace per test window.

    Returns window_meta: list of (i, window_start, window_end, test_window_df)
    with each trace saved to advi_traces/sliding_window_trace_NNN.nc.
    """
    all_files = list(dict.fromkeys(train_files + test_files))

    df_all, team_map = load_team_data(all_files)
    df_all = add_state_codes(filter_positive_durations(df_all), state_map)
    df_all['game_date'] = pd.to_datetime(df_all['game_date'])
    df_all = df_all.sort_values('game_date')

    train_ids = _read_game_ids(train_files)
    test_ids = _read_game_ids(test_files)
    df_hist = df_all[df_all['game_id'].isin(train_ids)].copy()
    df_test = df_all[df_all['game_id'].isin(test_ids)].copy()

    if df_hist.empty or df_test.empty:
        raise ValueError("Historical or test split is empty. Check input file lists.")

    print("--- Phase 1: Pre-training on historical data ---")
    print(f"  {len(df_hist)} events across {df_hist['game_id'].nunique()} games")
    hist_trace, _ = run_team_advi(df_hist, state_map, len(team_map), n_iter=pretrain_iter)
    current_priors = _posterior_to_team_priors(hist_trace)

    date_range = pd.date_range(
        start=df_test['game_date'].min(),
        end=df_test['game_date'].max(),
        freq=f'{update_days}D',
    )
    print(f"\n--- Phase 2: Fitting {len(date_range)} windows of {update_days} days ---")

    window_meta = []
    current_season_pool = df_test.iloc[0:0].copy()

    for i, window_start in enumerate(date_range):
        window_end = window_start + pd.Timedelta(days=update_days)
        test_window = df_test[
            (df_test['game_date'] >= window_start) &
            (df_test['game_date'] < window_end)
        ]
        if test_window.empty:
            continue

        # Pool grows monotonically so each new window updates from all prior in-season evidence.
        current_season_pool = pd.concat(
            [current_season_pool, test_window], ignore_index=True
        )
        print(
            f"\n  Window {i}: {window_start.date()} → {window_end.date()} | "
            f"{test_window['game_id'].nunique()} games | "
            f"pool={current_season_pool['game_id'].nunique()} games total"
        )

        trace, _ = run_team_advi(
            current_season_pool, state_map, len(team_map),
            n_iter=update_iter, priors=current_priors,
        )
        current_priors = _posterior_to_team_priors(trace)

        path = _trace_path(i)
        az.to_netcdf(trace, path)
        print(f"Saved trace: {path}")

        window_meta.append((i, window_start, window_end, test_window))

    return window_meta

def evaluate_sliding_windows(
    window_meta,
    state_map,
    penalty_share,
    checkpoints=None,
):
    checkpoints = checkpoints or DEFAULT_CHECKPOINTS
    results = []

    print("--- Phase 3: Monte Carlo evaluation ---")
    for i, window_start, window_end, test_window in window_meta:
        path = _trace_path(i)
        print(f"\n  Window {i}: {window_start.date()} → {window_end.date()} | loading {path}")

        mc = TeamMonteCarlo(
            path, state_map, "team_mapping.json",
            home_penalty_share=penalty_share,
        )

        def predict(sit, t_rem):
            res = mc.simulate(
                h_id=int(sit['home_team_id']),
                a_id=int(sit['away_team_id']),
                h_score=int(sit['home_score']),
                a_score=int(sit['away_score']),
                t_rem=int(t_rem),
                state_name=sit['manpower_state'],
            )
            # Keep probabilities away from exact 0/1 so log-loss remains finite.
            return np.clip(res['Home Win %'] / 100.0, 1e-6, 1 - 1e-6)

        scores = evaluate_checkpoints(
            test_window,
            checkpoints,
            predict,
            progress_msg=lambda t_rem, n: f"{t_rem // 60:3d} min left | {n} games",
        )

        for t_rem, metrics in scores.items():
            results.append({
                'date': window_start,
                'checkpoint_sec': int(t_rem),
                'mins_left': int(t_rem) // 60,
                **metrics,
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    state_map = DEFAULT_STATE_MAP

    train = [
        'nhl_raw_data/2023_2024_part1.json',
        'nhl_raw_data/2023_2024_part2.json',
        'nhl_raw_data/2024_2025_part1.json',
        'nhl_raw_data/2024_2025_part2.json',
    ]
    test = ['nhl_raw_data/2025_2026_part1.json', 'nhl_raw_data/2025_2026_part2.json']

    penalty_share = estimate_home_penalty_share(train)
    print(f"Home penalty share: {penalty_share:.3f}")

    window_meta = fit_sliding_windows(train, test, state_map)
    stats_df = evaluate_sliding_windows(window_meta, state_map, penalty_share)

    save_path = save_20_day_window_results(stats_df, model_name="team")
    print(f"\nSaved results: {save_path}")
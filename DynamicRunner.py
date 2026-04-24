import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss
import arviz as az

from HeirarchicalLoadData import load_h_data
from HeirarchicalADVI import run_h_advi
from HeirarchicalMonteCarlo import HeirarchicalMonteCarlo
from results_store import save_sliding_results
from penalty_utils import estimate_home_penalty_share
from training_pipeline import DEFAULT_STATE_MAP, add_state_codes, filter_positive_durations

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

    return {
        'off_mu': off.mean(dim=("chain", "draw")).values,
        'off_sd': np.clip(off.std(dim=("chain", "draw")).values, min_sd, None),
        'def_mu': deff.mean(dim=("chain", "draw")).values,
        'def_sd': np.clip(deff.std(dim=("chain", "draw")).values, min_sd, None),
    }


def run_optimized_sliding_window(train_files, test_files, state_map, update_days=20, pretrain_iter=40000, update_iter=10000):
    all_files = list(dict.fromkeys(train_files + test_files))
    penalty_share = estimate_home_penalty_share(train_files)
    print(f"Using data-driven home penalty share: {penalty_share:.3f}")

    df_all, team_map = load_h_data(all_files)
    df_all = add_state_codes(filter_positive_durations(df_all), state_map)
    df_all['game_date'] = pd.to_datetime(df_all['game_date'])
    df_all = df_all.sort_values('game_date')

    train_ids = _read_game_ids(train_files)
    test_ids = _read_game_ids(test_files)
    df_hist = df_all[df_all['game_id'].isin(train_ids)].copy()
    df_test = df_all[df_all['game_id'].isin(test_ids)].copy()

    if df_hist.empty or df_test.empty:
        raise ValueError("Historical or test split is empty. Check input file lists.")

    print("--- Phase 1: Establishing Historical Baseline ---")
    print(
        f"Pre-training on {len(df_hist)} historical events from "
        f"{df_hist['game_id'].nunique()} games..."
    )
    hist_trace, _ = run_h_advi(df_hist, state_map, len(team_map), n_iter=pretrain_iter)
    current_priors = _posterior_to_team_priors(hist_trace)
    az.to_netcdf(hist_trace, "current_sliding_trace.nc")
    mc = HeirarchicalMonteCarlo(
        "current_sliding_trace.nc",
        state_map,
        "h_team_mapping.json",
        home_penalty_share=penalty_share,
    )

    print("--- Phase 2: Sequential In-Season Updating ---")
    date_range = pd.date_range(
        start=df_test['game_date'].min(),
        end=df_test['game_date'].max(),
        freq=f'{update_days}D',
    )
    print(f"Running {len(date_range)} update windows of {update_days} days...")

    results = []
    current_season_pool = df_test.iloc[0:0].copy()

    for i, window_start in enumerate(date_range):
        window_end = window_start + pd.Timedelta(days=update_days)
        test_window = df_test[
            (df_test['game_date'] >= window_start) &
            (df_test['game_date'] < window_end)
        ]

        if test_window.empty:
            continue

        print(f"\n--- Window {i+1}: {window_start.date()} to {window_end.date()} ---")
        print(f"Predicting {test_window['game_id'].nunique()} games...")

        y_true, y_prob = [], []
        for g_id in test_window['game_id'].unique():
            g_data = test_window[test_window['game_id'] == g_id]
            idx = (g_data['time_remaining'] - 1800).abs().idxmin()
            sit = g_data.loc[idx]

            res = mc.simulate(
                h_id=int(sit['home_team_id']),
                a_id=int(sit['away_team_id']),
                h_score=int(sit['home_score']),
                a_score=int(sit['away_score']),
                t_rem=1800,
                state_name=sit['manpower_state'],
            )

            y_prob.append(np.clip(res['Home Win %'] / 100.0, 1e-6, 1 - 1e-6))
            actual = 1 if g_data.iloc[-1]['home_score'] > g_data.iloc[-1]['away_score'] else 0
            y_true.append(actual)

        bs = brier_score_loss(y_true, y_prob)
        ll = log_loss(y_true, y_prob, labels=[0, 1])
        pred_labels = [1 if p >= 0.5 else 0 for p in y_prob]
        acc = float(np.mean(np.array(pred_labels) == np.array(y_true)))
        results.append({'date': window_start, 'BS': bs, 'LL': ll, 'ACC': acc})

        current_season_pool = pd.concat([current_season_pool, test_window], ignore_index=True)
        print(
            f"Updating on {len(current_season_pool)} in-season events from "
            f"{current_season_pool['game_id'].nunique()} games..."
        )

        update_trace, _ = run_h_advi(
            current_season_pool,
            state_map,
            len(team_map),
            n_iter=update_iter,
            priors=current_priors,
        )
        current_priors = _posterior_to_team_priors(update_trace)
        az.to_netcdf(update_trace, "current_sliding_trace.nc")
        mc = HeirarchicalMonteCarlo(
            "current_sliding_trace.nc",
            state_map,
            "h_team_mapping.json",
            home_penalty_share=penalty_share,
        )

    return pd.DataFrame(results)

def plot_sliding_results(df_results):
    if df_results.empty:
        print("No evaluation windows produced predictions; skipping plot.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:green'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Brier Score (Lower is Better)', color=color)
    ax1.plot(df_results['date'], df_results['BS'], marker='o', color=color, label='Brier Score')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Log-Loss', color=color)
    ax2.plot(df_results['date'], df_results['LL'], marker='s', linestyle='--', color=color, label='Log-Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Dynamic Model Performance: Sliding Window Evaluation')
    fig.tight_layout()
    plt.grid(alpha=0.3)
    plt.show()

    if 'ACC' in df_results.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df_results['date'], df_results['ACC'], marker='^', color='tab:purple', label='Accuracy')
        plt.ylim(0, 1)
        plt.title('Dynamic Model Accuracy: Sliding Window Evaluation')
        plt.xlabel('Date')
        plt.ylabel('Accuracy')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    state_map = DEFAULT_STATE_MAP
    
    train = [
        'nhl_raw_data/2023_2024_part1.json',
        'nhl_raw_data/2023_2024_part2.json',
        'nhl_raw_data/2024_2025_part1.json',
        'nhl_raw_data/2024_2025_part2.json',
    ]
    test = ['nhl_raw_data/2025_2026_part1.json', 'nhl_raw_data/2025_2026_part2.json']
    
    stats_df = run_optimized_sliding_window(train, test, state_map)
    save_path = save_sliding_results(stats_df, model_name="hierarchical_sliding")
    print(f"Saved sliding-window results to {save_path}")
    plot_sliding_results(stats_df)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss

from MonteCarlo import NHLMonteCarlo as BaselineMC
from HeirarchicalMonteCarlo import HeirarchicalMonteCarlo as HierarchicalMC
from HeirarchicalLoadData import load_h_data
from results_store import save_dataframe
from training_pipeline import DEFAULT_TRAIN_FILE_PATHS, DEFAULT_STATE_MAP
from penalty_utils import estimate_home_penalty_share

def run_comparison():
    state_map = DEFAULT_STATE_MAP
    penalty_share = estimate_home_penalty_share(DEFAULT_TRAIN_FILE_PATHS)
    print(f"Using data-driven home penalty share: {penalty_share:.3f}")
    
    print("Loading Baseline Engine...")
    base_mc = BaselineMC("nhl_advi_trace.nc", state_map, home_penalty_share=penalty_share)
    
    print("Loading Hierarchical Engine...")
    h_mc = HierarchicalMC(
        "h_nhl_trace.nc",
        state_map,
        "h_team_mapping.json",
        home_penalty_share=penalty_share,
    )
    
    test_df, _ = load_h_data(['nhl_raw_data/2025_2026_part2.json'])
    
    checkpoints = [3000, 2400, 1800, 1200, 600, 120]
    game_ids = test_df['game_id'].unique()
    
    results = []

    for t_rem in checkpoints:
        print(f"Comparing at {t_rem//60} minutes remaining...")
        y_true = []
        y_pred_base = []
        y_pred_hier = []

        for g_id in game_ids:
            game_data = test_df[test_df['game_id'] == g_id]
            idx = (game_data['time_remaining'] - t_rem).abs().idxmin()
            sit = game_data.loc[idx]
            
            final_row = game_data.iloc[-1]
            home_won = 1 if final_row['home_score'] > final_row['away_score'] else 0
            y_true.append(home_won)
            
            res_b = base_mc.simulate_live_game(
                current_home_score=int(sit['home_score']),
                current_away_score=int(sit['away_score']),
                time_remaining_sec=t_rem,
                current_state_name=sit['manpower_state']
            )
            y_pred_base.append(res_b['Home Win %'] / 100.0)
            
            res_h = h_mc.simulate(
                h_id=int(sit['home_team_id']), 
                a_id=int(sit['away_team_id']),
                h_score=int(sit['home_score']), 
                a_score=int(sit['away_score']),
                t_rem=t_rem, 
                state_name=sit['manpower_state']
            )
            y_pred_hier.append(res_h['Home Win %'] / 100.0)

        bs_base = brier_score_loss(y_true, y_pred_base)
        bs_hier = brier_score_loss(y_true, y_pred_hier)

        base_labels = [1 if p >= 0.5 else 0 for p in y_pred_base]
        hier_labels = [1 if p >= 0.5 else 0 for p in y_pred_hier]
        acc_base = float(np.mean(np.array(base_labels) == np.array(y_true)))
        acc_hier = float(np.mean(np.array(hier_labels) == np.array(y_true)))
        
        ll_base = log_loss(y_true, y_pred_base, labels=[0, 1])
        ll_hier = log_loss(y_true, y_pred_hier, labels=[0, 1])
        
        results.append({
            'Mins Left': t_rem // 60,
            'Baseline BS': bs_base,
            'Hierarchical BS': bs_hier,
            'Baseline ACC': acc_base,
            'Hierarchical ACC': acc_hier,
            'Baseline LL': ll_base,
            'Hierarchical LL': ll_hier,
            'LL Improvement %': ((ll_base - ll_hier) / ll_base) * 100
        })

    df_res = pd.DataFrame(results)
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)

    save_path = save_dataframe(df_res, prefix="comparison")
    print(f"Saved comparison results to {save_path}")
    print(df_res.to_string(index=False))
    print("="*50)

    plt.figure(figsize=(10, 6))
    plt.plot(df_res['Mins Left'], df_res['Baseline BS'], marker='o', label='Baseline (League Avg)', color='blue')
    plt.plot(df_res['Mins Left'], df_res['Hierarchical BS'], marker='s', label='Hierarchical (Team Talent)', color='green')
    plt.axhline(0.25, color='red', linestyle='--', alpha=0.5, label='Naive Chance')
    
    plt.gca().invert_xaxis()
    plt.title("Brier Score: Baseline vs. Hierarchical Model", fontsize=14)
    plt.xlabel("Minutes Remaining in Game", fontsize=12)
    plt.ylabel("Brier Score (Lower is Better)", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_comparison()
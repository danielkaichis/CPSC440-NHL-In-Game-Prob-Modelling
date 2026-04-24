import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from HeirarchicalLoadData import load_h_data
from HeirarchicalMonteCarlo import HeirarchicalMonteCarlo
from results_store import save_checkpoint_scores
from training_pipeline import DEFAULT_TRAIN_FILE_PATHS, DEFAULT_STATE_MAP
from penalty_utils import estimate_home_penalty_share

def evaluate_h_model(test_df, mc_engine, checkpoints):
    results = {}
    game_ids = test_df['game_id'].unique()
    
    for t_rem in checkpoints:
        print(f"Evaluating {len(game_ids)} games at {t_rem//60} mins left...")
        preds, actuals = [], []

        for g_id in game_ids:
            game_data = test_df[test_df['game_id'] == g_id]
            idx = (game_data['time_remaining'] - t_rem).abs().idxmin()
            sit = game_data.loc[idx]
            
            final_row = game_data.iloc[-1]
            home_won = 1 if final_row['home_score'] > final_row['away_score'] else 0
            
            res = mc_engine.simulate(
                h_id=int(sit['home_team_id']), 
                a_id=int(sit['away_team_id']),
                h_score=int(sit['home_score']), 
                a_score=int(sit['away_score']),
                t_rem=t_rem, 
                state_name=sit['manpower_state']
            )
            
            preds.append(res['Home Win %'] / 100.0)
            actuals.append(home_won)

        pred_labels = [1 if p >= 0.5 else 0 for p in preds]
        acc = float(np.mean(np.array(pred_labels) == np.array(actuals)))
        results[t_rem] = {
            'BS': brier_score_loss(actuals, preds),
            'ACC': acc,
        }
    
    return results

if __name__ == "__main__":
    state_map = DEFAULT_STATE_MAP
    
    penalty_share = estimate_home_penalty_share(DEFAULT_TRAIN_FILE_PATHS)
    print(f"Using data-driven home penalty share: {penalty_share:.3f}")
    mc = HeirarchicalMonteCarlo(
        "h_nhl_trace.nc",
        state_map,
        "h_team_mapping.json",
        home_penalty_share=penalty_share,
    )
    
    test_df, _ = load_h_data(['nhl_raw_data/2025_2026_part2.json'])
    
    checkpoints = [3000, 2400, 1800, 1200, 600, 120]
    h_scores = evaluate_h_model(test_df, mc, checkpoints)
    save_path = save_checkpoint_scores(h_scores, model_name="hierarchical")
    print(f"Saved hierarchical checkpoint results to {save_path}")
    
    x = [(3600 - t)/60 for t in checkpoints]
    y_bs = [h_scores[t]['BS'] for t in checkpoints]
    y_acc = [h_scores[t]['ACC'] for t in checkpoints]
    
    plt.figure(figsize=(8,5))
    plt.plot(x, y_bs, marker='o', label='Hierarchical Team Model (BS)', color='green')
    plt.axhline(0.25, color='red', linestyle='--', label='Random Chance')
    plt.title("Hierarchical Model Performance")
    plt.ylabel("Brier Score")
    plt.xlabel("Minutes Elapsed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_acc, marker='s', color='teal', label='Hierarchical Team Model (ACC)')
    plt.ylim(0, 1)
    plt.title("Hierarchical Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Minutes Elapsed")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
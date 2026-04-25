import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss
from LoadGameData import load_game_data
from MonteCarlo import NHLMonteCarlo
from results_store import save_checkpoint_scores
from training_pipeline import DEFAULT_TRAIN_FILE_PATHS, DEFAULT_STATE_MAP
from penalty_utils import estimate_home_penalty_share

def evaluate_performance(test_df, mc_engine, checkpoints):
    """
    Evaluates the model's predictive performance at various time checkpoints.
    Returns a dictionary of Brier Scores for each checkpoint.
    """
    model_scores = {}
    game_ids = test_df['game_id'].unique()
    
    for t_rem in checkpoints:
        print(f"Analyzing games with {t_rem//60} minutes left...")
        preds, actuals = [], []

        for g_id in game_ids:
            game_data = test_df[test_df['game_id'] == g_id]
            
            idx = (game_data['time_remaining'] - t_rem).abs().idxmin()
            sit = game_data.loc[idx]
            
            final_row = game_data.iloc[-1]
            home_won = 1 if final_row['home_score'] > final_row['away_score'] else 0
            
            res = mc_engine.simulate_live_game(
                current_home_score=int(sit['home_score']),
                current_away_score=int(sit['away_score']),
                time_remaining_sec=t_rem,
                current_state_name=sit['manpower_state'],
                n_simulations=2000 
            )
            
            preds.append(res['Home Win %'] / 100.0)
            actuals.append(home_won)

        pred_labels = [1 if p >= 0.5 else 0 for p in preds]
        acc = float(np.mean(np.array(pred_labels) == np.array(actuals)))
        model_scores[t_rem] = {
            'BS': brier_score_loss(actuals, preds),
            'LL': log_loss(actuals, np.clip(preds, 1e-6, 1 - 1e-6), labels=[0, 1]),
            'ACC': acc,
        }
    
    return model_scores

if __name__ == "__main__":
    mapping = DEFAULT_STATE_MAP
    
    penalty_share = estimate_home_penalty_share(DEFAULT_TRAIN_FILE_PATHS)
    print(f"Using data-driven home penalty share: {penalty_share:.3f}")
    mc = NHLMonteCarlo("nhl_advi_trace.nc", mapping, home_penalty_share=penalty_share)
    test_data = load_game_data('nhl_raw_data/2025_2026_part2.json')

    checkpoints = [3000, 2400, 1800, 1200, 600, 120] # Every 10 mins down to 2 mins
    scores = evaluate_performance(test_data, mc, checkpoints)
    save_path = save_checkpoint_scores(scores, model_name="baseline")
    print(f"Saved baseline checkpoint results to {save_path}")

    x = [(3600 - t)/60 for t in checkpoints]
    y_bs = [scores[t]['BS'] for t in checkpoints]
    y_acc = [scores[t]['ACC'] for t in checkpoints]
    
    plt.plot(x, y_bs, marker='o', label='Bayesian MC (BS)')
    plt.axhline(0.25, color='red', linestyle='--', label='Random Chance')
    plt.ylabel('Brier Score (Lower is Better)')
    plt.xlabel('Minutes Elapsed')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(x, y_acc, marker='s', color='purple', label='Bayesian MC (ACC)')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Minutes Elapsed')
    plt.title('Checkpoint Accuracy')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()
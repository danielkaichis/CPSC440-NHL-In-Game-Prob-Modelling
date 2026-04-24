import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from LoadGameData import load_game_data
from MonteCarlo import NHLMonteCarlo

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
            
            # Find the row closest to the time checkpoint
            idx = (game_data['time_remaining'] - t_rem).abs().idxmin()
            sit = game_data.loc[idx]
            
            # Get the ground truth (Did Home win in regulation?)
            final_row = game_data.iloc[-1]
            home_won = 1 if final_row['home_score'] > final_row['away_score'] else 0
            
            # Simulate the outcome
            res = mc_engine.simulate_live_game(
                current_home_score=int(sit['home_score']),
                current_away_score=int(sit['away_score']),
                time_remaining_sec=t_rem,
                current_state_name=sit['manpower_state'],
                n_simulations=2000 
            )
            
            preds.append(res['Home Win %'] / 100.0)
            actuals.append(home_won)

        model_scores[t_rem] = brier_score_loss(actuals, preds)
    
    return model_scores

if __name__ == "__main__":
    mapping = {0: '3v3', 1: '4v4', 2: '5v5', 3: 'away_PP_1', 4: 'away_PP_2', 
               5: 'away_empty_net', 6: 'home_PP_1', 7: 'home_PP_2', 
               8: 'home_empty_net', 9: 'special'}
    
    mc = NHLMonteCarlo("nhl_advi_trace.nc", mapping)
    test_data = load_game_data('nhl_raw_data/2025_2026_part2.json')

    # Run Evaluation
    checkpoints = [3000, 2400, 1800, 1200, 600, 120] # Every 10 mins down to 2 mins
    scores = evaluate_performance(test_data, mc, checkpoints)

    # Plot
    x = [(3600 - t)/60 for t in checkpoints]
    y = [scores[t] for t in checkpoints]
    
    plt.plot(x, y, marker='o', label='Bayesian MC')
    plt.axhline(0.25, color='red', linestyle='--', label='Random Chance')
    plt.ylabel('Brier Score (Lower is Better)')
    plt.xlabel('Minutes Elapsed')
    plt.legend()
    plt.show()
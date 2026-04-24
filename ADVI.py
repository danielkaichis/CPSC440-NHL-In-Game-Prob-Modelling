import os
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import numpy as np

from LoadGameData import load_game_data
from EmpBayes import calc_priors

def run_baseline_advi(df, priors, state_mapping):
    """
    The PyMC Inference block. 
    """
    state_idx = df['state_code'].values
    durations = df['duration_seconds'].values
    home_goals_obs = df['is_home_goal'].values
    away_goals_obs = df['is_away_goal'].values
    penalties_obs = df['is_penalty'].values
    
    alpha_h = priors['alpha_home'].values
    beta_h = priors['beta_home'].values
    alpha_a = priors['alpha_away'].values
    beta_a = priors['beta_away'].values

    alpha_p = priors['alpha_pen'].iloc[0] 
    beta_p = priors['beta_pen'].iloc[0]
    
    print("Building PyMC Model...")
    with pm.Model() as nhl_model:
        lambda_home = pm.Gamma('lambda_home', alpha=alpha_h, beta=beta_h, shape=len(state_mapping))
        lambda_away = pm.Gamma('lambda_away', alpha=alpha_a, beta=beta_a, shape=len(state_mapping))
        lambda_pen = pm.Gamma('lambda_pen', alpha=alpha_p, beta=beta_p)
        
        mu_home = lambda_home[state_idx] * durations
        mu_away = lambda_away[state_idx] * durations
        mu_pen = lambda_pen * durations
        
        home_scoring = pm.Poisson('home_scoring', mu=mu_home, observed=home_goals_obs)
        away_scoring = pm.Poisson('away_scoring', mu=mu_away, observed=away_goals_obs)
        pen_calling = pm.Poisson('pen_calling', mu=mu_pen, observed=penalties_obs)
        
        print("Running ADVI Inference...")
        mean_field = pm.fit(method='advi', n=30000, obj_optimizer=pm.adam(learning_rate=0.01))
        trace = mean_field.sample(1000)
        
    return trace, nhl_model


def plot_model_results(trace, state_mapping):
    """
    Generates the two most important plots for your Bayesian analysis.
    """
    print("Generating plots...")
    
    # The Trace Plot (Checking ADVI Health)
    # proves model successfully found the posterior distributions without going off to infinity.
    az.plot_trace(trace, var_names=['lambda_home', 'lambda_away'])
    plt.suptitle("Trace Plot: Scoring Rate Posteriors", fontsize=16)
    plt.tight_layout()
    plt.show()

    # plot for manpower states
    # Extract the samples for the home team
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # ArviZ takes the trace directly and finds 'lambda_home' on its own
    az.plot_forest(trace, var_names=['lambda_home'], combined=True, ax=ax)
    
    labels = [state_mapping[i] for i in range(len(state_mapping))]
    ax.set_yticklabels(labels[::-1]) 
    ax.set_title(r"94% Credible Intervals: Home Scoring Rates ($\lambda$) by Manpower State")
    ax.set_xlabel("Expected Goals per Second")
    
    plt.tight_layout()
    fig.savefig('manpower_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    file_paths = [
        'nhl_raw_data/2023_2024_part1.json',
        'nhl_raw_data/2023_2024_part2.json',
        'nhl_raw_data/2024_2025_part1.json',
        'nhl_raw_data/2024_2025_part2.json',
        'nhl_raw_data/2025_2026_part1.json',
    ]
    
    print("Flattening JSON files via LoadGameData module...")
    all_dfs = []
    
    for filepath in file_paths:
        if os.path.exists(filepath):
            print(f"Processing {filepath}...")
            df_part = load_game_data(filepath) 
            all_dfs.append(df_part)
        else:
            print(f"Warning: Could not find {filepath}. Check your directory.")
            
    # Combine the four files
    df_events = pd.concat(all_dfs, ignore_index=True)
    print(f"\nMaster Event Table created with {len(df_events)} total events.")
    
    # Clean the combined data
    df_events = df_events[df_events['duration_seconds'] > 0].copy()
    
    print("Calculating Empirical Priors...")
    df_ready, priors_df, mapping = calc_priors(df_events, prior_weight_games=20)
    
    # Run Inference
    posterior_trace, model = run_baseline_advi(df_ready, priors_df, mapping)
    
    print("\nInference Complete. State Mapping:")
    print(mapping)

    az.to_netcdf(posterior_trace, "nhl_advi_trace.nc")
    print("Saved successfully to nhl_advi_trace.nc")

    plot_model_results(posterior_trace, mapping)
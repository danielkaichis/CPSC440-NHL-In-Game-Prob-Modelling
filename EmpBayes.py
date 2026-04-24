import json
import pandas as pd
import numpy as np
import pymc as pm


def calc_priors(df, prior_weight_games=20):
    # Map string states to integer codes for PyMC vectorization
    df['manpower_state'] = df['manpower_state'].astype('category')
    df['state_code'] = df['manpower_state'].cat.codes
    state_mapping = dict(enumerate(df['manpower_state'].cat.categories))
    
    # Calculate historical totals
    priors = df.groupby('state_code').agg(
        total_duration=('duration_seconds', 'sum'),
        total_home_goals=('is_home_goal', 'sum'),
        total_away_goals=('is_away_goal', 'sum')
    ).reset_index()
    
    # Scale priors
    scale_factor = (prior_weight_games * 3600) / priors['total_duration'].sum()

    total_league_penalties = df['is_penalty'].sum()
    total_league_time = df['duration_seconds'].sum()
    
    # Add small epsilon to prevent alpha=0 or beta=0 which breaks Gamma distribution
    eps = 1e-5
    priors['alpha_home'] = (priors['total_home_goals'] * scale_factor) + eps
    priors['beta_home'] = (priors['total_duration'] * scale_factor) + eps
    
    priors['alpha_away'] = (priors['total_away_goals'] * scale_factor) + eps
    priors['beta_away'] = (priors['total_duration'] * scale_factor) + eps

    priors['alpha_pen'] = (total_league_penalties * scale_factor) + eps
    priors['beta_pen'] = (total_league_time * scale_factor) + eps
    
    return df, priors, state_mapping
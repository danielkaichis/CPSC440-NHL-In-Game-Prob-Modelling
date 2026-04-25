import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

from .team_load_data import load_team_data
from utils.advi_utils import run_advi_inference
from utils.emp_bayes import calc_team_priors
from utils.training_utils import (
    DEFAULT_TRAIN_FILE_PATHS,
    DEFAULT_STATE_MAP,
    add_state_codes,
    filter_positive_durations,
    save_trace_to_netcdf,
)

def run_team_advi(df, state_mapping, n_teams, n_iter=None, priors=None, prior_weight_games=20):
    """
    The Mathematical Core: 
    Uses latent variables (off_stars, def_stars) to model team talent.

    format of priors (optional):
    {
        'off_mu': array-like shape (n_teams,),
        'off_sd': array-like shape (n_teams,),
        'def_mu': array-like shape (n_teams,),
        'def_sd': array-like shape (n_teams,),
    }
    """
    h_idx = df['h_team_code'].to_numpy(dtype='int64')
    a_idx = df['a_team_code'].to_numpy(dtype='int64')
    s_idx = df['state_code'].to_numpy(dtype='int64')
    dur = df['duration_seconds'].to_numpy(dtype='float64')

    # Warm-start team latents from empirical-Bayes estimates if priors not provided.
    if priors is None:
        priors = calc_team_priors(df, n_teams=n_teams, prior_weight_games=prior_weight_games)

    if n_iter is None:
        n_iter = 40000 if priors is None else 10000

    use_custom_priors = priors is not None
    if use_custom_priors:
        # Prior means/SDs are calibrated from empirical Bayes team-level posteriors.
        off_mu = np.asarray(priors['off_mu'])
        off_sd = np.clip(np.asarray(priors['off_sd']), 1e-3, None)
        def_mu = np.asarray(priors['def_mu'])
        def_sd = np.clip(np.asarray(priors['def_sd']), 1e-3, None)

        if len(off_mu) != n_teams or len(off_sd) != n_teams or len(def_mu) != n_teams or len(def_sd) != n_teams:
            raise ValueError("Prior vectors must all have length n_teams.")
    
    with pm.Model() as h_model:
        if use_custom_priors:
            off_stars = pm.Normal('off_stars', mu=off_mu, sigma=off_sd, shape=n_teams)
            def_stars = pm.Normal('def_stars', mu=def_mu, sigma=def_sd, shape=n_teams)
        else:
            # Shared hyperpriors shrink team effects toward league-average skill.
            sigma_off = pm.Exponential('sigma_off', 1.0)
            sigma_def = pm.Exponential('sigma_def', 1.0)
            off_stars = pm.Normal('off_stars', mu=0, sigma=sigma_off, shape=n_teams)
            def_stars = pm.Normal('def_stars', mu=0, sigma=sigma_def, shape=n_teams)
        
        # Home-ice advantage is modeled as a global additive boost to home team log-rates.
        home_ice = pm.Normal('home_ice', mu=0, sigma=0.1)
        # State intercepts absorb baseline scoring pace differences by manpower state.
        state_ints = pm.Normal('state_ints', mu=-7, sigma=1.5, shape=len(state_mapping))
        
        lambda_pen = pm.Gamma('lambda_pen', alpha=1, beta=1)

        # Team offense/defense effects combine with state intercepts on log-rate scale.
        log_h = state_ints[s_idx] + home_ice + off_stars[h_idx] - def_stars[a_idx]
        log_a = state_ints[s_idx] + off_stars[a_idx] - def_stars[h_idx]
        
        # Exponentiate linear predictors to recover nonnegative Poisson rates.
        pm.Poisson('h_goals', mu=pm.math.exp(log_h) * dur, observed=df['is_h_goal'])
        pm.Poisson('a_goals', mu=pm.math.exp(log_a) * dur, observed=df['is_a_goal'])
        pm.Poisson('pens', mu=lambda_pen * dur, observed=df['is_penalty'])

        trace = run_advi_inference(
            n_iter=n_iter,
            learning_rate=0.01,
            n_samples=1000,
            start_msg=f"Starting ADVI for {n_teams} teams across {len(df)} events...",
        )
        
    return trace, h_model

if __name__ == "__main__":
    file_paths = DEFAULT_TRAIN_FILE_PATHS
    
    state_map = DEFAULT_STATE_MAP

    print("Flattening JSON files for Team Specific Model...")
    df_events, team_map = load_team_data(file_paths)
    
    df_events = filter_positive_durations(df_events)
    df_events = add_state_codes(df_events, state_map)
    
    print(f"Master Event Ledger created: {len(df_events)} events.")
    print(f"Teams identified: {len(team_map)}")

    posterior_trace, h_model = run_team_advi(df_events, state_map, len(team_map))
    
    print("\nInference Complete.")

    save_trace_to_netcdf(
        posterior_trace,
        "advi_traces/team_nhl_trace.nc",
        success_msg="Saved Team Specific ADVI trace to advi_traces/team_nhl_trace.nc",
    )

    az.plot_forest(posterior_trace, var_names=['off_stars'], combined=True)
    plt.title("Latent Offensive Strength by Team")
    plt.show()
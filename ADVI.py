import pymc as pm
import matplotlib.pyplot as plt
import arviz as az

from LoadGameData import load_game_data
from EmpBayes import calc_priors
from advi_utils import run_advi_inference
from training_pipeline import (
    DEFAULT_TRAIN_FILE_PATHS,
    filter_positive_durations,
    load_and_concat_event_files,
    save_trace_to_netcdf,
)

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
        
        trace = run_advi_inference(
            n_iter=30000,
            learning_rate=0.01,
            n_samples=1000,
            start_msg="Running ADVI Inference...",
        )
        
    return trace, nhl_model


def plot_model_results(trace, state_mapping):
    """
    Generates the two most important plots for your Bayesian analysis.
    """
    
    az.plot_trace(trace, var_names=['lambda_home', 'lambda_away'])
    plt.suptitle("Trace Plot: Scoring Rate Posteriors", fontsize=16)
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    
    az.plot_forest(trace, var_names=['lambda_home'], combined=True, ax=ax)
    
    labels = [state_mapping[i] for i in range(len(state_mapping))]
    ax.set_yticklabels(labels[::-1]) 
    ax.set_title(r"94% Credible Intervals: Home Scoring Rates ($\lambda$) by Manpower State")
    ax.set_xlabel("Expected Goals per Second")
    
    plt.tight_layout()
    fig.savefig('manpower_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    file_paths = DEFAULT_TRAIN_FILE_PATHS
    
    print("Flattening JSON files via LoadGameData module...")
    df_events = load_and_concat_event_files(file_paths, load_game_data)
    print(f"\nMaster Event Table created with {len(df_events)} total events.")
    
    df_events = filter_positive_durations(df_events)
    
    df_ready, priors_df, mapping = calc_priors(df_events, prior_weight_games=20)
    
    posterior_trace, model = run_baseline_advi(df_ready, priors_df, mapping)
    
    save_trace_to_netcdf(
        posterior_trace,
        "nhl_advi_trace.nc",
        success_msg="Saved successfully to nhl_advi_trace.nc",
    )

    plot_model_results(posterior_trace, mapping)
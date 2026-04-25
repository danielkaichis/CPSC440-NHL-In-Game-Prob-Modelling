import math
import pandas as pd
import numpy as np

def _gamma_poisson_nll(theta, y, t, w):
    """Negative log marginal likelihood for Gamma-Poisson model on aggregated rows."""
    log_alpha, log_beta = theta
    alpha = np.exp(log_alpha)
    beta = np.exp(log_beta)

    lgamma_alpha = math.lgamma(alpha)
    lgamma_y_plus_alpha = np.array([math.lgamma(v) for v in (y + alpha)])
    lgamma_y_plus_1 = np.array([math.lgamma(v + 1.0) for v in y])

    log_t = np.where(y > 0, np.log(t), 0.0)
    logp = (
        alpha * np.log(beta)
        - lgamma_alpha
        + lgamma_y_plus_alpha
        - (y + alpha) * np.log(beta + t)
        + y * log_t
        - lgamma_y_plus_1
    )
    return -np.sum(w * logp)


def _fit_gamma_poisson_mle(y, t, max_iter=120):
    """Fit Gamma(alpha, beta) hyperparameters by MLE for Poisson counts with duration t."""
    y = np.asarray(y, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)

    valid = t > 0
    y = y[valid]
    t = t[valid]
    if y.size == 0:
        return 1.0, 1.0

    agg = (
        pd.DataFrame({'y': y, 't': t})
        .groupby(['y', 't'], as_index=False)
        .size()
    )
    y_u = agg['y'].to_numpy(dtype=np.float64)
    t_u = agg['t'].to_numpy(dtype=np.float64)
    w_u = agg['size'].to_numpy(dtype=np.float64)

    rate_mean = np.sum(y) / np.sum(t)
    alpha0 = 1.0
    beta0 = alpha0 / max(rate_mean, 1e-8)
    theta = np.log(np.array([alpha0, beta0], dtype=np.float64))

    lr = 0.2
    eps = 1e-4
    best_theta = theta.copy()
    best_val = _gamma_poisson_nll(theta, y_u, t_u, w_u)

    for _ in range(max_iter):
        grad = np.zeros_like(theta)
        for j in range(2):
            step = np.zeros_like(theta)
            step[j] = eps
            f_plus = _gamma_poisson_nll(theta + step, y_u, t_u, w_u)
            f_minus = _gamma_poisson_nll(theta - step, y_u, t_u, w_u)
            grad[j] = (f_plus - f_minus) / (2 * eps)

        trial = theta - lr * grad
        trial_val = _gamma_poisson_nll(trial, y_u, t_u, w_u)

        if np.isfinite(trial_val) and trial_val < best_val:
            theta = trial
            best_theta = trial
            best_val = trial_val
            lr = min(lr * 1.05, 1.0)
        else:
            lr *= 0.5

        if lr < 1e-6 or np.linalg.norm(grad) < 1e-5:
            break

    alpha_hat, beta_hat = np.exp(best_theta)
    return float(alpha_hat), float(beta_hat)


def calc_priors(df, prior_weight_games=20):
    df['manpower_state'] = df['manpower_state'].astype('category')
    df['state_code'] = df['manpower_state'].cat.codes
    state_mapping = dict(enumerate(df['manpower_state'].cat.categories))
    
    priors = df.groupby('state_code').agg(
        total_duration=('duration_seconds', 'sum'),
        total_home_goals=('is_home_goal', 'sum'),
        total_away_goals=('is_away_goal', 'sum')
    ).reset_index()

    total_duration_all = priors['total_duration'].sum()

    alpha_home_list, beta_home_list = [], []
    alpha_away_list, beta_away_list = [], []

    for state_code in priors['state_code']:
        state_mask = df['state_code'] == state_code
        t_state = df.loc[state_mask, 'duration_seconds'].to_numpy(dtype=np.float64)

        y_home = df.loc[state_mask, 'is_home_goal'].to_numpy(dtype=np.float64)
        y_away = df.loc[state_mask, 'is_away_goal'].to_numpy(dtype=np.float64)

        a_h_raw, b_h_raw = _fit_gamma_poisson_mle(y_home, t_state)
        a_a_raw, b_a_raw = _fit_gamma_poisson_mle(y_away, t_state)

        mean_h = a_h_raw / max(b_h_raw, 1e-12)
        mean_a = a_a_raw / max(b_a_raw, 1e-12)

        state_total_time = float(np.sum(t_state))
        target_beta_state = prior_weight_games * 3600.0 * (state_total_time / max(total_duration_all, 1e-12))

        alpha_home_list.append(mean_h * target_beta_state)
        beta_home_list.append(target_beta_state)
        alpha_away_list.append(mean_a * target_beta_state)
        beta_away_list.append(target_beta_state)

    y_pen = df['is_penalty'].to_numpy(dtype=np.float64)
    t_pen = df['duration_seconds'].to_numpy(dtype=np.float64)
    a_p_raw, b_p_raw = _fit_gamma_poisson_mle(y_pen, t_pen)
    mean_pen = a_p_raw / max(b_p_raw, 1e-12)
    target_beta_pen = prior_weight_games * 3600.0

    eps = 1e-5
    priors['alpha_home'] = np.array(alpha_home_list, dtype=np.float64) + eps
    priors['beta_home'] = np.array(beta_home_list, dtype=np.float64) + eps

    priors['alpha_away'] = np.array(alpha_away_list, dtype=np.float64) + eps
    priors['beta_away'] = np.array(beta_away_list, dtype=np.float64) + eps

    priors['alpha_pen'] = (mean_pen * target_beta_pen) + eps
    priors['beta_pen'] = target_beta_pen + eps
    
    return df, priors, state_mapping


def calc_hierarchical_priors(df, n_teams, prior_weight_games=20):
    """Build empirical-Bayes priors for hierarchical team offense/defense stars.

    Returns a dict with vectors:
    - off_mu, off_sd
    - def_mu, def_sd
    """
    eps = 1e-8

    team_ids = np.arange(n_teams, dtype=np.int64)
    base = pd.DataFrame({'team': team_ids})

    exp_home = df.groupby('h_team_code', as_index=False)['duration_seconds'].sum().rename(
        columns={'h_team_code': 'team', 'duration_seconds': 'exp_home'}
    )
    exp_away = df.groupby('a_team_code', as_index=False)['duration_seconds'].sum().rename(
        columns={'a_team_code': 'team', 'duration_seconds': 'exp_away'}
    )

    gf_home = df.groupby('h_team_code', as_index=False)['is_h_goal'].sum().rename(
        columns={'h_team_code': 'team', 'is_h_goal': 'gf_home'}
    )
    gf_away = df.groupby('a_team_code', as_index=False)['is_a_goal'].sum().rename(
        columns={'a_team_code': 'team', 'is_a_goal': 'gf_away'}
    )

    ga_home = df.groupby('h_team_code', as_index=False)['is_a_goal'].sum().rename(
        columns={'h_team_code': 'team', 'is_a_goal': 'ga_home'}
    )
    ga_away = df.groupby('a_team_code', as_index=False)['is_h_goal'].sum().rename(
        columns={'a_team_code': 'team', 'is_h_goal': 'ga_away'}
    )

    team = base.merge(exp_home, on='team', how='left').merge(exp_away, on='team', how='left')
    team = team.merge(gf_home, on='team', how='left').merge(gf_away, on='team', how='left')
    team = team.merge(ga_home, on='team', how='left').merge(ga_away, on='team', how='left')
    team = team.fillna(0.0)

    exposure = team['exp_home'].to_numpy(dtype=np.float64) + team['exp_away'].to_numpy(dtype=np.float64)
    goals_for = team['gf_home'].to_numpy(dtype=np.float64) + team['gf_away'].to_numpy(dtype=np.float64)
    goals_against = team['ga_home'].to_numpy(dtype=np.float64) + team['ga_away'].to_numpy(dtype=np.float64)

    a_off, b_off = _fit_gamma_poisson_mle(goals_for, exposure)
    a_ga, b_ga = _fit_gamma_poisson_mle(goals_against, exposure)

    off_post_alpha = a_off + goals_for
    off_post_beta = b_off + exposure
    ga_post_alpha = a_ga + goals_against
    ga_post_beta = b_ga + exposure

    off_rate_mean = off_post_alpha / np.maximum(off_post_beta, eps)
    off_rate_var = off_post_alpha / np.maximum(off_post_beta ** 2, eps)

    ga_rate_mean = ga_post_alpha / np.maximum(ga_post_beta, eps)
    ga_rate_var = ga_post_alpha / np.maximum(ga_post_beta ** 2, eps)

    league_off_rate = np.sum(goals_for) / max(np.sum(exposure), eps)
    league_ga_rate = np.sum(goals_against) / max(np.sum(exposure), eps)

    off_mu = np.log(np.maximum(off_rate_mean, eps)) - np.log(max(league_off_rate, eps))
    def_mu = np.log(max(league_ga_rate, eps)) - np.log(np.maximum(ga_rate_mean, eps))

    off_sd = np.sqrt(np.maximum(off_rate_var, eps)) / np.maximum(off_rate_mean, eps)
    def_sd = np.sqrt(np.maximum(ga_rate_var, eps)) / np.maximum(ga_rate_mean, eps)

    scale = prior_weight_games / max(np.sum(exposure) / 3600.0, eps)
    shrink = np.clip(scale, 0.05, 1.0)
    off_sd = np.maximum(off_sd / np.sqrt(shrink), 1e-3)
    def_sd = np.maximum(def_sd / np.sqrt(shrink), 1e-3)

    off_mu = off_mu - np.mean(off_mu)
    def_mu = def_mu - np.mean(def_mu)

    return {
        'off_mu': off_mu.astype(np.float64),
        'off_sd': off_sd.astype(np.float64),
        'def_mu': def_mu.astype(np.float64),
        'def_sd': def_sd.astype(np.float64),
    }
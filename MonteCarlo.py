import numpy as np
import arviz as az

class NHLMonteCarlo:
    def __init__(self, trace_filepath, state_mapping, home_penalty_share=0.52):
        """
        Loads the pre-calculated Bayesian scoring and penalty rates.
        """
        print("Loading ADVI posterior trace...")
        self.trace = az.from_netcdf(trace_filepath)
        self.state_mapping = state_mapping
        
        # Reverse lookup maps state-name strings to posterior tensor column indices.
        self.name_to_idx = {v: k for k, v in state_mapping.items()}
        
        self.home_lambdas = self.trace.posterior['lambda_home'].values.reshape(-1, len(state_mapping))
        self.away_lambdas = self.trace.posterior['lambda_away'].values.reshape(-1, len(state_mapping))
        
        self.gamma_pen = self.trace.posterior['lambda_pen'].values.flatten()
        self.home_penalty_share = float(np.clip(home_penalty_share, 1e-3, 1 - 1e-3))

    def simulate_live_game(self, current_home_score, current_away_score, 
                           time_remaining_sec, current_state_name='5v5', 
                           penalty_sec_remaining=0, n_simulations=10000):
        """
        Runs N Monte Carlo simulations of the remaining game time.
        """
        if current_state_name != '5v5' and penalty_sec_remaining > 0:
            # Simulate current special-teams state first, then simulate the remaining future time.
            current_state_time = min(penalty_sec_remaining, time_remaining_sec)
            future_time = time_remaining_sec - current_state_time
        else:
            current_state_name = '5v5'
            current_state_time = 0
            future_time = time_remaining_sec

        idx_curr = self.name_to_idx.get(current_state_name, self.name_to_idx['5v5'])
        idx_5v5 = self.name_to_idx['5v5']
        idx_h_pp = self.name_to_idx['home_PP_1']
        idx_a_pp = self.name_to_idx['away_PP_1']

        # Each simulation path samples one posterior universe for all rate.
        rand_idx = np.random.randint(0, self.home_lambdas.shape[0], size=n_simulations)
        
        # Get home/away scoring rates for the current state, baseline 5v5 state, and PP states.
        h_rate_curr = self.home_lambdas[rand_idx, idx_curr]
        a_rate_curr = self.away_lambdas[rand_idx, idx_curr]
        
        h_rate_5v5 = self.home_lambdas[rand_idx, idx_5v5]
        a_rate_5v5 = self.away_lambdas[rand_idx, idx_5v5]
        
        h_rate_pp = self.home_lambdas[rand_idx, idx_h_pp]
        a_rate_pp = self.away_lambdas[rand_idx, idx_a_pp]
        
        # Get penalty arrival rates and split by home/away share learned from data.
        p_arrival_rates = self.gamma_pen[rand_idx]

        h_pen_rates = p_arrival_rates * self.home_penalty_share
        a_pen_rates = p_arrival_rates * (1.0 - self.home_penalty_share)
        h_adv_count = np.random.poisson(h_pen_rates * future_time)
        a_adv_count = np.random.poisson(a_pen_rates * future_time)
        
       # Simulate penalty time for each team based on expected penalty arrivals, and an average penalty time of 105s.
        h_pp_time = h_adv_count * 105
        a_pp_time = a_adv_count * 105
        
        h_pp_time = np.minimum(h_pp_time, future_time)
        a_pp_time = np.minimum(a_pp_time, future_time - h_pp_time)
        t_5v5 = np.maximum(0, future_time - h_pp_time - a_pp_time)

        # Simulate goals scored during the current state, then during the future 5v5 and PP phases separately.
        h_g_curr = np.random.poisson(h_rate_curr * current_state_time)
        a_g_curr = np.random.poisson(a_rate_curr * current_state_time)

        # During a home PP, away scores at home-shorthanded rates, and vice versa.
        h_g_pp = np.random.poisson(h_rate_pp * h_pp_time)
        a_g_sh = np.random.poisson(self.away_lambdas[rand_idx, idx_h_pp] * h_pp_time)

        h_g_sh = np.random.poisson(self.home_lambdas[rand_idx, idx_a_pp] * a_pp_time)
        a_g_pp = np.random.poisson(a_rate_pp * a_pp_time)

        h_g_5v5 = np.random.poisson(h_rate_5v5 * t_5v5)
        a_g_5v5 = np.random.poisson(a_rate_5v5 * t_5v5)
        # Final scores are current score plus simulated goals in each phase.
        final_home = current_home_score + h_g_curr + h_g_pp + h_g_sh + h_g_5v5
        final_away = current_away_score + a_g_curr + a_g_pp + a_g_sh + a_g_5v5

        # Win probabilities are Monte Carlo frequencies over simulated final score outcomes.
        home_wins = np.sum(final_home > final_away)
        away_wins = np.sum(final_away > final_home)
        draws = np.sum(final_home == final_away)

        return {
            "Home Win %": (home_wins / n_simulations) * 100,
            "Away Win %": (away_wins / n_simulations) * 100,
            "Overtime %": (draws / n_simulations) * 100
        }
import numpy as np
import arviz as az

class NHLMonteCarlo:
    def __init__(self, trace_filepath, state_mapping):
        """
        Loads the pre-calculated Bayesian scoring and penalty rates.
        """
        print("Loading ADVI posterior trace...")
        self.trace = az.from_netcdf(trace_filepath)
        self.state_mapping = state_mapping
        
        # Look up indices by name (e.g., '5v5' -> 2)
        self.name_to_idx = {v: k for k, v in state_mapping.items()}
        
        # Extract the scoring rates: shape (1000 samples, 10 states)
        self.home_lambdas = self.trace.posterior['lambda_home'].values.reshape(-1, len(state_mapping))
        self.away_lambdas = self.trace.posterior['lambda_away'].values.reshape(-1, len(state_mapping))
        
        # Extract the Penalty Arrival Rate: shape (1000 samples,)
        self.gamma_pen = self.trace.posterior['lambda_pen'].values.flatten()

    def simulate_live_game(self, current_home_score, current_away_score, 
                           time_remaining_sec, current_state_name='5v5', 
                           penalty_sec_remaining=0, n_simulations=10000):
        """
        Runs N Monte Carlo simulations of the remaining game time.
        """
        # Resolve current state time vs future time
        if current_state_name != '5v5' and penalty_sec_remaining > 0:
            current_state_time = min(penalty_sec_remaining, time_remaining_sec)
            future_time = time_remaining_sec - current_state_time
        else:
            current_state_name = '5v5'
            current_state_time = 0
            future_time = time_remaining_sec

        # Get State Indices for Lookups
        idx_curr = self.name_to_idx.get(current_state_name, self.name_to_idx['5v5'])
        idx_5v5 = self.name_to_idx['5v5']
        idx_h_pp = self.name_to_idx['home_PP_1']
        idx_a_pp = self.name_to_idx['away_PP_1']

        # Randomly select 10,000 Bayesian "Universes" from the trace
        rand_idx = np.random.randint(0, self.home_lambdas.shape[0], size=n_simulations)
        
        h_rate_curr = self.home_lambdas[rand_idx, idx_curr]
        a_rate_curr = self.away_lambdas[rand_idx, idx_curr]
        
        h_rate_5v5 = self.home_lambdas[rand_idx, idx_5v5]
        a_rate_5v5 = self.away_lambdas[rand_idx, idx_5v5]
        
        h_rate_pp = self.home_lambdas[rand_idx, idx_h_pp]
        a_rate_pp = self.away_lambdas[rand_idx, idx_a_pp]
        
        # Penalty arrival rates for each simulation
        p_arrival_rates = self.gamma_pen[rand_idx]

        # Predict total future penalties based on remaining 5v5 time
        num_penalties = np.random.poisson(p_arrival_rates * future_time)
        
        # Split penalties between Home and Away advantages
        h_adv_count = np.random.binomial(num_penalties, 0.5)
        a_adv_count = num_penalties - h_adv_count
        
        # Assume average effective PP length (105s) (We can calculate this from the data later if we want, but using 
        # an average is reasonable given the complexity of using various PP lengths in the simulation)
        h_pp_time = h_adv_count * 105
        a_pp_time = a_adv_count * 105
        
        # Cap PP time by total future time and calculate remaining 5v5
        h_pp_time = np.minimum(h_pp_time, future_time)
        a_pp_time = np.minimum(a_pp_time, future_time - h_pp_time)
        t_5v5 = np.maximum(0, future_time - h_pp_time - a_pp_time)


        # Goals during the CURRENT state (e.g., if a penalty is active now)
        h_g_curr = np.random.poisson(h_rate_curr * current_state_time)
        a_g_curr = np.random.poisson(a_rate_curr * current_state_time)

        # Goals during FUTURE Home Powerplays
        h_g_pp = np.random.poisson(h_rate_pp * h_pp_time)
        a_g_sh = np.random.poisson(self.away_lambdas[rand_idx, idx_h_pp] * h_pp_time)

        # Goals during FUTURE Away Powerplays
        h_g_sh = np.random.poisson(self.home_lambdas[rand_idx, idx_a_pp] * a_pp_time)
        a_g_pp = np.random.poisson(a_rate_pp * a_pp_time)

        # Goals during future 5v5 time
        h_g_5v5 = np.random.poisson(h_rate_5v5 * t_5v5)
        a_g_5v5 = np.random.poisson(a_rate_5v5 * t_5v5)

        # FINAL RESULTS
        final_home = current_home_score + h_g_curr + h_g_pp + h_g_sh + h_g_5v5
        final_away = current_away_score + a_g_curr + a_g_pp + a_g_sh + a_g_5v5

        home_wins = np.sum(final_home > final_away)
        away_wins = np.sum(final_away > final_home)
        draws = np.sum(final_home == final_away)

        return {
            "Home Win %": (home_wins / n_simulations) * 100,
            "Away Win %": (away_wins / n_simulations) * 100,
            "Overtime %": (draws / n_simulations) * 100
        }
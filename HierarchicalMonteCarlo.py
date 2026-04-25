import numpy as np
import arviz as az
import json

class HierarchicalMonteCarlo:
    def __init__(self, trace_path, state_mapping, team_mapping_path, home_penalty_share=0.52):
        print("Loading Hierarchical Brain...")
        trace = az.from_netcdf(trace_path)

        if hasattr(trace, 'load'):
            trace.load()
        self.state_map = state_mapping
        self.name_to_idx = {v: k for k, v in state_mapping.items()}
        
        with open(team_mapping_path, 'r') as f:
            self.team_map = {int(k): v for k, v in json.load(f).items()}
            
        post = trace.posterior
        
        n_teams_in_trace = post.sizes['off_stars_dim_0'] 
        
        self.off = post['off_stars'].values.reshape(-1, n_teams_in_trace)
        self.dims = post['def_stars'].values.reshape(-1, n_teams_in_trace)
        self.ints = post['state_ints'].values.reshape(-1, post.sizes['state_ints_dim_0'])
        
        self.h_ice = post['home_ice'].values.flatten()
        self.g_pen = post['lambda_pen'].values.flatten()
        self.home_penalty_share = float(np.clip(home_penalty_share, 1e-3, 1 - 1e-3))
        
        print(f"Model loaded with {n_teams_in_trace} teams and {len(state_mapping)} game states.")
        
        if n_teams_in_trace != len(self.team_map):
            print(f"WARNING: Trace has {n_teams_in_trace} teams but mapping has {len(self.team_map)}.")

        if hasattr(trace, 'close'):
            trace.close()

    def simulate(self, h_id, a_id, h_score, a_score, t_rem, state_name, pen_rem=0, n_sims=10000):
        h_c, a_c = self.team_map[h_id], self.team_map[a_id]
        
        s_idx = self.name_to_idx.get(state_name, self.name_to_idx['5v5'])
        idx_5v5 = self.name_to_idx['5v5']
        idx_h_pp = self.name_to_idx['home_PP_1']
        idx_a_pp = self.name_to_idx['away_PP_1']
        
        u = np.random.randint(0, self.off.shape[0], size=n_sims)

        
        h_rate_curr = np.exp(self.ints[u, s_idx] + self.h_ice[u] + self.off[u, h_c] - self.dims[u, a_c])
        a_rate_curr = np.exp(self.ints[u, s_idx] + self.off[u, a_c] - self.dims[u, h_c])
        
        h_rate_5v5 = np.exp(self.ints[u, idx_5v5] + self.h_ice[u] + self.off[u, h_c] - self.dims[u, a_c])
        a_rate_5v5 = np.exp(self.ints[u, idx_5v5] + self.off[u, a_c] - self.dims[u, h_c])

        h_rate_pp = np.exp(self.ints[u, idx_h_pp] + self.h_ice[u] + self.off[u, h_c] - self.dims[u, a_c])
        a_rate_pp = np.exp(self.ints[u, idx_a_pp] + self.off[u, a_c] - self.dims[u, h_c])

        future_time = max(0, t_rem - pen_rem)
        h_pen_rates = self.g_pen[u] * self.home_penalty_share
        a_pen_rates = self.g_pen[u] * (1.0 - self.home_penalty_share)
        h_adv_count = np.random.poisson(h_pen_rates * future_time)
        a_adv_count = np.random.poisson(a_pen_rates * future_time)

        h_pp_time = h_adv_count * 105
        a_pp_time = a_adv_count * 105
        
        h_pp_time = np.minimum(h_pp_time, future_time)
        a_pp_time = np.minimum(a_pp_time, future_time - h_pp_time)
        t_5v5 = np.maximum(0, future_time - h_pp_time - a_pp_time)

        h_goals = (np.random.poisson(h_rate_curr * pen_rem) + 
                   np.random.poisson(h_rate_5v5 * t_5v5) + 
                   np.random.poisson(h_rate_pp * h_pp_time))
                  
        a_goals = (np.random.poisson(a_rate_curr * pen_rem) + 
                   np.random.poisson(a_rate_5v5 * t_5v5) + 
                   np.random.poisson(a_rate_pp * a_pp_time))

        return {
            "Home Win %": np.mean((h_score + h_goals) > (a_score + a_goals)) * 100,
            "Away Win %": np.mean((a_score + a_goals) > (h_score + h_goals)) * 100,
            "OT %": np.mean((h_score + h_goals) == (a_score + a_goals)) * 100
        }
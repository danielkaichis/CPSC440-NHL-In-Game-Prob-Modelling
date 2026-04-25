from.team_load_data import load_team_data
from .team_monte_carlo import TeamMonteCarlo
from utils.persistence_utils import save_checkpoint_scores
from utils.training_utils import DEFAULT_TRAIN_FILE_PATHS, DEFAULT_STATE_MAP
from utils.penalty_utils import estimate_home_penalty_share
from utils.evaluation_utils import (
    DEFAULT_CHECKPOINTS,
    evaluate_checkpoints,
)

def evaluate_team_model(test_df, mc_engine, checkpoints):
    def _predict_home_prob(sit, t_rem):
        res = mc_engine.simulate(
            h_id=int(sit["home_team_id"]),
            a_id=int(sit["away_team_id"]),
            h_score=int(sit["home_score"]),
            a_score=int(sit["away_score"]),
            t_rem=t_rem,
            state_name=sit["manpower_state"],
        )
        # Monte Carlo output is percent, convert back to probability scale for evaluation.
        return res["Home Win %"] / 100.0

    return evaluate_checkpoints(
        test_df,
        checkpoints,
        predict_home_win_prob=_predict_home_prob,
        progress_msg=lambda t_rem, n_games: f"Evaluating {n_games} games at {t_rem//60} mins left...",
    )

if __name__ == "__main__":
    state_map = DEFAULT_STATE_MAP
    
    penalty_share = estimate_home_penalty_share(DEFAULT_TRAIN_FILE_PATHS)
    print(f"Using data-driven home penalty share: {penalty_share:.3f}")
    mc = TeamMonteCarlo(
        "advi_traces/team_nhl_trace.nc",
        state_map,
        "team_mapping.json",
        home_penalty_share=penalty_share,
    )
    
    test_df, _ = load_team_data(['nhl_raw_data/2025_2026_part2.json'])
    
    checkpoints = DEFAULT_CHECKPOINTS
    h_scores = evaluate_team_model(test_df, mc, checkpoints)
    save_path = save_checkpoint_scores(h_scores, model_name="team")
    print(f"Saved team specific checkpoint results to {save_path}")
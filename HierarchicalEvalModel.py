from HierarchicalLoadData import load_h_data
from HierarchicalMonteCarlo import HierarchicalMonteCarlo
from results_store import save_checkpoint_scores
from training_pipeline import DEFAULT_TRAIN_FILE_PATHS, DEFAULT_STATE_MAP
from penalty_utils import estimate_home_penalty_share
from evaluation_common import (
    DEFAULT_CHECKPOINTS,
    evaluate_checkpoints,
)

def evaluate_h_model(test_df, mc_engine, checkpoints):
    def _predict_home_prob(sit, t_rem):
        res = mc_engine.simulate(
            h_id=int(sit["home_team_id"]),
            a_id=int(sit["away_team_id"]),
            h_score=int(sit["home_score"]),
            a_score=int(sit["away_score"]),
            t_rem=t_rem,
            state_name=sit["manpower_state"],
        )
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
    mc = HierarchicalMonteCarlo(
        "h_nhl_trace.nc",
        state_map,
        "h_team_mapping.json",
        home_penalty_share=penalty_share,
    )
    
    test_df, _ = load_h_data(['nhl_raw_data/2025_2026_part2.json'])
    
    checkpoints = DEFAULT_CHECKPOINTS
    h_scores = evaluate_h_model(test_df, mc, checkpoints)
    save_path = save_checkpoint_scores(h_scores, model_name="hierarchical")
    print(f"Saved hierarchical checkpoint results to {save_path}")
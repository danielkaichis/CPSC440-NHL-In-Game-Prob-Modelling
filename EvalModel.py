from LoadGameData import load_game_data
from MonteCarlo import NHLMonteCarlo
from results_store import save_checkpoint_scores
from training_pipeline import DEFAULT_TRAIN_FILE_PATHS, DEFAULT_STATE_MAP
from penalty_utils import estimate_home_penalty_share
from evaluation_common import (
    DEFAULT_CHECKPOINTS,
    evaluate_checkpoints,
)

def evaluate_performance(test_df, mc_engine, checkpoints):
    def _predict_home_prob(sit, t_rem):
        res = mc_engine.simulate_live_game(
            current_home_score=int(sit["home_score"]),
            current_away_score=int(sit["away_score"]),
            time_remaining_sec=t_rem,
            current_state_name=sit["manpower_state"],
            n_simulations=2000,
        )
        return res["Home Win %"] / 100.0

    return evaluate_checkpoints(
        test_df,
        checkpoints,
        predict_home_win_prob=_predict_home_prob,
        progress_msg=lambda t_rem, _n_games: f"Analyzing games with {t_rem//60} minutes left...",
    )

if __name__ == "__main__":
    mapping = DEFAULT_STATE_MAP
    
    penalty_share = estimate_home_penalty_share(DEFAULT_TRAIN_FILE_PATHS)
    print(f"Using data-driven home penalty share: {penalty_share:.3f}")
    mc = NHLMonteCarlo("nhl_advi_trace.nc", mapping, home_penalty_share=penalty_share)
    test_data = load_game_data('nhl_raw_data/2025_2026_part2.json')

    checkpoints = DEFAULT_CHECKPOINTS
    scores = evaluate_performance(test_data, mc, checkpoints)
    save_path = save_checkpoint_scores(scores, model_name="baseline")
    print(f"Saved baseline checkpoint results to {save_path}")
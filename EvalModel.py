import numpy as np
from LoadGameData import load_game_data
from MonteCarlo import NHLMonteCarlo
from results_store import save_checkpoint_scores
from training_pipeline import DEFAULT_TRAIN_FILE_PATHS, DEFAULT_STATE_MAP
from penalty_utils import estimate_home_penalty_share
from evaluation_common import (
    DEFAULT_CHECKPOINTS,
    evaluate_checkpoints,
)

_PP_STATES = {"home_PP_1", "home_PP_2", "away_PP_1", "away_PP_2"}

_N_SIMULATIONS = 5000
_PROB_CLIP_LOW = 1e-6
_PROB_CLIP_HIGH = 1.0 - 1e-6


def _attach_penalty_seconds_remaining(df):
    """Estimate remaining time in the current manpower segment for each event row."""
    out = df.copy()
    out["penalty_sec_remaining"] = 0

    for _game_id, game_data in out.groupby("game_id", sort=False):
        g = game_data.sort_values("time_remaining", ascending=False)
        times = g["time_remaining"].to_numpy(dtype=np.int64)
        states = g["manpower_state"].astype(str).to_numpy()
        rem = np.zeros(len(g), dtype=np.int64)

        i = 0
        n = len(g)
        while i < n:
            state = states[i]
            j = i + 1
            while j < n and states[j] == state:
                j += 1

            if state in _PP_STATES:
                # Approximate remaining penalty as time until this PP segment ends.
                segment_end_time = times[j] if j < n else 0
                rem[i:j] = np.maximum(0, times[i:j] - segment_end_time)
            i = j
        out.loc[g.index, "penalty_sec_remaining"] = rem

    return out


def evaluate_performance(test_df, mc_engine, checkpoints):
    """Evaluate model predictions at fixed checkpoints for each game in test_df by 
    simulating forward from the nearest observed game state to the checkpoint time."""
    eval_df = _attach_penalty_seconds_remaining(test_df)

    def _predict_home_prob(sit, t_rem):
        penalty_sec_remaining = int(max(0, sit.get("penalty_sec_remaining", 0)))
        res = mc_engine.simulate_live_game(
            current_home_score=int(sit["home_score"]),
            current_away_score=int(sit["away_score"]),
            time_remaining_sec=t_rem,
            current_state_name=sit["manpower_state"],
            penalty_sec_remaining=penalty_sec_remaining,
            n_simulations=_N_SIMULATIONS,
        )

        home_pct = res["Home Win %"]
        away_pct = res["Away Win %"]
        decided = home_pct + away_pct

        # normalise out ties
        raw_prob = (home_pct / decided) if decided > 0 else 0.5

        return float(np.clip(raw_prob, _PROB_CLIP_LOW, _PROB_CLIP_HIGH))

    return evaluate_checkpoints(
        eval_df,
        checkpoints,
        predict_home_win_prob=_predict_home_prob,
        progress_msg=lambda t_rem, _n_games: (
            f"Analyzing games with {t_rem // 60} minutes left..."
        ),
    )


if __name__ == "__main__":
    mapping = DEFAULT_STATE_MAP

    penalty_share = estimate_home_penalty_share(DEFAULT_TRAIN_FILE_PATHS)
    print(f"Using data-driven home penalty share: {penalty_share:.3f}")

    mc = NHLMonteCarlo("models/nhl_advi_trace.nc", mapping, home_penalty_share=penalty_share)
    test_data = load_game_data("nhl_raw_data/2025_2026_part2.json")

    scores = evaluate_performance(test_data, mc, DEFAULT_CHECKPOINTS)
    save_path = save_checkpoint_scores(scores, model_name="baseline")
    print(f"Saved baseline checkpoint results to {save_path}")
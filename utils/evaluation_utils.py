import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


DEFAULT_CHECKPOINTS = [3000, 2400, 1800, 1200, 600, 120]


def evaluate_checkpoints(test_df, checkpoints, predict_home_win_prob, progress_msg):
    """Evaluate model predictions at fixed checkpoints for each game in test_df."""
    scores = {}
    game_ids = test_df["game_id"].unique()

    for t_rem in checkpoints:
        print(progress_msg(t_rem, len(game_ids)))
        preds, actuals = [], []

        for g_id in game_ids:
            game_data = test_df[test_df["game_id"] == g_id]
            # Evaluate the nearest observed game state to the checkpoint time.
            idx = (game_data["time_remaining"] - t_rem).abs().idxmin()
            sit = game_data.loc[idx]

            final_row = game_data.iloc[-1]
            home_won = 1 if final_row["home_score"] > final_row["away_score"] else 0

            preds.append(float(predict_home_win_prob(sit, t_rem)))
            actuals.append(home_won)

        pred_labels = [1 if p >= 0.5 else 0 for p in preds]
        acc = float(np.mean(np.array(pred_labels) == np.array(actuals)))

        scores[t_rem] = {
            "BS": brier_score_loss(actuals, preds),
            "LL": log_loss(actuals, np.clip(preds, 1e-6, 1 - 1e-6), labels=[0, 1]),
            "ACC": acc,
        }

    return scores
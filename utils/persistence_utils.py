import os
from datetime import datetime
import pandas as pd

def _stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_20_day_window_results(df_results, model_name, run_name=None, results_dir="results"):
    run_id = run_name or _stamp()

    out = df_results.copy()
    out["model"] = model_name
    out["run_id"] = run_id
    out["eval_type"] = "sliding"
    out["saved_at"] = datetime.now().isoformat()

    file_path = os.path.join(results_dir, f"sliding_{model_name}_{run_id}.csv")
    out.to_csv(file_path, index=False)
    return file_path

def save_checkpoint_scores(scores, model_name, metric_name="BS", run_name=None, results_dir="results"):
    run_id = run_name or _stamp()

    rows = []
    for t_rem, value in scores.items():
        row = {
            "checkpoint_sec": int(t_rem),
            "mins_left": int(t_rem) // 60,
        }

        if isinstance(value, dict):
            # Accept multi-metric score payloads (e.g., BS/LL/ACC) without schema changes.
            for k, v in value.items():
                row[str(k)] = float(v)
        else:
            row[metric_name] = float(value)

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("checkpoint_sec", ascending=False)
    out["model"] = model_name
    out["run_id"] = run_id
    out["eval_type"] = "checkpoint"
    out["saved_at"] = datetime.now().isoformat()

    file_path = os.path.join(results_dir, f"checkpoint_{model_name}_{run_id}.csv")
    out.to_csv(file_path, index=False)
    return file_path

def save_dataframe(df, prefix, run_name=None, results_dir="results"):
    run_id = run_name or _stamp()
    file_path = os.path.join(results_dir, f"{prefix}_{run_id}.csv")
    df.to_csv(file_path, index=False)
    return file_path

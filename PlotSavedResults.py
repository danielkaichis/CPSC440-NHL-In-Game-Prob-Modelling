import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"

def _latest_per_model(df, model_col="model", run_col="run_id", time_col="saved_at"):
    if df.empty:
        return df
    df = df.copy()
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        idx = df.sort_values(time_col).groupby(model_col)[time_col].idxmax()
    else:
        idx = df.groupby(model_col)[run_col].idxmax()
    return df.loc[idx]


def plot_saved_results(results_dir=RESULTS_DIR):
    sliding_files = glob.glob(os.path.join(results_dir, "sliding_*.csv"))
    checkpoint_files = glob.glob(os.path.join(results_dir, "checkpoint_*.csv"))
    compare_files = glob.glob(os.path.join(results_dir, "comparison_*.csv"))

    if not (sliding_files or checkpoint_files or compare_files):
        print(f"No saved result files found in {results_dir}.")
        return

    if sliding_files:
        sliding = pd.concat([pd.read_csv(p) for p in sliding_files], ignore_index=True)
        sliding["date"] = pd.to_datetime(sliding["date"], errors="coerce")

        has_checkpoint_cols = (
            "checkpoint_sec" in sliding.columns
            and "mins_left" in sliding.columns
            and sliding["checkpoint_sec"].nunique() > 1
        )

        latest_sliding = _latest_per_model(sliding)
        if not latest_sliding.empty:
            if has_checkpoint_cols:
                mins_levels = sorted(sliding["mins_left"].dropna().unique(), reverse=True)

                fig, ax1 = plt.subplots(figsize=(12, 6))
                for model in latest_sliding["model"].unique():
                    model_rows = sliding[sliding["model"] == model]
                    run_id = latest_sliding[latest_sliding["model"] == model]["run_id"].iloc[0]
                    run_rows = model_rows[model_rows["run_id"] == run_id]

                    for m in mins_levels:
                        rows = run_rows[run_rows["mins_left"] == m].sort_values("date")
                        if not rows.empty:
                            ax1.plot(rows["date"], rows["BS"], marker="o", label=f"{model} - {int(m)}m")

                ax1.set_title("Latest Sliding-Window Runs (Brier Score by Checkpoint)")
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Brier Score")
                ax1.grid(alpha=0.3)
                ax1.legend(ncol=2)
                plt.tight_layout()
                plt.show()

                if "LL" in sliding.columns:
                    fig, ax2 = plt.subplots(figsize=(12, 6))
                    for model in latest_sliding["model"].unique():
                        model_rows = sliding[sliding["model"] == model]
                        run_id = latest_sliding[latest_sliding["model"] == model]["run_id"].iloc[0]
                        run_rows = model_rows[model_rows["run_id"] == run_id]

                        for m in mins_levels:
                            rows = run_rows[run_rows["mins_left"] == m].sort_values("date")
                            if not rows.empty:
                                ax2.plot(rows["date"], rows["LL"], marker="s", label=f"{model} - {int(m)}m")

                    ax2.set_title("Latest Sliding-Window Runs (Log-Loss by Checkpoint)")
                    ax2.set_xlabel("Date")
                    ax2.set_ylabel("Log-Loss")
                    ax2.grid(alpha=0.3)
                    ax2.legend(ncol=2)
                    plt.tight_layout()
                    plt.show()

                if "ACC" in sliding.columns:
                    fig, ax3 = plt.subplots(figsize=(12, 6))
                    for model in latest_sliding["model"].unique():
                        model_rows = sliding[sliding["model"] == model]
                        run_id = latest_sliding[latest_sliding["model"] == model]["run_id"].iloc[0]
                        run_rows = model_rows[model_rows["run_id"] == run_id]

                        for m in mins_levels:
                            rows = run_rows[run_rows["mins_left"] == m].sort_values("date")
                            if not rows.empty:
                                ax3.plot(rows["date"], rows["ACC"], marker="^", label=f"{model} - {int(m)}m")

                    ax3.set_title("Latest Sliding-Window Runs (Accuracy by Checkpoint)")
                    ax3.set_xlabel("Date")
                    ax3.set_ylabel("Accuracy")
                    ax3.set_ylim(0, 1)
                    ax3.grid(alpha=0.3)
                    ax3.legend(ncol=2)
                    plt.tight_layout()
                    plt.show()
            else:
                fig, ax1 = plt.subplots(figsize=(12, 6))
                for model in latest_sliding["model"].unique():
                    model_rows = sliding[sliding["model"] == model]
                    run_id = latest_sliding[latest_sliding["model"] == model]["run_id"].iloc[0]
                    run_rows = model_rows[model_rows["run_id"] == run_id].sort_values("date")
                    ax1.plot(run_rows["date"], run_rows["BS"], marker="o", label=f"{model} BS")

                ax1.set_title("Latest Sliding-Window Runs")
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Brier Score")
                ax1.grid(alpha=0.3)
                ax1.legend()
                plt.tight_layout()
                plt.show()

                fig, ax2 = plt.subplots(figsize=(12, 6))
                for model in latest_sliding["model"].unique():
                    model_rows = sliding[sliding["model"] == model]
                    run_id = latest_sliding[latest_sliding["model"] == model]["run_id"].iloc[0]
                    run_rows = model_rows[model_rows["run_id"] == run_id].sort_values("date")
                    if "LL" in run_rows.columns:
                        ax2.plot(run_rows["date"], run_rows["LL"], marker="s", label=f"{model} LL")

                ax2.set_title("Latest Sliding-Window Log-Loss")
                ax2.set_xlabel("Date")
                ax2.set_ylabel("Log-Loss")
                ax2.grid(alpha=0.3)
                ax2.legend()
                plt.tight_layout()
                plt.show()

                if "ACC" in sliding.columns:
                    fig, ax3 = plt.subplots(figsize=(12, 6))
                    for model in latest_sliding["model"].unique():
                        model_rows = sliding[sliding["model"] == model]
                        run_id = latest_sliding[latest_sliding["model"] == model]["run_id"].iloc[0]
                        run_rows = model_rows[model_rows["run_id"] == run_id].sort_values("date")
                        if "ACC" in run_rows.columns:
                            ax3.plot(run_rows["date"], run_rows["ACC"], marker="^", label=f"{model} ACC")

                    ax3.set_title("Latest Sliding-Window Accuracy")
                    ax3.set_xlabel("Date")
                    ax3.set_ylabel("Accuracy")
                    ax3.set_ylim(0, 1)
                    ax3.grid(alpha=0.3)
                    ax3.legend()
                    plt.tight_layout()
                    plt.show()

        # Also show all sliding runs, not just the latest, to compare historical runs.
        if "run_id" in sliding.columns and sliding["run_id"].nunique() > 1:
            fig, ax_all = plt.subplots(figsize=(12, 6))
            if has_checkpoint_cols:
                # For checkpoint-style sliding results, summarize each run by averaging BS over checkpoints.
                all_bs = (
                    sliding.groupby(["model", "run_id", "date"], as_index=False)["BS"].mean()
                    .sort_values("date")
                )
                for (model, run_id), rows in all_bs.groupby(["model", "run_id"]):
                    ax_all.plot(rows["date"], rows["BS"], marker="o", alpha=0.8, label=f"{model} ({run_id})")
                ax_all.set_title("All Sliding Runs (Mean BS Across Checkpoints)")
            else:
                for (model, run_id), rows in sliding.groupby(["model", "run_id"]):
                    rows = rows.sort_values("date")
                    ax_all.plot(rows["date"], rows["BS"], marker="o", alpha=0.8, label=f"{model} ({run_id})")
                ax_all.set_title("All Sliding Runs (Brier Score)")

            ax_all.set_xlabel("Date")
            ax_all.set_ylabel("Brier Score")
            ax_all.grid(alpha=0.3)
            ax_all.legend(ncol=2)
            plt.tight_layout()
            plt.show()

    if checkpoint_files:
        ckpt = pd.concat([pd.read_csv(p) for p in checkpoint_files], ignore_index=True)
        latest_ckpt = _latest_per_model(ckpt)

        if not latest_ckpt.empty:
            plt.figure(figsize=(10, 6))
            for model in latest_ckpt["model"].unique():
                run_id = latest_ckpt[latest_ckpt["model"] == model]["run_id"].iloc[0]
                rows = ckpt[(ckpt["model"] == model) & (ckpt["run_id"] == run_id)].sort_values("mins_left")
                if "BS" in rows.columns:
                    plt.plot(rows["mins_left"], rows["BS"], marker="o", label=model)

            plt.gca().invert_xaxis()
            plt.title("Checkpoint Brier Score (Latest Runs)")
            plt.xlabel("Minutes Remaining")
            plt.ylabel("Brier Score")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

            if "ACC" in ckpt.columns:
                plt.figure(figsize=(10, 6))
                for model in latest_ckpt["model"].unique():
                    run_id = latest_ckpt[latest_ckpt["model"] == model]["run_id"].iloc[0]
                    rows = ckpt[(ckpt["model"] == model) & (ckpt["run_id"] == run_id)].sort_values("mins_left")
                    if "ACC" in rows.columns:
                        plt.plot(rows["mins_left"], rows["ACC"], marker="^", label=model)

                plt.gca().invert_xaxis()
                plt.title("Checkpoint Accuracy (Latest Runs)")
                plt.xlabel("Minutes Remaining")
                plt.ylabel("Accuracy")
                plt.ylim(0, 1)
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()

    if compare_files:
        comp = pd.concat([pd.read_csv(p) for p in compare_files], ignore_index=True)
        comp = comp.sort_values("Mins Left")

        plt.figure(figsize=(10, 6))
        if "Baseline BS" in comp.columns and "Hierarchical BS" in comp.columns:
            plt.plot(comp["Mins Left"], comp["Baseline BS"], marker="o", label="Baseline BS")
            plt.plot(comp["Mins Left"], comp["Hierarchical BS"], marker="s", label="Hierarchical BS")
            plt.gca().invert_xaxis()
            plt.title("Baseline vs Hierarchical Brier Score")
            plt.xlabel("Minutes Remaining")
            plt.ylabel("Brier Score")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

        if "Baseline ACC" in comp.columns and "Hierarchical ACC" in comp.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(comp["Mins Left"], comp["Baseline ACC"], marker="o", label="Baseline ACC")
            plt.plot(comp["Mins Left"], comp["Hierarchical ACC"], marker="s", label="Hierarchical ACC")
            plt.gca().invert_xaxis()
            plt.title("Baseline vs Hierarchical Accuracy")
            plt.xlabel("Minutes Remaining")
            plt.ylabel("Accuracy")
            plt.ylim(0, 1)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    plot_saved_results()

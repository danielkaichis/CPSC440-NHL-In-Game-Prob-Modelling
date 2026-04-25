import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"


def _print_table(title, df):
    if df is None or df.empty:
        return
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(df.to_string(index=False))

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

    if not (sliding_files or checkpoint_files):
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
                latest_rows = []

                fig, ax1 = plt.subplots(figsize=(12, 6))
                for model in latest_sliding["model"].unique():
                    model_rows = sliding[sliding["model"] == model]
                    run_id = latest_sliding[latest_sliding["model"] == model]["run_id"].iloc[0]
                    run_rows = model_rows[model_rows["run_id"] == run_id]
                    latest_rows.append(run_rows.copy())

                    for m in mins_levels:
                        rows = run_rows[run_rows["mins_left"] == m].sort_values("date")
                        if not rows.empty:
                            ax1.plot(rows["date"], rows["BS"], marker="o", label=f"{int(m)} minutes left")

                ax1.set_title("Brier Score by 20 Day Window")
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Brier Score")
                ax1.grid(alpha=0.3)
                ax1.legend(ncol=2)
                plt.tight_layout()
                plt.show()

                if latest_rows:
                    latest_df = pd.concat(latest_rows, ignore_index=True).sort_values(
                        ["model", "date", "mins_left"], ascending=[True, True, False]
                    )
                    cols = ["model", "run_id", "date", "mins_left", "BS"]
                    if "LL" in latest_df.columns:
                        cols.append("LL")
                    if "ACC" in latest_df.columns:
                        cols.append("ACC")
                    _print_table("20 Day Window Values", latest_df[cols])

                if "LL" in sliding.columns:
                    fig, ax2 = plt.subplots(figsize=(12, 6))
                    for model in latest_sliding["model"].unique():
                        model_rows = sliding[sliding["model"] == model]
                        run_id = latest_sliding[latest_sliding["model"] == model]["run_id"].iloc[0]
                        run_rows = model_rows[model_rows["run_id"] == run_id]

                        for m in mins_levels:
                            rows = run_rows[run_rows["mins_left"] == m].sort_values("date")
                            if not rows.empty:
                                ax2.plot(rows["date"], rows["LL"], marker="s", label=f"{int(m)} minutes left")

                    ax2.set_title("Log Loss by 20 Day Window")
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
                                ax3.plot(rows["date"], rows["ACC"], marker="^", label=f"{int(m)} minutes left")

                    ax3.set_title("Accuracy by 20 Day Window")
                    ax3.set_xlabel("Date")
                    ax3.set_ylabel("Accuracy")
                    ax3.set_ylim(0, 1)
                    ax3.axhline(0.557, color="red", linestyle="--", alpha=0.5, label="Always Guess The Favourite")
                    ax3.grid(alpha=0.3)
                    ax3.legend(ncol=2)
                    plt.tight_layout()
                    plt.show()
            else:
                fig, ax1 = plt.subplots(figsize=(12, 6))
                latest_rows = []
                for model in latest_sliding["model"].unique():
                    model_rows = sliding[sliding["model"] == model]
                    run_id = latest_sliding[latest_sliding["model"] == model]["run_id"].iloc[0]
                    run_rows = model_rows[model_rows["run_id"] == run_id].sort_values("date")
                    latest_rows.append(run_rows.copy())
                    ax1.plot(run_rows["date"], run_rows["BS"], marker="o", label=f"{model} BS")

                ax1.set_title("20 Day Window Brier Score")
                ax1.set_xlabel("Date")
                ax1.set_ylabel("Brier Score")
                ax1.axhline(0.25, color="red", linestyle="--", alpha=0.5, label="Random Chance")
                ax1.grid(alpha=0.3)
                ax1.legend()
                plt.tight_layout()
                plt.show()

                if latest_rows:
                    latest_df = pd.concat(latest_rows, ignore_index=True).sort_values(["model", "date"])
                    cols = ["model", "run_id", "date", "BS"]
                    if "LL" in latest_df.columns:
                        cols.append("LL")
                    if "ACC" in latest_df.columns:
                        cols.append("ACC")
                    _print_table("Latest Sliding Runs Values", latest_df[cols])

                fig, ax2 = plt.subplots(figsize=(12, 6))
                for model in latest_sliding["model"].unique():
                    model_rows = sliding[sliding["model"] == model]
                    run_id = latest_sliding[latest_sliding["model"] == model]["run_id"].iloc[0]
                    run_rows = model_rows[model_rows["run_id"] == run_id].sort_values("date")
                    if "LL" in run_rows.columns:
                        ax2.plot(run_rows["date"], run_rows["LL"], marker="s", label=f"{model} LL")

                ax2.set_title("20 Day Window Log-Loss")
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

                    ax3.set_title("20 Day Window Accuracy")
                    ax3.set_xlabel("Date")
                    ax3.set_ylabel("Accuracy")
                    ax3.set_ylim(0, 1)
                    ax3.grid(alpha=0.3)
                    ax3.legend()
                    plt.tight_layout()
                    plt.show()


    if checkpoint_files:
        ckpt = pd.concat([pd.read_csv(p) for p in checkpoint_files], ignore_index=True)
        latest_ckpt = _latest_per_model(ckpt)

        if not latest_ckpt.empty:
            ckpt_latest_rows = []
            plt.figure(figsize=(10, 6))
            for model in latest_ckpt["model"].unique():
                run_id = latest_ckpt[latest_ckpt["model"] == model]["run_id"].iloc[0]
                rows = ckpt[(ckpt["model"] == model) & (ckpt["run_id"] == run_id)].sort_values("mins_left")
                ckpt_latest_rows.append(rows.copy())
                if "BS" in rows.columns:
                    plt.plot(rows["mins_left"], rows["BS"], marker="o", label=model)

            plt.gca().invert_xaxis()
            plt.title("Baseline vs Hierarchical Model Brier Scores")
            plt.xlabel("Minutes Remaining")
            plt.ylabel("Brier Score")
            plt.axhline(0.25, color="red", linestyle="--", alpha=0.5, label="Random Chance")
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

            if "LL" in ckpt.columns:
                plt.figure(figsize=(10, 6))
                for model in latest_ckpt["model"].unique():
                    run_id = latest_ckpt[latest_ckpt["model"] == model]["run_id"].iloc[0]
                    rows = ckpt[(ckpt["model"] == model) & (ckpt["run_id"] == run_id)].sort_values("mins_left")
                    if "LL" in rows.columns:
                        plt.plot(rows["mins_left"], rows["LL"], marker="s", label=model)

                plt.gca().invert_xaxis()
                plt.title("Baseline vs Hierarchical Model Log-Loss")
                plt.xlabel("Minutes Remaining")
                plt.ylabel("Log-Loss")
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()

            if ckpt_latest_rows:
                ckpt_latest_df = pd.concat(ckpt_latest_rows, ignore_index=True).sort_values(
                    ["model", "mins_left"], ascending=[True, False]
                )
                cols = ["model", "run_id", "mins_left", "BS"]
                if "ACC" in ckpt_latest_df.columns:
                    cols.append("ACC")
                _print_table("Checkpoint Latest Runs Values", ckpt_latest_df[cols])

            if "ACC" in ckpt.columns:
                plt.figure(figsize=(10, 6))
                for model in latest_ckpt["model"].unique():
                    run_id = latest_ckpt[latest_ckpt["model"] == model]["run_id"].iloc[0]
                    rows = ckpt[(ckpt["model"] == model) & (ckpt["run_id"] == run_id)].sort_values("mins_left")
                    if "ACC" in rows.columns:
                        plt.plot(rows["mins_left"], rows["ACC"], marker="^", label=model)

                plt.gca().invert_xaxis()
                plt.title("Baseline vs Hierarchical Model Accuracy")
                plt.xlabel("Minutes Remaining")
                plt.ylabel("Accuracy")
                plt.ylim(0, 1)
                plt.axhline(0.557, color="red", linestyle="--", alpha=0.5, label="Always Guess The Favourite")
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()

if __name__ == "__main__":
    plot_saved_results()

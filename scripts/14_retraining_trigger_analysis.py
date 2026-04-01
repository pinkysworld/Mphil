"""
14_retraining_trigger_analysis.py — Retraining-trigger analysis from walk-forward drift.

Derives a simple, auditable retraining policy from the archived monthly
walk-forward profile.

Outputs:
  - artifacts/retraining_trigger/retraining_trigger_windows.csv
  - artifacts/retraining_trigger/retraining_trigger_episodes.csv
  - artifacts/retraining_trigger/retraining_trigger_summary.json
  - artifacts/retraining_trigger/retraining_trigger_plot.png
  - artifacts/retraining_trigger/README.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_experiment import PROJECT_ROOT


OUT_DIR = PROJECT_ROOT / "artifacts" / "retraining_trigger"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def default_walk_forward_path():
    candidates = [
        PROJECT_ROOT / "artifacts" / "walk_forward" / "walk_forward_results.csv",
        PROJECT_ROOT
        / "results"
        / "2026-03-22_verified"
        / "walk_forward"
        / "walk_forward_results.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No walk-forward results file was found.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyse walk-forward drift and derive retraining triggers."
    )
    parser.add_argument(
        "--input-csv",
        default=str(default_walk_forward_path()),
        help="Walk-forward results CSV. Defaults to artifacts/ or the verified bundle.",
    )
    parser.add_argument(
        "--trailing-window",
        type=int,
        default=3,
        help="Number of previous windows used for the rolling baseline.",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=0.75,
        help="Absolute macro-F1 floor that triggers retraining.",
    )
    parser.add_argument(
        "--drop-margin",
        type=float,
        default=0.10,
        help="Absolute drop below the trailing mean that triggers retraining.",
    )
    return parser.parse_args()


def build_episode_rows(df):
    episodes = []
    current = None

    for row in df.itertuples(index=False):
        if row.trigger:
            if current is None:
                current = {
                    "start_boundary": row.boundary.strftime("%Y-%m-%d"),
                    "end_boundary": row.boundary.strftime("%Y-%m-%d"),
                    "start_test_start": row.test_start.strftime("%Y-%m-%d"),
                    "end_test_end": row.test_end.strftime("%Y-%m-%d"),
                    "trigger_reason": row.trigger_reason,
                    "min_macro_f1": row.api_macro_f1,
                    "n_windows": 1,
                }
            else:
                current["end_boundary"] = row.boundary.strftime("%Y-%m-%d")
                current["end_test_end"] = row.test_end.strftime("%Y-%m-%d")
                current["min_macro_f1"] = min(current["min_macro_f1"], row.api_macro_f1)
                current["n_windows"] += 1
        elif current is not None:
            episodes.append(current)
            current = None

    if current is not None:
        episodes.append(current)
    return pd.DataFrame(episodes)


def build_readme(path, input_csv, summary):
    lines = [
        "# Retraining-trigger analysis",
        "",
        f"Input walk-forward file: `{input_csv}`",
        f"Windows analysed: `{summary['n_windows']}`",
        f"Trigger windows: `{summary['n_trigger_windows']}`",
        f"Trigger episodes: `{summary['n_trigger_episodes']}`",
        "",
        "Files:",
        "",
        "- `retraining_trigger_windows.csv`",
        "- `retraining_trigger_episodes.csv`",
        "- `retraining_trigger_summary.json`",
        "- `retraining_trigger_plot.png`",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Walk-forward CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    df["boundary"] = pd.to_datetime(df["boundary"])
    df["test_start"] = pd.to_datetime(df["test_start"])
    df["test_end"] = pd.to_datetime(df["test_end"])
    df = df.sort_values("boundary").reset_index(drop=True)

    rows = []
    for idx, row in df.iterrows():
        trailing = df.iloc[max(0, idx - args.trailing_window) : idx]["api_macro_f1"]
        trailing_mean = float(trailing.mean()) if len(trailing) == args.trailing_window else np.nan
        drop_from_mean = (
            float(row["api_macro_f1"] - trailing_mean)
            if not np.isnan(trailing_mean)
            else np.nan
        )
        floor_trigger = bool(row["api_macro_f1"] < args.floor)
        drop_trigger = bool(
            not np.isnan(trailing_mean)
            and row["api_macro_f1"] < (trailing_mean - args.drop_margin)
        )
        trigger = floor_trigger or drop_trigger

        trigger_reason = []
        if floor_trigger:
            trigger_reason.append("floor")
        if drop_trigger:
            trigger_reason.append("trailing_mean_drop")

        rows.append(
            {
                "boundary": row["boundary"],
                "test_start": row["test_start"],
                "test_end": row["test_end"],
                "n_train": int(row["n_train"]),
                "n_test": int(row["n_test"]),
                "families_present": int(row["families_present"]),
                "api_macro_f1": float(row["api_macro_f1"]),
                "trailing_mean_macro_f1": trailing_mean,
                "drop_from_trailing_mean": drop_from_mean,
                "floor_trigger": floor_trigger,
                "drop_trigger": drop_trigger,
                "trigger": trigger,
                "trigger_reason": "+".join(trigger_reason) if trigger_reason else "",
            }
        )

    trigger_df = pd.DataFrame(rows)
    previous_trigger = trigger_df["trigger"].shift(fill_value=False)
    trigger_df["retrain_recommended"] = trigger_df["trigger"] & (~previous_trigger)

    episode_df = build_episode_rows(trigger_df)

    csv_df = trigger_df.copy()
    for column in ("boundary", "test_start", "test_end"):
        csv_df[column] = csv_df[column].dt.strftime("%Y-%m-%d")
    csv_df.to_csv(OUT_DIR / "retraining_trigger_windows.csv", index=False)
    episode_df.to_csv(OUT_DIR / "retraining_trigger_episodes.csv", index=False)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "input_csv": str(input_csv),
        "trailing_window": args.trailing_window,
        "floor": args.floor,
        "drop_margin": args.drop_margin,
        "n_windows": int(len(trigger_df)),
        "n_trigger_windows": int(trigger_df["trigger"].sum()),
        "n_trigger_episodes": int(len(episode_df)),
        "min_macro_f1": round(float(trigger_df["api_macro_f1"].min()), 4),
        "max_macro_f1": round(float(trigger_df["api_macro_f1"].max()), 4),
        "mean_macro_f1": round(float(trigger_df["api_macro_f1"].mean()), 4),
        "recommended_retrain_boundaries": csv_df.loc[
            csv_df["retrain_recommended"], "boundary"
        ].tolist(),
    }
    with open(OUT_DIR / "retraining_trigger_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(
        trigger_df["boundary"],
        trigger_df["api_macro_f1"],
        marker="o",
        label="Observed macro-F1",
    )
    ax.plot(
        trigger_df["boundary"],
        trigger_df["trailing_mean_macro_f1"],
        linestyle="--",
        color="tab:gray",
        label=f"Trailing mean ({args.trailing_window} windows)",
    )
    ax.axhline(args.floor, linestyle=":", color="tab:red", label="Trigger floor")

    trigger_points = trigger_df[trigger_df["trigger"]]
    if not trigger_points.empty:
        ax.scatter(
            trigger_points["boundary"],
            trigger_points["api_macro_f1"],
            color="tab:red",
            s=50,
            label="Trigger window",
            zorder=3,
        )

    retrain_points = trigger_df[trigger_df["retrain_recommended"]]
    for row in retrain_points.itertuples(index=False):
        ax.axvline(row.boundary, color="tab:orange", alpha=0.3)

    ax.set_title("Walk-forward drift profile with retraining triggers")
    ax.set_xlabel("Boundary date")
    ax.set_ylabel("macro-F1")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "retraining_trigger_plot.png", dpi=200)
    plt.close(fig)

    build_readme(OUT_DIR / "README.md", input_csv, summary)
    print(f"[ok] retraining-trigger outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()

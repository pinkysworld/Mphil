"""
13_archive_token_case_studies.py — Archive token-level explanation case studies.

Curates vocabulary-based local explanation cases from the archived explanation
folders into a compact bundle that is easier to cite in the thesis.

Outputs:
  - artifacts/explainability_case_studies/case_index.csv
  - artifacts/explainability_case_studies/token_case_studies.csv
  - artifacts/explainability_case_studies/casebook.md
  - artifacts/explainability_case_studies/manifest.json
  - artifacts/explainability_case_studies/README.md
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from run_experiment import PROJECT_ROOT


OUT_DIR = PROJECT_ROOT / "artifacts" / "explainability_case_studies"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_PREFIXES = ("api::", "artifact::")


def default_source_dir():
    candidates = [
        PROJECT_ROOT / "results" / "2026-03-22_verified" / "explainability",
        PROJECT_ROOT / "artifacts" / "explainability",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    raise FileNotFoundError("No explanation source directory was found.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Archive token-level explanation cases from vocabulary-based runs."
    )
    parser.add_argument(
        "--source-dir",
        default=str(default_source_dir()),
        help="Explanation directory to scan. Defaults to the verified results bundle.",
    )
    parser.add_argument(
        "--top-k-per-side",
        type=int,
        default=6,
        help="Number of supporting and opposing token rows to keep per case.",
    )
    return parser.parse_args()


def is_vocabulary_run(metadata):
    views = metadata.get("views", [])
    return any(view in {"api_tfidf", "art_tfidf"} for view in views)


def token_case_rows(contrib_df, top_k_per_side):
    token_mask = contrib_df["feature_name"].astype(str).str.startswith(TOKEN_PREFIXES)
    token_df = contrib_df[token_mask].copy()
    if token_df.empty:
        return token_df

    token_df["rank"] = token_df["rank"].astype(int)
    rows = []
    for (run_name, case_label, side), group in token_df.groupby(
        ["run_name", "case_label", "side"], sort=False
    ):
        rows.append(group.sort_values("rank").head(top_k_per_side))
    return pd.concat(rows, ignore_index=True) if rows else token_df.iloc[0:0]


def build_casebook(path, case_index_df, token_df, source_dir):
    lines = [
        "# Token-level case studies",
        "",
        f"Source directory: `{source_dir}`",
        "",
    ]

    if case_index_df.empty:
        lines.append("No vocabulary-based case studies were found.")
    else:
        for case in case_index_df.itertuples(index=False):
            lines.extend(
                [
                    f"## {case.run_name} / {case.case_label}",
                    "",
                    f"- SHA256: `{case.sha256}`",
                    f"- Date: `{case.date}`",
                    f"- True family: `{case.true_family}`",
                    f"- Predicted family: `{case.predicted_family}`",
                    f"- Correct: `{case.correct}`",
                    f"- Confidence: `{case.confidence}`",
                    f"- Margin: `{case.margin}`",
                    "",
                    "Supporting tokens:",
                ]
            )
            supporting = token_df[
                (token_df["run_name"] == case.run_name)
                & (token_df["case_label"] == case.case_label)
                & (token_df["side"] == "supporting")
            ]
            if supporting.empty:
                lines.append("- none")
            else:
                for row in supporting.itertuples(index=False):
                    lines.append(
                        f"- `{row.feature_name}` ({row.feature_group}, contribution={row.contribution:.4f})"
                    )

            lines.append("")
            lines.append("Opposing tokens:")
            opposing = token_df[
                (token_df["run_name"] == case.run_name)
                & (token_df["case_label"] == case.case_label)
                & (token_df["side"] == "opposing")
            ]
            if opposing.empty:
                lines.append("- none")
            else:
                for row in opposing.itertuples(index=False):
                    lines.append(
                        f"- `{row.feature_name}` ({row.feature_group}, contribution={row.contribution:.4f})"
                    )
            lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_readme(path, source_dir, manifest):
    lines = [
        "# Archived token-level case studies",
        "",
        f"Source directory: `{source_dir}`",
        f"Runs included: `{manifest['runs_included']}`",
        f"Cases exported: `{manifest['cases_exported']}`",
        "",
        "Files:",
        "",
        "- `case_index.csv`",
        "- `token_case_studies.csv`",
        "- `casebook.md`",
        "- `manifest.json`",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source explanation directory not found: {source_dir}")

    case_index_rows = []
    token_rows = []
    runs_included = []

    for run_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
        metadata_path = run_dir / "metadata.json"
        overview_path = run_dir / "case_overview.csv"
        contrib_path = run_dir / "case_feature_contributions.csv"
        if not (metadata_path.exists() and overview_path.exists() and contrib_path.exists()):
            continue

        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not is_vocabulary_run(metadata):
            continue

        overview_df = pd.read_csv(overview_path)
        overview_df.insert(0, "run_name", run_dir.name)
        overview_df["view_request"] = metadata.get("view_request", "")
        overview_df["split"] = metadata.get("split", "")
        overview_df["model"] = metadata.get("model", "")
        case_index_rows.append(overview_df)

        contrib_df = pd.read_csv(contrib_path)
        contrib_df.insert(0, "run_name", run_dir.name)
        token_subset = token_case_rows(contrib_df, args.top_k_per_side)
        if not token_subset.empty:
            token_rows.append(token_subset)
        runs_included.append(run_dir.name)

    case_index_df = (
        pd.concat(case_index_rows, ignore_index=True)
        if case_index_rows
        else pd.DataFrame(
            columns=[
                "run_name",
                "case_label",
                "sha256",
                "date",
                "true_family",
                "predicted_family",
                "correct",
                "confidence",
                "margin",
                "view_request",
                "split",
                "model",
            ]
        )
    )
    token_df = (
        pd.concat(token_rows, ignore_index=True)
        if token_rows
        else pd.DataFrame(
            columns=[
                "run_name",
                "case_label",
                "sha256",
                "true_family",
                "predicted_family",
                "side",
                "rank",
                "feature_name",
                "feature_group",
                "feature_value",
                "contribution",
            ]
        )
    )

    case_index_df.to_csv(OUT_DIR / "case_index.csv", index=False)
    token_df.to_csv(OUT_DIR / "token_case_studies.csv", index=False)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "source_dir": str(source_dir),
        "runs_included": len(sorted(set(runs_included))),
        "cases_exported": int(len(case_index_df)),
        "token_rows_exported": int(len(token_df)),
    }
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    build_casebook(OUT_DIR / "casebook.md", case_index_df, token_df, source_dir)
    build_readme(OUT_DIR / "README.md", source_dir, manifest)
    print(f"[ok] token-level case studies written to {OUT_DIR}")


if __name__ == "__main__":
    main()

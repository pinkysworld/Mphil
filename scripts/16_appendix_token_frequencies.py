"""Regenerate the descriptive token-frequency tables used in Appendix D."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from feature_extraction import (
    extract_artifact_tokens,
    extractor_policy,
    filter_family_names,
)
from reproducibility import file_record, sha256_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "metadata.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "reproducibility" / "appendix_d"
TOKEN_PATTERN = r"(?u)\b\w\w+\b"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def top_rows(counters: dict, view: str, top_n: int) -> list:
    rows = []
    for family in sorted(counters):
        ordered = sorted(counters[family].items(), key=lambda item: (-item[1], item[0]))
        for rank, (token, report_count) in enumerate(ordered[:top_n], start=1):
            rows.append(
                {
                    "view": view,
                    "family": family,
                    "rank": rank,
                    "token": token,
                    "report_count": report_count,
                }
            )
    return rows


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(args.metadata)
    metadata = metadata[metadata["has_report"]].reset_index(drop=True)

    artifact_analyzer = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 1),
    ).build_analyzer()

    api_entry_prevalence = defaultdict(Counter)
    artifact_component_prevalence = defaultdict(Counter)

    for position, sample in enumerate(metadata.itertuples(index=False), start=1):
        try:
            with Path(sample.report_path).open(
                "r", encoding="utf-8", errors="replace"
            ) as handle:
                report = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError):
            report = {}

        apis = report.get("behavior", {}).get("summary", {}).get("resolved_apis", [])
        if isinstance(apis, list):
            entries = {
                filtered
                for api in apis
                if isinstance(api, str)
                for filtered in [filter_family_names(api.lower().strip())]
                if filtered
            }
            api_entry_prevalence[sample.family].update(entries)

        artifact_document = extract_artifact_tokens(report)
        artifact_component_prevalence[sample.family].update(
            set(artifact_analyzer(artifact_document))
        )

        if position % 5000 == 0:
            print(f"Processed {position} / {len(metadata)} reports")

    api_rows = top_rows(api_entry_prevalence, "api_entry", args.top_n)
    artifact_rows = top_rows(
        artifact_component_prevalence, "artifact_model_component", args.top_n
    )
    api_path = args.output_dir / "api_entry_report_prevalence.csv"
    artifact_path = args.output_dir / "artifact_component_report_prevalence.csv"
    pd.DataFrame(api_rows).to_csv(api_path, index=False, lineterminator="\n")
    pd.DataFrame(artifact_rows).to_csv(artifact_path, index=False, lineterminator="\n")

    analyzer_parameters = {
        "analyzer": "word",
        "lowercase": True,
        "token_pattern": TOKEN_PATTERN,
        "ngram_range": [1, 1],
        "counting_unit": "number_of_family_reports_containing_token",
    }
    manifest = {
        "schema_version": "1.0",
        "extractor": extractor_policy(),
        "api_table_definition": (
            "Exact leakage-filtered resolved_apis entries; each entry contributes "
            "at most once per report."
        ),
        "artifact_table_definition": (
            "Unigram components produced by the explicit model word analyzer from "
            "the shared artifact extractor; each component contributes at most once "
            "per report."
        ),
        "artifact_analyzer": analyzer_parameters,
        "artifact_analyzer_sha256": sha256_json(analyzer_parameters),
        "top_n": args.top_n,
        "inputs": {"metadata": file_record(args.metadata, PROJECT_ROOT)},
        "outputs": [
            file_record(api_path, PROJECT_ROOT),
            file_record(artifact_path, PROJECT_ROOT),
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"Appendix D tables written to {args.output_dir}")


if __name__ == "__main__":
    main()

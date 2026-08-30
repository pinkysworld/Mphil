"""Audit provenance of malware-family targets used by the thesis.

The public Avast-CTU metadata provides family/type/detection-date information
but not the independent provenance fields requested by the supervisor. This
script records what is and is not identifiable and optionally accepts a
supplemental per-SHA provenance table later.

Expected optional supplemental columns:
  sha256, label_source, label_date, engine_version, cross_vendor_agreement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from run_experiment import PROJECT_ROOT

PROCESSED = PROJECT_ROOT / "data" / "processed" / "metadata.parquet"
OUT_DIR = PROJECT_ROOT / "artifacts" / "supervisor_revision"
EXPECTED_SUPPLEMENTAL = [
    "label_source",
    "label_date",
    "engine_version",
    "cross_vendor_agreement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--supplemental-provenance",
        type=Path,
        help=(
            "Optional CSV keyed by sha256 with independently sourced provenance "
            "fields. No external values are inferred when this file is absent."
        ),
    )
    return parser.parse_args()


def normalize_sha(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip()


def coverage(series: pd.Series) -> dict[str, float | int]:
    present = series.notna() & series.astype(str).str.strip().ne("")
    return {
        "n_present": int(present.sum()),
        "n_total": int(len(series)),
        "fraction_present": float(present.mean()) if len(series) else 0.0,
    }


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_parquet(PROCESSED).copy()
    required = {"sha256", "family", "date"}
    missing = required.difference(meta.columns)
    if missing:
        raise ValueError(f"metadata.parquet missing required columns: {sorted(missing)}")

    meta["sha256"] = normalize_sha(meta["sha256"])
    meta["family"] = meta["family"].astype(str).str.lower().str.strip()
    meta["date"] = pd.to_datetime(meta["date"], errors="coerce")

    source_columns = [str(c) for c in meta.columns]
    availability = {
        "family_target": {
            "status": "available",
            "column": "family",
            "provenance": (
                "Avast-CTU public_labels.csv; official dataset documentation "
                "states classification is from Avast backend records."
            ),
        },
        "detection_date": {
            "status": "available",
            "column": "date",
            "coverage": coverage(meta["date"]),
        },
        "separate_labeling_date": {
            "status": "not_available_in_public_dataset",
        },
        "engine_version_used_for_labeling": {
            "status": "not_available_in_public_dataset",
        },
        "cross_vendor_agreement": {
            "status": "not_available_in_public_dataset",
        },
    }

    if "type" in meta.columns:
        availability["malware_type"] = {
            "status": "available",
            "column": "type",
            "note": (
                "This is metadata from the same public dataset and is not treated "
                "as an independent label source."
            ),
            "coverage": coverage(meta["type"]),
        }

    supplemental_summary: dict[str, object] = {
        "provided": False,
        "path": None,
        "coverage": {},
    }
    if args.supplemental_provenance:
        supplemental = pd.read_csv(args.supplemental_provenance)
        if "sha256" not in supplemental.columns:
            raise ValueError("Supplemental provenance CSV must contain sha256.")
        supplemental["sha256"] = normalize_sha(supplemental["sha256"])
        keep = ["sha256"] + [
            col for col in EXPECTED_SUPPLEMENTAL if col in supplemental.columns
        ]
        supplemental = supplemental[keep].drop_duplicates("sha256", keep="last")
        meta = meta.merge(supplemental, on="sha256", how="left", validate="one_to_one")

        supplemental_summary["provided"] = True
        supplemental_summary["path"] = str(args.supplemental_provenance)
        for column in EXPECTED_SUPPLEMENTAL:
            if column in meta.columns:
                supplemental_summary["coverage"][column] = coverage(meta[column])

        if "label_date" in meta.columns:
            availability["separate_labeling_date"] = {
                "status": "supplemental",
                "coverage": coverage(meta["label_date"]),
            }
        if "engine_version" in meta.columns:
            availability["engine_version_used_for_labeling"] = {
                "status": "supplemental",
                "coverage": coverage(meta["engine_version"]),
            }
        if "cross_vendor_agreement" in meta.columns:
            availability["cross_vendor_agreement"] = {
                "status": "supplemental",
                "coverage": coverage(meta["cross_vendor_agreement"]),
            }

    dated = meta.dropna(subset=["date"]).copy()
    dated["year"] = dated["date"].dt.year.astype(int)
    dated["month"] = dated["date"].dt.to_period("M").astype(str)

    yearly = (
        dated.groupby(["year", "family"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    yearly["year_total"] = yearly.groupby("year")["n"].transform("sum")
    yearly["family_share"] = yearly["n"] / yearly["year_total"]
    yearly.to_csv(OUT_DIR / "label_composition_by_year.csv", index=False)

    monthly = (
        dated.groupby(["month", "family"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    monthly["month_total"] = monthly.groupby("month")["n"].transform("sum")
    monthly["family_share"] = monthly["n"] / monthly["month_total"]
    monthly.to_csv(OUT_DIR / "label_composition_by_month.csv", index=False)

    family_dates = (
        dated.groupby("family", observed=True)["date"]
        .agg(["count", "min", "max"])
        .reset_index()
    )
    family_dates.to_csv(OUT_DIR / "label_family_date_coverage.csv", index=False)

    payload = {
        "schema_version": "1.0",
        "n_samples": int(len(meta)),
        "source_columns_observed": source_columns,
        "public_dataset_target_provenance": (
            "Family labels are supplied by the Avast-CTU public dataset and "
            "originate from Avast backend classification records according to "
            "the official dataset documentation."
        ),
        "field_availability": availability,
        "supplemental_provenance": supplemental_summary,
        "date_range": {
            "min": str(dated["date"].min()) if len(dated) else None,
            "max": str(dated["date"].max()) if len(dated) else None,
        },
        "families": sorted(meta["family"].dropna().unique().tolist()),
        "identifiability_boundary": (
            "Without an independently sourced label history, engine version, or "
            "cross-vendor agreement, this dataset alone cannot separate malware "
            "behaviour drift from vendor labelling-policy drift. Temporal label "
            "composition summaries are diagnostic only and must not be presented "
            "as proof of labelling-policy change."
        ),
        "labels_removed_from_target": False,
        "input_side_leakage_control_note": (
            "Family names/detection signals are removed from X where prohibited; "
            "the family label remains y for supervised classification."
        ),
    }
    (OUT_DIR / "label_provenance_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()

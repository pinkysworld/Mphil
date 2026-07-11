"""Build a self-checking reproducibility bundle for a dated result archive."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from reproducibility import (
    environment_manifest,
    file_record,
    sha256_file,
    write_deterministic_gzip,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "results" / "2026-07-11"
DEFAULT_METADATA = PROJECT_ROOT / "data" / "processed" / "metadata.parquet"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/reproducibility.",
    )
    parser.add_argument(
        "--skip-report-checksums",
        action="store_true",
        help="Skip JSON report content checksums for a faster partial bundle.",
    )
    return parser.parse_args()


def copy_if_present(
    source: Path, destination_dir: Path, destination_name: str | None = None
) -> Path | None:
    if not source.exists():
        return None
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / (destination_name or source.name)
    shutil.copy2(source, destination)
    return destination


def build_report_manifest(meta: pd.DataFrame, output_path: Path) -> dict:
    rows = []
    for position, sample in enumerate(meta.itertuples(index=False), start=1):
        report_path = Path(sample.report_path)
        if not report_path.exists():
            raise FileNotFoundError(f"Missing report for {sample.sha256}: {report_path}")
        rows.append(
            {
                "sha256": sample.sha256,
                "report_json_sha256": sha256_file(report_path),
                "report_bytes": report_path.stat().st_size,
            }
        )
        if position % 5000 == 0:
            print(f"  Checksummed {position} / {len(meta)} reports")

    payload = pd.DataFrame(rows).to_csv(index=False, lineterminator="\n").encode("utf-8")
    write_deterministic_gzip(output_path, payload)
    return file_record(output_path, PROJECT_ROOT)


def build_checksums(run_dir: Path, output_path: Path) -> int:
    records = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.resolve() == output_path.resolve():
            continue
        relative = path.relative_to(run_dir)
        records.append(f"{sha256_file(path)}  {relative.as_posix()}")
    output_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    return len(records)


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Result directory does not exist: {run_dir}")

    output_dir = (args.output_dir or run_dir / "reproducibility").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(args.metadata)
    meta = meta[meta["has_report"]].reset_index(drop=True)

    copied = []
    sources = [
        (PROJECT_ROOT / ".python-version", "python-version.txt"),
        (PROJECT_ROOT / "requirements.txt", None),
        (PROJECT_ROOT / "requirements-lock.txt", None),
        (
            PROJECT_ROOT / "docs" / "reproducibility.md",
            "EXACT_REPRODUCIBILITY.md",
        ),
        (PROJECT_ROOT / "artifacts" / "reproducibility" / "splits" / "sample_catalog.csv.gz", None),
        (PROJECT_ROOT / "artifacts" / "reproducibility" / "splits" / "split_assignments.csv.gz", None),
        (PROJECT_ROOT / "artifacts" / "reproducibility" / "splits" / "split_manifest.json", None),
        (PROJECT_ROOT / "artifacts" / "reproducibility" / "feature_manifest.json", None),
        (PROJECT_ROOT / "artifacts" / "walk_forward" / "walk_forward_manifest.json", None),
        (PROJECT_ROOT / "artifacts" / "walk_forward" / "walk_forward_predictions.csv.gz", None),
        (PROJECT_ROOT / "artifacts" / "walk_forward" / "walk_forward_sample_catalog.csv.gz", None),
        (PROJECT_ROOT / "artifacts" / "reproducibility" / "appendix_d" / "api_entry_report_prevalence.csv", None),
        (PROJECT_ROOT / "artifacts" / "reproducibility" / "appendix_d" / "artifact_component_report_prevalence.csv", None),
        (PROJECT_ROOT / "artifacts" / "reproducibility" / "appendix_d" / "manifest.json", "appendix_d_manifest.json"),
    ]
    for source, destination_name in sources:
        destination = copy_if_present(source, output_dir, destination_name)
        if destination:
            copied.append(file_record(destination, PROJECT_ROOT))

    environment_path = output_dir / "environment.json"
    environment = environment_manifest(PROJECT_ROOT)
    write_json(environment_path, environment)

    inputs = {"metadata": file_record(args.metadata, PROJECT_ROOT)}
    labels_path = PROJECT_ROOT / "data" / "raw" / "public_labels.csv"
    if labels_path.exists():
        inputs["labels"] = file_record(labels_path, PROJECT_ROOT)

    archive_candidates = [
        PROJECT_ROOT / "data" / "raw" / "avast_ctu_reduced.zip",
        PROJECT_ROOT / "data" / "raw" / "public_small_reports.zip",
    ]
    for archive in archive_candidates:
        if archive.exists():
            inputs["reduced_report_archive"] = file_record(archive, PROJECT_ROOT)
            break

    report_manifest = None
    if not args.skip_report_checksums:
        report_manifest_path = output_dir / "report_content_checksums.csv.gz"
        if report_manifest_path.exists():
            print(f"  Reusing report-content manifest: {report_manifest_path}")
            report_manifest = file_record(report_manifest_path, PROJECT_ROOT)
        else:
            report_manifest = build_report_manifest(meta, report_manifest_path)

    locked_files = []
    for relative in [
        ".python-version",
        "requirements.txt",
        "requirements-lock.txt",
        "scripts/03_build_splits.py",
        "scripts/04_extract_features.py",
        "scripts/06_walk_forward.py",
        "scripts/15_build_reproducibility_bundle.py",
        "scripts/16_appendix_token_frequencies.py",
        "scripts/feature_extraction.py",
        "scripts/reproducibility.py",
    ]:
        path = PROJECT_ROOT / relative
        if path.exists():
            locked_files.append(file_record(path, PROJECT_ROOT))

    bundle_manifest = {
        "schema_version": "1.0",
        "scope": (
            "Exact input, split, feature, vocabulary or hashing configuration, "
            "environment, prediction, and result checksums for the defended run."
        ),
        "environment": file_record(environment_path, PROJECT_ROOT),
        "inputs": inputs,
        "report_content_manifest": report_manifest,
        "copied_reproducibility_artifacts": copied,
        "locked_code_and_environment_files": locked_files,
    }
    bundle_manifest_path = output_dir / "bundle_manifest.json"
    write_json(bundle_manifest_path, bundle_manifest)

    checksums_path = output_dir / "SHA256SUMS"
    count = build_checksums(run_dir, checksums_path)
    print(f"Reproducibility bundle: {output_dir}")
    print(f"Checksummed result files: {count}")
    print(f"Checksum list: {checksums_path}")


if __name__ == "__main__":
    main()

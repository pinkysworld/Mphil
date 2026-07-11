"""
04_extract_features.py — Feature extraction for all views

Extracts features for each view from the reduced reports and
caches them as sparse matrices / DataFrames.

Views:
  A: API tokens (resolved_apis → TF-IDF / hashing n-grams)
  B: Artifact tokens (files, registry, mutexes, commands, services)
  C: Behavioral counts (low-dimensional numeric)
  D: Static PE numeric

Output (per split):
  - data/cache/api_tfidf_{split}.npz + fitted vectorizer
  - data/cache/api_hash_{split}.npz
  - data/cache/art_tfidf_{split}.npz + fitted vectorizer
  - data/cache/counts_scaled_{split}.npy + fitted scaler
  - data/cache/pe_scaled_{split}.npy + fitted scaler

Shared outputs:
  - data/cache/labels.parquet
  - data/cache/counts_raw.parquet
  - data/cache/pe_raw.parquet
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import joblib

from feature_extraction import (
    extract_api_tokens,
    extract_artifact_tokens,
    extract_counts,
    extract_pe_features,
    extractor_policy,
)
from reproducibility import (
    environment_manifest,
    file_record,
    hashing_vectorizer_manifest,
    ordered_rows_sha256,
    vocabulary_manifest,
    write_json,
)

# ── Configuration ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RAW_FEATURE_DIR = CACHE_DIR / "raw_feature_chunks"
RAW_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
REPRO_DIR = PROJECT_ROOT / "artifacts" / "reproducibility"
DEFAULT_WORD_TOKEN_PATTERN = r"(?u)\b\w\w+\b"
RAW_FEATURE_CHUNK_SIZE = 5000


def _stable_json(value: dict) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _load_or_extract_raw_features(meta: pd.DataFrame):
    """Extract raw views in restart-safe chunks without changing feature logic."""
    api_docs = []
    art_docs = []
    count_rows = []
    pe_rows = []

    for start in range(0, len(meta), RAW_FEATURE_CHUNK_SIZE):
        stop = min(start + RAW_FEATURE_CHUNK_SIZE, len(meta))
        chunk_path = RAW_FEATURE_DIR / f"features_{start:05d}_{stop:05d}.parquet"
        expected_sha256 = meta.iloc[start:stop]["sha256"].astype(str).tolist()

        if chunk_path.exists():
            chunk = pd.read_parquet(chunk_path)
            required = {"sha256", "api_doc", "art_doc", "counts_json", "pe_json"}
            if required.issubset(chunk.columns) and (
                chunk["sha256"].astype(str).tolist() == expected_sha256
            ):
                print(f"  Reusing checkpoint {start + 1}-{stop} / {len(meta)}")
            else:
                raise RuntimeError(
                    f"Raw-feature checkpoint does not match metadata: {chunk_path}"
                )
        else:
            rows = []
            for position in range(start, stop):
                row = meta.iloc[position]
                report_path = Path(row["report_path"])
                try:
                    with open(
                        report_path, "r", encoding="utf-8", errors="replace"
                    ) as handle:
                        report = json.load(handle)
                except (json.JSONDecodeError, OSError):
                    report = {}

                rows.append(
                    {
                        "sha256": str(row["sha256"]),
                        "api_doc": extract_api_tokens(report),
                        "art_doc": extract_artifact_tokens(report),
                        "counts_json": _stable_json(extract_counts(report)),
                        "pe_json": _stable_json(extract_pe_features(report)),
                    }
                )

                processed = position + 1
                if processed % 5000 == 0 or processed == len(meta):
                    print(f"  Processed {processed} / {len(meta)}")

            chunk = pd.DataFrame(rows)
            temporary_path = chunk_path.with_suffix(".parquet.tmp")
            chunk.to_parquet(temporary_path, index=False)
            temporary_path.replace(chunk_path)

        api_docs.extend(chunk["api_doc"].astype(str).tolist())
        art_docs.extend(chunk["art_doc"].astype(str).tolist())
        count_rows.extend(json.loads(value) for value in chunk["counts_json"])
        pe_rows.extend(json.loads(value) for value in chunk["pe_json"])

    return api_docs, art_docs, count_rows, pe_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Create restart-safe raw-feature checkpoints and shared caches only.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=[
            "global_chronological",
            "per_family_chronological",
            "random_stratified",
        ],
        help="Vectorise only the named split; repeat to select multiple splits.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MPhil Feature Extraction")
    print("=" * 60)

    metadata_path = PROCESSED_DIR / "metadata.parquet"
    meta = pd.read_parquet(metadata_path)
    meta = meta[meta["has_report"]].reset_index(drop=True)
    n = len(meta)
    print(f"Extracting features from {n} reports...")

    api_docs, art_docs, count_rows, pe_rows = _load_or_extract_raw_features(meta)

    print("\nExtraction complete. Vectorising and caching...")

    feature_manifest = {
        "schema_version": "1.0",
        "extractor": extractor_policy(),
        "environment": environment_manifest(PROJECT_ROOT),
        "inputs": {
            "metadata": file_record(metadata_path, PROJECT_ROOT),
            "ordered_samples_sha256": ordered_rows_sha256(
                f"{i},{row.sha256},{row.family},{row.date}"
                for i, row in meta.iterrows()
            ),
            "api_documents_sha256": ordered_rows_sha256(api_docs),
            "artifact_documents_sha256": ordered_rows_sha256(art_docs),
        },
        "vectorizer_note": (
            "TF-IDF and hashing use scikit-learn's explicit default word "
            "token pattern. Punctuation-delimited API and artifact strings are "
            "therefore represented as alphanumeric components and component "
            "bigrams, not as indivisible dotted API-name tokens."
        ),
        "raw_feature_checkpoints": [
            file_record(path, PROJECT_ROOT)
            for path in sorted(RAW_FEATURE_DIR.glob("features_*.parquet"))
        ],
        "splits": {},
    }

    # Save shared cache inputs.
    labels_path = CACHE_DIR / "labels.parquet"
    counts_path = CACHE_DIR / "counts_raw.parquet"
    pe_path = CACHE_DIR / "pe_raw.parquet"
    meta[["sha256", "family", "date"]].to_parquet(labels_path)

    # Counts
    counts_df = pd.DataFrame(count_rows).fillna(0)
    counts_df.to_parquet(counts_path)

    # PE
    pe_df = pd.DataFrame(pe_rows)
    pe_df.to_parquet(pe_path)
    feature_manifest["common_cache_files"] = [
        file_record(path, PROJECT_ROOT) for path in [labels_path, counts_path, pe_path]
    ]

    manifest_path = REPRO_DIR / "feature_manifest.json"
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        same_documents = (
            previous_manifest.get("inputs", {}).get("api_documents_sha256")
            == feature_manifest["inputs"]["api_documents_sha256"]
            and previous_manifest.get("inputs", {}).get("artifact_documents_sha256")
            == feature_manifest["inputs"]["artifact_documents_sha256"]
        )
        if same_documents:
            feature_manifest["splits"] = previous_manifest.get("splits", {})

    if args.extract_only:
        write_json(manifest_path, feature_manifest)
        print(f"\n✓ Raw features and shared caches ready in {CACHE_DIR}")
        print(f"  Reproducibility manifest: {manifest_path}")
        return

    # Now vectorise per split
    split_names = args.split or [
        "global_chronological",
        "per_family_chronological",
        "random_stratified",
    ]
    for split_name in split_names:
        split_path = SPLITS_DIR / f"{split_name}.json"
        if not split_path.exists():
            print(f"  Skip {split_name} (split file not found)")
            continue

        with open(split_path) as f:
            split = json.load(f)
        train_idx = split["train"]
        print(f"\n  Vectorising for split: {split_name} "
              f"(train={len(train_idx)})")

        # API TF-IDF (vocabulary model)
        tfidf = TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            token_pattern=DEFAULT_WORD_TOKEN_PATTERN,
            ngram_range=(1, 2),
            max_features=50000,
            min_df=2,
            sublinear_tf=True,
            norm="l2",
        )
        train_api = [api_docs[i] for i in train_idx]
        tfidf.fit(train_api)
        all_api_tfidf = tfidf.transform(api_docs)
        sp.save_npz(CACHE_DIR / f"api_tfidf_{split_name}.npz", all_api_tfidf)
        joblib.dump(tfidf, CACHE_DIR / f"api_tfidf_vectorizer_{split_name}.pkl")

        # API Hashing (scalable)
        hasher = HashingVectorizer(
            analyzer="word",
            lowercase=True,
            token_pattern=DEFAULT_WORD_TOKEN_PATTERN,
            n_features=262144,
            ngram_range=(1, 2),
            alternate_sign=False,
            norm="l2",
        )
        all_api_hash = hasher.transform(api_docs)
        sp.save_npz(CACHE_DIR / f"api_hash_{split_name}.npz", all_api_hash)

        # Artifact TF-IDF
        art_tfidf = TfidfVectorizer(
            analyzer="word",
            lowercase=True,
            token_pattern=DEFAULT_WORD_TOKEN_PATTERN,
            ngram_range=(1, 1),
            max_features=50000,
            min_df=2,
            sublinear_tf=True,
            norm="l2",
        )
        train_art = [art_docs[i] for i in train_idx]
        art_tfidf.fit(train_art)
        all_art_tfidf = art_tfidf.transform(art_docs)
        sp.save_npz(CACHE_DIR / f"art_tfidf_{split_name}.npz", all_art_tfidf)
        joblib.dump(art_tfidf, CACHE_DIR / f"art_tfidf_vectorizer_{split_name}.pkl")

        # Counts (scaled on train)
        scaler_counts = MaxAbsScaler()
        scaler_counts.fit(counts_df.iloc[train_idx])
        counts_scaled = scaler_counts.transform(counts_df)
        np.save(CACHE_DIR / f"counts_scaled_{split_name}.npy", counts_scaled)
        joblib.dump(scaler_counts, CACHE_DIR / f"counts_scaler_{split_name}.pkl")

        # PE (scaled on train, imputed)
        pe_filled = pe_df.fillna(pe_df.iloc[train_idx].median())
        scaler_pe = StandardScaler()
        scaler_pe.fit(pe_filled.iloc[train_idx])
        pe_scaled = scaler_pe.transform(pe_filled)
        np.save(CACHE_DIR / f"pe_scaled_{split_name}.npy", pe_scaled)
        joblib.dump(scaler_pe, CACHE_DIR / f"pe_scaler_{split_name}.pkl")

        cache_paths = [
            CACHE_DIR / f"api_tfidf_{split_name}.npz",
            CACHE_DIR / f"api_tfidf_vectorizer_{split_name}.pkl",
            CACHE_DIR / f"api_hash_{split_name}.npz",
            CACHE_DIR / f"art_tfidf_{split_name}.npz",
            CACHE_DIR / f"art_tfidf_vectorizer_{split_name}.pkl",
            CACHE_DIR / f"counts_scaled_{split_name}.npy",
            CACHE_DIR / f"counts_scaler_{split_name}.pkl",
            CACHE_DIR / f"pe_scaled_{split_name}.npy",
            CACHE_DIR / f"pe_scaler_{split_name}.pkl",
        ]
        feature_manifest["splits"][split_name] = {
            "split_file": file_record(split_path, PROJECT_ROOT),
            "api_tfidf": vocabulary_manifest(tfidf),
            "api_hash": hashing_vectorizer_manifest(hasher),
            "artifact_tfidf": vocabulary_manifest(art_tfidf),
            "cache_files": [file_record(path, PROJECT_ROOT) for path in cache_paths],
        }

    write_json(manifest_path, feature_manifest)

    print(f"\n✓ All features cached in {CACHE_DIR}")
    print(f"  Reproducibility manifest: {manifest_path}")


if __name__ == "__main__":
    main()

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
  - data/cache/api_tfidf_{split}.npz + vocab
  - data/cache/api_hash_{split}.npz
  - data/cache/artifacts_tfidf_{split}.npz + vocab
  - data/cache/artifacts_hash_{split}.npz
  - data/cache/counts_{split}.parquet
  - data/cache/pe_{split}.parquet
  - data/cache/labels_{split}.parquet
"""

import json
import re
import os
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import joblib

# ── Configuration ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Family names for leakage filtering (Rule 1)
FAMILY_NAMES = {
    "emotet", "swisyn", "qakbot", "trickbot", "lokibot",
    "njrat", "zeus", "ursnif", "adload", "harhar"
}

# Volatile identifier patterns (Rule 3)
GUID_PATTERN = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
    re.IGNORECASE
)
HEX_PATTERN = re.compile(r'[0-9a-f]{8,}', re.IGNORECASE)
NUMERIC_PATTERN = re.compile(r'^\d+$')


# ── Normalisation helpers ──────────────────────────────────

def normalise_path(path_str: str) -> str:
    """Normalise a file path for tokenisation."""
    s = path_str.lower().replace("\\", "/")
    # Replace user profile paths
    s = re.sub(r'c:/users/[^/]+', 'c:/users/<user>', s)
    # Replace temp dirs
    s = re.sub(r'/temp/[^/]+', '/temp/<tmp>', s)
    # Replace GUIDs
    s = GUID_PATTERN.sub('<guid>', s)
    # Replace long hex
    s = HEX_PATTERN.sub('<hex>', s)
    return s


def normalise_registry(key_str: str) -> str:
    """Normalise a registry key path (keep prefix to depth 3)."""
    s = key_str.lower().replace("\\", "/")
    s = GUID_PATTERN.sub('<guid>', s)
    s = re.sub(r'/s-1-[0-9-]+', '/<sid>', s)
    parts = s.split("/")
    return "/".join(parts[:4])  # hive + 3 levels


def normalise_mutex(mutex_str: str) -> str:
    """Normalise a mutex name."""
    s = mutex_str.lower()
    s = GUID_PATTERN.sub('<guid>', s)
    s = HEX_PATTERN.sub('<hex>', s)
    return s


def filter_family_names(token: str) -> str:
    """Remove family-name segments (Rule 1)."""
    for fam in FAMILY_NAMES:
        token = token.replace(fam, "")
    return token.strip()


# ── View extractors ────────────────────────────────────────

def extract_api_tokens(report: dict) -> str:
    """Extract resolved API call names as a space-separated string."""
    behavior = report.get("behavior", {})
    summary = behavior.get("summary", {})
    apis = summary.get("resolved_apis", [])
    if not isinstance(apis, list):
        return ""
    # Normalise: lowercase, filter family names
    tokens = []
    for api in apis:
        if isinstance(api, str):
            t = filter_family_names(api.lower().strip())
            if t:
                tokens.append(t)
    return " ".join(tokens)


def extract_artifact_tokens(report: dict) -> str:
    """Extract normalised artifact tokens from files, registry, mutexes, etc."""
    behavior = report.get("behavior", {})
    summary = behavior.get("summary", {})
    tokens = []

    # File artifacts
    for field in ["files", "read_files", "write_files", "delete_files"]:
        for item in summary.get(field, []):
            if isinstance(item, str):
                normed = normalise_path(item)
                normed = filter_family_names(normed)
                # Extract basename and extension
                parts = normed.rsplit("/", 1)
                basename = parts[-1] if len(parts) > 1 else normed
                ext = ""
                if "." in basename:
                    ext = basename.rsplit(".", 1)[-1]
                    tokens.append(f"FILE_EXT:{ext}")
                tokens.append(f"FILE:{basename[:80]}")

    # Registry artifacts
    for field in ["keys", "read_keys", "write_keys", "delete_keys"]:
        for item in summary.get(field, []):
            if isinstance(item, str):
                normed = normalise_registry(item)
                normed = filter_family_names(normed)
                tokens.append(f"REG:{normed}")

    # Mutex artifacts
    for item in summary.get("mutexes", []):
        if isinstance(item, str):
            normed = normalise_mutex(item)
            normed = filter_family_names(normed)
            tokens.append(f"MUTEX:{normed}")

    # Command artifacts
    for item in summary.get("executed_commands", []):
        if isinstance(item, str):
            exe = item.lower().split()[0] if item.split() else item.lower()
            exe = filter_family_names(exe)
            tokens.append(f"CMD:{exe[:80]}")

    # Service artifacts
    for field in ["started_services", "created_services"]:
        for item in summary.get(field, []):
            if isinstance(item, str):
                normed = filter_family_names(item.lower().strip())
                tokens.append(f"SVC:{normed[:80]}")

    return " ".join(tokens[:400])  # cap to 400 tokens per sample


def extract_counts(report: dict) -> dict:
    """Extract low-dimensional behavioral count features."""
    behavior = report.get("behavior", {})
    summary = behavior.get("summary", {})
    counts = {}
    for field in ["files", "read_files", "write_files", "delete_files",
                   "keys", "read_keys", "write_keys", "delete_keys",
                   "mutexes", "executed_commands", "resolved_apis",
                   "started_services", "created_services"]:
        val = summary.get(field, [])
        counts[f"count_{field}"] = len(val) if isinstance(val, list) else 0
    return counts


def extract_pe_features(report: dict) -> dict:
    """Extract numeric PE features from static.pe."""
    static = report.get("static", {})
    pe = static.get("pe", {})
    if not isinstance(pe, dict):
        return {}

    features = {}
    # Direct numeric fields
    for key in ["timestamp", "imagebase", "entrypoint"]:
        val = pe.get(key)
        if isinstance(val, (int, float)):
            features[f"pe_{key}"] = float(val)

    # Sections
    sections = pe.get("sections", [])
    if isinstance(sections, list):
        features["pe_n_sections"] = len(sections)
        entropies = []
        sizes = []
        for sec in sections:
            if isinstance(sec, dict):
                e = sec.get("entropy")
                s = sec.get("size_of_data") or sec.get("virtual_size")
                if isinstance(e, (int, float)):
                    entropies.append(float(e))
                if isinstance(s, (int, float)):
                    sizes.append(float(s))
        if entropies:
            features["pe_mean_entropy"] = np.mean(entropies)
            features["pe_max_entropy"] = max(entropies)
            features["pe_std_entropy"] = np.std(entropies) if len(entropies) > 1 else 0
        if sizes:
            features["pe_total_size"] = sum(sizes)

    # Imports
    imports = pe.get("imports", [])
    if isinstance(imports, list):
        features["pe_n_import_dlls"] = len(imports)
        total_funcs = 0
        for imp in imports:
            if isinstance(imp, dict):
                funcs = imp.get("imports", [])
                if isinstance(funcs, list):
                    total_funcs += len(funcs)
        features["pe_n_import_funcs"] = total_funcs

    # Exports
    exports = pe.get("exports", [])
    features["pe_n_exports"] = len(exports) if isinstance(exports, list) else 0

    return features


def main():
    print("=" * 60)
    print("MPhil Feature Extraction")
    print("=" * 60)

    meta = pd.read_parquet(PROCESSED_DIR / "metadata.parquet")
    meta = meta[meta["has_report"] == True].reset_index(drop=True)
    n = len(meta)
    print(f"Extracting features from {n} reports...")

    # Collect raw features for all samples
    api_docs = []
    art_docs = []
    count_rows = []
    pe_rows = []

    for i, row in meta.iterrows():
        report_path = Path(row["report_path"])
        try:
            with open(report_path, "r", encoding="utf-8", errors="replace") as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError):
            report = {}

        api_docs.append(extract_api_tokens(report))
        art_docs.append(extract_artifact_tokens(report))
        count_rows.append(extract_counts(report))
        pe_rows.append(extract_pe_features(report))

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1} / {n}")

    print(f"\nExtraction complete. Vectorising and caching...")

    # Save labels
    meta[["sha256", "family", "date"]].to_parquet(CACHE_DIR / "labels.parquet")

    # Counts
    counts_df = pd.DataFrame(count_rows).fillna(0)
    counts_df.to_parquet(CACHE_DIR / "counts_raw.parquet")

    # PE
    pe_df = pd.DataFrame(pe_rows)
    pe_df.to_parquet(CACHE_DIR / "pe_raw.parquet")

    # Now vectorise per split
    for split_name in ["global_chronological", "per_family_chronological",
                       "random_stratified"]:
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
            ngram_range=(1, 2), max_features=50000, min_df=2,
            sublinear_tf=True
        )
        train_api = [api_docs[i] for i in train_idx]
        tfidf.fit(train_api)
        all_api_tfidf = tfidf.transform(api_docs)
        sp.save_npz(CACHE_DIR / f"api_tfidf_{split_name}.npz", all_api_tfidf)
        joblib.dump(tfidf, CACHE_DIR / f"api_tfidf_vectorizer_{split_name}.pkl")

        # API Hashing (scalable)
        hasher = HashingVectorizer(
            n_features=262144, ngram_range=(1, 2), alternate_sign=False
        )
        all_api_hash = hasher.transform(api_docs)
        sp.save_npz(CACHE_DIR / f"api_hash_{split_name}.npz", all_api_hash)

        # Artifact TF-IDF
        art_tfidf = TfidfVectorizer(
            ngram_range=(1, 1), max_features=50000, min_df=2,
            sublinear_tf=True
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

    print(f"\n✓ All features cached in {CACHE_DIR}")


if __name__ == "__main__":
    main()

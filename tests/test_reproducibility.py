"""Regression checks for deterministic manifests and vocabulary hashes."""

import gzip
import sys
import tempfile
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproducibility import (  # noqa: E402
    sha256_file,
    vocabulary_manifest,
    write_deterministic_gzip,
)


def main():
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "first.csv.gz"
        second = Path(directory) / "second.csv.gz"
        payload = b"role,sha256\ntrain,abc\ntest,def\n"
        write_deterministic_gzip(first, payload)
        write_deterministic_gzip(second, payload)
        assert first.read_bytes() == second.read_bytes()
        assert gzip.decompress(first.read_bytes()) == payload
        assert sha256_file(first) == sha256_file(second)

    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 2),
        min_df=1,
    ).fit(["kernel32 dll createfilew", "advapi32 dll regopenkeyexw"])
    vectorizer.vocabulary_ = {
        token: np.int64(index) for token, index in vectorizer.vocabulary_.items()
    }
    first_manifest = vocabulary_manifest(vectorizer)
    second_manifest = vocabulary_manifest(vectorizer)
    assert first_manifest == second_manifest
    assert first_manifest["n_features"] == len(vectorizer.vocabulary_)
    assert len(first_manifest["vocabulary_sha256"]) == 64
    assert len(first_manifest["idf_float64_le_sha256"]) == 64

    print("ok: deterministic reproducibility helpers passed")


if __name__ == "__main__":
    main()

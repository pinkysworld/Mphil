"""Deterministic manifest and checksum helpers for thesis experiments."""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "1.0"


def canonical_json_bytes(value) -> bytes:
    """Serialise a JSON-compatible value in a stable form."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_rows_sha256(rows) -> str:
    """Hash an ordered iterable of strings with unambiguous line framing."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_deterministic_gzip(path: Path, payload: bytes) -> None:
    """Write gzip bytes with a fixed timestamp for reproducible checksums."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))


def vocabulary_manifest(vectorizer) -> dict:
    """Hash vocabulary membership, indices, and fitted IDF values."""
    # scikit-learn may expose indices as NumPy integer scalars. Canonical JSON
    # must use portable built-in integers so the hash is stable and serialisable.
    entries = sorted(
        (str(token), int(index))
        for token, index in vectorizer.vocabulary_.items()
    )
    vocabulary_sha256 = sha256_json(entries)
    idf = np.asarray(vectorizer.idf_, dtype="<f8")
    return {
        "kind": "learned_vocabulary",
        "n_features": len(entries),
        "vocabulary_sha256": vocabulary_sha256,
        "idf_float64_le_sha256": sha256_bytes(idf.tobytes(order="C")),
    }


def hashing_vectorizer_manifest(vectorizer) -> dict:
    """Record and hash the configuration of a stateless hashing vectorizer."""
    keys = [
        "analyzer",
        "alternate_sign",
        "binary",
        "decode_error",
        "encoding",
        "input",
        "lowercase",
        "n_features",
        "ngram_range",
        "norm",
        "stop_words",
        "strip_accents",
        "token_pattern",
    ]
    parameters = {}
    all_parameters = vectorizer.get_params(deep=False)
    for key in keys:
        value = all_parameters.get(key)
        if isinstance(value, tuple):
            value = list(value)
        parameters[key] = value
    return {
        "kind": "stateless_hashing",
        "vocabulary_sha256": None,
        "parameters": parameters,
        "parameters_sha256": sha256_json(parameters),
    }


def git_state(project_root: Path) -> dict:
    """Capture the revision and whether uncommitted files were present."""
    project_root = Path(project_root)

    def run_git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": run_git("rev-parse", "HEAD"),
            "branch": run_git("branch", "--show-current") or None,
            "dirty": bool(run_git("status", "--porcelain")),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def environment_manifest(project_root: Path, package_names=None) -> dict:
    """Capture the interpreter, platform, packages, and git revision."""
    if package_names is None:
        package_names = [
            "joblib",
            "lightgbm",
            "matplotlib",
            "numpy",
            "pandas",
            "pyarrow",
            "scikit-learn",
            "scipy",
        ]

    packages = {}
    for name in sorted(package_names):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    return {
        "schema_version": SCHEMA_VERSION,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": packages,
        "git": git_state(project_root),
    }


def file_record(path: Path, project_root: Path) -> dict:
    path = Path(path)
    project_root = Path(project_root)
    try:
        display_path = str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        display_path = str(path.resolve())
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }

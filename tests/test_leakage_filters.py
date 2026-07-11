"""Regression tests for exact-segment leakage filtering.

Run from the repository root with:
    python tests/test_leakage_filters.py
"""

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    features = load_script("04_extract_features.py")
    audit = load_script("02_leakage_audit.py")
    walk_forward = load_script("06_walk_forward.py")
    import feature_extraction as shared

    assert features.extract_api_tokens is shared.extract_api_tokens
    assert walk_forward.extract_api_tokens is shared.extract_api_tokens

    assert shared.filter_family_names("kernel32.dll.createremotethread") == "kernel32.dll.createremotethread"
    assert shared.filter_family_names("global\\trickbot") == "global\\"
    assert shared.filter_family_names("c:/tmp/emotet.py") == "c:/tmp/.py"

    benign_api_report = {
        "behavior": {
            "summary": {
                "resolved_apis": [
                    "kernel32.dll.CreateRemoteThread",
                    "advapi32.dll.CryptVerifySignatureW",
                ]
            }
        }
    }
    assert audit.scan_for_leakage("sha", benign_api_report) == []

    leaked_report = {"behavior": {"summary": {"mutexes": ["Global\\TrickBot"]}}}
    hits = audit.scan_for_leakage("sha", leaked_report)
    assert len(hits) == 1
    assert hits[0]["matched_term"] == "trickbot"

    defended_report = {
        "behavior": {
            "summary": {
                "resolved_apis": [
                    "kernel32.dll.CreateRemoteThread",
                    "Global\\TrickBot",
                ]
            }
        }
    }
    expected = "kernel32.dll.createremotethread global\\"
    assert shared.extract_api_tokens(defended_report) == expected
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "report.json"
        path.write_text(json.dumps(defended_report), encoding="utf-8")
        assert walk_forward.load_api_document(path) == expected

    hasher = walk_forward.build_hasher(131072)
    assert hasher.token_pattern == r"(?u)\b\w\w+\b"
    assert hasher.n_features == 131072
    assert hasher.ngram_range == (1, 2)
    assert hasher.alternate_sign is False

    print("ok: shared exact-segment leakage filtering regression tests passed")


if __name__ == "__main__":
    main()

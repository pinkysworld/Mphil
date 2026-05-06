"""Regression tests for exact-segment leakage filtering.

Run from the repository root with:
    python tests/test_leakage_filters.py
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    features = load_script("04_extract_features.py")
    audit = load_script("02_leakage_audit.py")

    assert features.filter_family_names("kernel32.dll.createremotethread") == "kernel32.dll.createremotethread"
    assert features.filter_family_names("global\\trickbot") == "global\\"
    assert features.filter_family_names("c:/tmp/emotet.py") == "c:/tmp/.py"

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

    print("ok: exact-segment leakage filtering regression tests passed")


if __name__ == "__main__":
    main()

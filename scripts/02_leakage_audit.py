"""
02_leakage_audit.py — String-level leakage audit

Scans all reduced reports for family-name segments and
detection-indicator terms in allowed fields.

Output:
  - artifacts/leakage_audit/audit_results.json
  - artifacts/leakage_audit/flagged_samples.csv
"""

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import pandas as pd

# ── Configuration ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
AUDIT_DIR = PROJECT_ROOT / "artifacts" / "leakage_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# Family names to scan for (case-insensitive)
FAMILY_NAMES = [
    "emotet", "swisyn", "qakbot", "trickbot", "lokibot",
    "njrat", "zeus", "ursnif", "adload", "harhar"
]

# Detection-indicator terms. These are checked as exact alphanumeric
# segments, not substrings, to avoid false positives such as
# CryptVerifySignatureW -> "signature".
DETECTION_TERMS = [
    "yara", "signature", "detection", "avclass",
    "malware", "classification", "sig"
]

SEGMENT_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
FAMILY_NAME_SET = set(FAMILY_NAMES)
DETECTION_TERM_SET = set(DETECTION_TERMS)

# Allowed behavioral string fields to scan
ALLOWED_STRING_FIELDS = [
    # behavior.summary lists
    "files", "read_files", "write_files", "delete_files",
    "keys", "read_keys", "write_keys", "delete_keys",
    "mutexes", "executed_commands", "resolved_apis",
    "started_services", "created_services",
    # Network-related
    "domains", "hosts", "urls", "ips",
]


def extract_strings_from_report(report: dict) -> list:
    """Extract all string values from allowed fields in the reduced report."""
    strings = []
    # Navigate to behavior.summary
    behavior = report.get("behavior", {})
    summary = behavior.get("summary", {})
    for field in ALLOWED_STRING_FIELDS:
        values = summary.get(field, [])
        if isinstance(values, list):
            for v in values:
                if isinstance(v, str):
                    strings.append((field, v))
                elif isinstance(v, dict):
                    # Some fields may be dicts (e.g., network)
                    for sv in v.values():
                        if isinstance(sv, str):
                            strings.append((field, sv))
    return strings


def alphanumeric_segments(value: str) -> set:
    """Return exact alphanumeric segments for leakage matching.

    Exact segment matching prevents accidental hits where a family name or
    detection word is embedded inside an unrelated API/function name. For
    example, CreateRemoteThread contains the letters "emotet" across an
    internal boundary, and CryptVerifySignatureW contains "signature"; neither
    should be treated as a confirmed leakage hit.
    """
    return {segment.lower() for segment in SEGMENT_RE.findall(value or "")}


def scan_for_leakage(sha256: str, report: dict) -> list:
    """Scan a single report for exact-segment leakage indicators."""
    hits = []
    strings = extract_strings_from_report(report)

    for field, value in strings:
        segments = alphanumeric_segments(value)

        for fam in sorted(FAMILY_NAME_SET & segments):
            hits.append({
                "sha256": sha256,
                "field": field,
                "value": value[:200],  # truncate for storage
                "match_type": "family_name",
                "matched_term": fam,
            })

        for term in sorted(DETECTION_TERM_SET & segments):
            hits.append({
                "sha256": sha256,
                "field": field,
                "value": value[:200],
                "match_type": "detection_indicator",
                "matched_term": term,
            })
    return hits


def main():
    print("=" * 60)
    print("MPhil Leakage Audit")
    print("=" * 60)

    # Load metadata
    meta = pd.read_parquet(PROCESSED_DIR / "metadata.parquet")
    meta = meta[meta["has_report"] == True]
    print(f"Scanning {len(meta)} reports...")

    all_hits = []
    flagged_shas = set()
    n_scanned = 0

    for _, row in meta.iterrows():
        report_path = Path(row["report_path"])
        if not report_path.exists():
            continue

        try:
            with open(report_path, "r", encoding="utf-8", errors="replace") as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        hits = scan_for_leakage(row["sha256"], report)
        if hits:
            all_hits.extend(hits)
            flagged_shas.add(row["sha256"])

        n_scanned += 1
        if n_scanned % 5000 == 0:
            print(f"  Scanned {n_scanned} / {len(meta)} "
                  f"({len(all_hits)} hits so far)")

    print(f"\nScan complete: {n_scanned} reports scanned.")
    print(f"Total hits: {len(all_hits)}")
    print(f"Unique flagged samples: {len(flagged_shas)}")

    # Summarise by match type and term
    summary = defaultdict(int)
    family_hits = defaultdict(int)
    for h in all_hits:
        key = f"{h['match_type']}:{h['matched_term']}"
        summary[key] += 1
        if h["match_type"] == "family_name":
            family_hits[h["matched_term"]] += 1

    print("\nFamily-name hits by family:")
    for fam, count in sorted(family_hits.items(), key=lambda x: -x[1]):
        print(f"  {fam}: {count}")

    # Save results
    audit_result = {
        "timestamp": datetime.now().isoformat(),
        "n_scanned": n_scanned,
        "n_total_hits": len(all_hits),
        "n_flagged_samples": len(flagged_shas),
        "hit_rate": len(flagged_shas) / n_scanned if n_scanned > 0 else 0,
        "summary_by_term": dict(summary),
        "family_name_hits": dict(family_hits),
        "detection_indicator_hits": {
            k.split(":")[1]: v for k, v in summary.items()
            if k.startswith("detection_indicator")
        },
        "families_scanned": FAMILY_NAMES,
        "detection_terms_scanned": DETECTION_TERMS,
        "fields_scanned": ALLOWED_STRING_FIELDS,
        "matching_policy": "exact_alphanumeric_segments",
    }

    with open(AUDIT_DIR / "audit_results.json", "w") as f:
        json.dump(audit_result, f, indent=2)

    if all_hits:
        hits_df = pd.DataFrame(all_hits)
        hits_df.to_csv(AUDIT_DIR / "flagged_samples.csv", index=False)
        print(f"\nFlagged samples saved to: {AUDIT_DIR / 'flagged_samples.csv'}")

    print(f"Audit results saved to: {AUDIT_DIR / 'audit_results.json'}")

    # Verdict
    if len(flagged_shas) <= 15:
        print(f"\n✓ PASS: {len(flagged_shas)} flagged (< 0.05% of dataset).")
        print("  These will be handled by token normalisation Rule 1.")
    else:
        print(f"\n⚠ REVIEW: {len(flagged_shas)} flagged samples. Inspect manually.")


if __name__ == "__main__":
    main()

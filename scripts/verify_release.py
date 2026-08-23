"""Verify the public MPhil research supplement without regenerating results."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OBSOLETE_RESULT_PREFIX = "results/" + "2026-03-21/"
PRIVATE_SNAPSHOT_TOKEN = b"results/" + b"2026-03-22_verified"
ABSOLUTE_HOME_TOKEN = b"/" + b"Users/"
HISTORICAL_PATH_RECORDS = {
    "results/2026-07-11/bootstrap/README.md",
    "results/2026-07-11/bootstrap/bootstrap_ci_results.json",
}
MANIFESTS = (
    PROJECT_ROOT / "artifacts" / "decision_support" / "manifest.json",
    PROJECT_ROOT / "artifacts" / "calibrated_selective_policy" / "manifest.json",
    PROJECT_ROOT / "artifacts" / "explainability_case_studies" / "manifest.json",
)


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id"):
            self.ids.add(str(attributes["id"]))
        for key in ("href", "src"):
            if attributes.get(key):
                self.links.append(str(attributes[key]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def verify_manifest(path: Path, errors: list[str]) -> int:
    if not path.is_file():
        errors.append(f"missing manifest: {path.relative_to(PROJECT_ROOT)}")
        return 0
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checked = 0
    entries = []
    for key in ("sources", "inputs", "outputs"):
        value = manifest.get(key, [])
        if isinstance(value, list):
            entries.extend(value)
    if not entries:
        errors.append(f"manifest has no hash entries: {path.relative_to(PROJECT_ROOT)}")
    for entry in entries:
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not relative or not expected:
            errors.append(f"incomplete hash entry in {path.relative_to(PROJECT_ROOT)}")
            continue
        target = PROJECT_ROOT / relative
        if not target.is_file():
            errors.append(f"manifest target missing: {relative}")
            continue
        actual = sha256_file(target)
        checked += 1
        if actual != expected:
            errors.append(f"manifest hash mismatch: {relative}")
    return checked


def verify_dated_bundle(errors: list[str]) -> int:
    result_root = PROJECT_ROOT / "results" / "2026-07-11"
    checksum_path = result_root / "reproducibility" / "SHA256SUMS"
    if not checksum_path.is_file():
        errors.append("missing defended SHA256SUMS")
        return 0
    checked = 0
    for number, raw_line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            errors.append(f"malformed SHA256SUMS line {number}")
            continue
        relative = relative.lstrip("*")
        target = result_root / relative
        if not target.is_file():
            errors.append(f"dated-bundle target missing: {relative}")
            continue
        actual = sha256_file(target)
        checked += 1
        if actual != expected:
            errors.append(f"dated-bundle hash mismatch: {relative}")
    return checked


def verify_site(errors: list[str]) -> int:
    pages: dict[Path, LinkCollector] = {}
    for path in sorted((PROJECT_ROOT / "site").glob("*.html")):
        parser = LinkCollector()
        parser.feed(path.read_text(encoding="utf-8"))
        pages[path.resolve()] = parser

    checked = 0
    for source, parser in pages.items():
        for raw_link in parser.links:
            parts = urlsplit(raw_link)
            if parts.scheme or parts.netloc or raw_link.startswith(("mailto:", "tel:")):
                continue
            target = (source.parent / (parts.path or source.name)).resolve()
            checked += 1
            if not target.is_file():
                errors.append(
                    f"broken site link in {source.name}: {raw_link}"
                )
                continue
            if parts.fragment and target.suffix.lower() == ".html":
                target_parser = pages.get(target)
                if target_parser is None:
                    target_parser = LinkCollector()
                    target_parser.feed(target.read_text(encoding="utf-8"))
                if parts.fragment not in target_parser.ids:
                    errors.append(
                        f"missing site fragment in {source.name}: {raw_link}"
                    )
    return checked


def audit_inventory(files: list[str], errors: list[str], warnings: list[str]) -> None:
    result_directories = {
        path.split("/", 2)[1]
        for path in files
        if path.startswith("results/") and path.count("/") >= 2
    }
    if result_directories != {"2026-07-11"}:
        errors.append(
            "unexpected dated result directories: "
            + ", ".join(sorted(result_directories))
        )

    forbidden_prefixes = ("experiments/", "notebooks/", OBSOLETE_RESULT_PREFIX)
    forbidden_suffixes = (".pyc", ".pyo", ".DS_Store", "~")
    for relative in files:
        if relative.startswith(forbidden_prefixes):
            errors.append(f"obsolete tracked path: {relative}")
        if relative.endswith(forbidden_suffixes):
            errors.append(f"temporary tracked file: {relative}")

        target = PROJECT_ROOT / relative
        if not target.is_file():
            errors.append(f"tracked file missing from worktree: {relative}")
            continue
        data = target.read_bytes()
        if ABSOLUTE_HOME_TOKEN in data:
            errors.append(f"absolute workstation path in tracked file: {relative}")
        if PRIVATE_SNAPSHOT_TOKEN in data:
            errors.append(f"removed private snapshot referenced by: {relative}")
        if OBSOLETE_RESULT_PREFIX.encode() in data and relative not in HISTORICAL_PATH_RECORDS:
            errors.append(f"obsolete result path referenced by: {relative}")

    for relative in sorted(HISTORICAL_PATH_RECORDS):
        if relative in files:
            warnings.append(f"historical execution path retained in immutable metadata: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Fail when the Git worktree or index contains tracked changes.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    files = tracked_files()
    audit_inventory(files, errors, warnings)

    checked_hashes = verify_dated_bundle(errors)
    for manifest in MANIFESTS:
        checked_hashes += verify_manifest(manifest, errors)
    checked_site_links = verify_site(errors)

    if args.require_clean:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            errors.append("Git worktree or index is not clean")

    report = {
        "status": "ok" if not errors else "failed",
        "tracked_files": len(files),
        "verified_hashes": checked_hashes,
        "verified_site_links": checked_site_links,
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

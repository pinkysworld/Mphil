"""Shared, leakage-controlled feature extraction helpers.

All experiment entry points that read free-text fields from reduced CAPEv2
reports import their extractors from this module. Keeping one implementation
prevents the walk-forward pipeline from drifting away from the defended
feature policy.
"""

from __future__ import annotations

import re

import numpy as np


EXTRACTOR_SCHEMA_VERSION = "2026-07-11.1"
LEAKAGE_MATCHING_POLICY = "exact_alphanumeric_segments"

FAMILY_NAMES = frozenset(
    {
        "emotet",
        "swisyn",
        "qakbot",
        "trickbot",
        "lokibot",
        "njrat",
        "zeus",
        "ursnif",
        "adload",
        "harhar",
    }
)

SEGMENT_SPLIT_RE = re.compile(r"([A-Za-z0-9]+)")
GUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
    re.IGNORECASE,
)
HEX_PATTERN = re.compile(r"[0-9a-f]{8,}", re.IGNORECASE)


def extractor_policy() -> dict:
    """Return the canonical policy recorded in reproducibility manifests."""
    return {
        "schema_version": EXTRACTOR_SCHEMA_VERSION,
        "leakage_matching_policy": LEAKAGE_MATCHING_POLICY,
        "family_names": sorted(FAMILY_NAMES),
        "api_source": "behavior.summary.resolved_apis",
        "artifact_token_cap": 400,
    }


def normalise_path(path_str: str) -> str:
    """Normalise a file path before model-document serialisation."""
    value = path_str.lower().replace("\\", "/")
    value = re.sub(r"c:/users/[^/]+", "c:/users/<user>", value)
    value = re.sub(r"/temp/[^/]+", "/temp/<tmp>", value)
    value = GUID_PATTERN.sub("<guid>", value)
    return HEX_PATTERN.sub("<hex>", value)


def normalise_registry(key_str: str) -> str:
    """Normalise a registry path and retain the hive plus three levels."""
    value = key_str.lower().replace("\\", "/")
    value = GUID_PATTERN.sub("<guid>", value)
    value = re.sub(r"/s-1-[0-9-]+", "/<sid>", value)
    return "/".join(value.split("/")[:4])


def normalise_mutex(mutex_str: str) -> str:
    """Normalise volatile identifiers in a mutex string."""
    value = GUID_PATTERN.sub("<guid>", mutex_str.lower())
    return HEX_PATTERN.sub("<hex>", value)


def filter_family_names(token: str) -> str:
    """Remove exact family-name segments without mutating other strings."""
    parts = SEGMENT_SPLIT_RE.split(token or "")
    return "".join(
        part for part in parts if part.lower() not in FAMILY_NAMES
    ).strip()


def extract_api_tokens(report: dict) -> str:
    """Serialise leakage-filtered resolved API entries in stored list order."""
    summary = report.get("behavior", {}).get("summary", {})
    apis = summary.get("resolved_apis", [])
    if not isinstance(apis, list):
        return ""

    tokens = []
    for api in apis:
        if isinstance(api, str):
            token = filter_family_names(api.lower().strip())
            if token:
                tokens.append(token)
    return " ".join(tokens)


def extract_artifact_tokens(report: dict) -> str:
    """Serialise leakage-filtered file, registry, mutex, command, and service data."""
    summary = report.get("behavior", {}).get("summary", {})
    tokens = []

    for field in ["files", "read_files", "write_files", "delete_files"]:
        values = summary.get(field, [])
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, str):
                continue
            normalised = filter_family_names(normalise_path(item))
            basename = normalised.rsplit("/", 1)[-1]
            if "." in basename:
                tokens.append(f"FILE_EXT:{basename.rsplit('.', 1)[-1]}")
            tokens.append(f"FILE:{basename[:80]}")

    for field in ["keys", "read_keys", "write_keys", "delete_keys"]:
        values = summary.get(field, [])
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, str):
                normalised = filter_family_names(normalise_registry(item))
                tokens.append(f"REG:{normalised}")

    mutexes = summary.get("mutexes", [])
    if isinstance(mutexes, list):
        for item in mutexes:
            if isinstance(item, str):
                normalised = filter_family_names(normalise_mutex(item))
                tokens.append(f"MUTEX:{normalised}")

    commands = summary.get("executed_commands", [])
    if isinstance(commands, list):
        for item in commands:
            if isinstance(item, str):
                executable = item.lower().split()[0] if item.split() else item.lower()
                tokens.append(f"CMD:{filter_family_names(executable)[:80]}")

    for field in ["started_services", "created_services"]:
        values = summary.get(field, [])
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, str):
                normalised = filter_family_names(item.lower().strip())
                tokens.append(f"SVC:{normalised[:80]}")

    return " ".join(tokens[:400])


def extract_counts(report: dict) -> dict:
    """Extract the defended low-dimensional behavioural count view."""
    summary = report.get("behavior", {}).get("summary", {})
    fields = [
        "files",
        "read_files",
        "write_files",
        "delete_files",
        "keys",
        "read_keys",
        "write_keys",
        "delete_keys",
        "mutexes",
        "executed_commands",
        "resolved_apis",
        "started_services",
        "created_services",
    ]
    return {
        f"count_{field}": len(summary.get(field, []))
        if isinstance(summary.get(field, []), list)
        else 0
        for field in fields
    }


def extract_pe_features(report: dict) -> dict:
    """Extract the defended compact numeric PE view."""
    pe = report.get("static", {}).get("pe", {})
    if not isinstance(pe, dict):
        return {}

    features = {}
    for key in ["timestamp", "imagebase", "entrypoint"]:
        value = pe.get(key)
        if isinstance(value, (int, float)):
            features[f"pe_{key}"] = float(value)

    sections = pe.get("sections", [])
    if isinstance(sections, list):
        features["pe_n_sections"] = len(sections)
        entropies = []
        sizes = []
        for section in sections:
            if not isinstance(section, dict):
                continue
            entropy = section.get("entropy")
            size = section.get("size_of_data") or section.get("virtual_size")
            if isinstance(entropy, (int, float)):
                entropies.append(float(entropy))
            if isinstance(size, (int, float)):
                sizes.append(float(size))
        if entropies:
            features["pe_mean_entropy"] = float(np.mean(entropies))
            features["pe_max_entropy"] = max(entropies)
            features["pe_std_entropy"] = (
                float(np.std(entropies)) if len(entropies) > 1 else 0.0
            )
        if sizes:
            features["pe_total_size"] = sum(sizes)

    imports = pe.get("imports", [])
    if isinstance(imports, list):
        features["pe_n_import_dlls"] = len(imports)
        features["pe_n_import_funcs"] = sum(
            len(item.get("imports", []))
            for item in imports
            if isinstance(item, dict) and isinstance(item.get("imports", []), list)
        )

    exports = pe.get("exports", [])
    features["pe_n_exports"] = len(exports) if isinstance(exports, list) else 0
    return features

"""Reproducibility metadata and atomic file helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(master_seed: int, *parts: object) -> int:
    """Derive a stable NumPy-compatible seed from labelled analysis parts."""

    digest = hashlib.blake2b(digest_size=8, person=b"te-gcmi")
    digest.update(str(int(master_seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") % (2**32 - 1)


def stable_json_hash(payload: dict[str, Any]) -> str:
    """Hash a JSON-compatible mapping with deterministic key ordering."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def package_version(name: str) -> str | None:
    """Return an installed package version, or ``None`` if unavailable."""

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_metadata(config: dict[str, Any], manifest_path: str | Path) -> dict[str, Any]:
    """Build the audit record written beside every analysis output."""

    manifest_path = Path(manifest_path).resolve()
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "te-population-gcmi": package_version("te-population-gcmi") or "source-tree",
        },
        "config_sha256": stable_json_hash(config),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "independent_unit": "network_id",
        "nested_units": ["trajectory_id", "segment_id", "time_sample"],
    }


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so interrupted runs do not leave valid-looking files."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

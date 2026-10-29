from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import pickle
import uuid
from pathlib import Path
from typing import Any, Iterable


def _temporary_sibling(path: Path) -> Path:
    """Return a process-unique sibling so parallel jobs never share a temp file."""
    return path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary_sibling(path)
    try:
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_npz(path: Path, compressed: bool = False, **arrays: Any) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary_sibling(path)
    try:
        with temp.open("wb") as handle:
            if compressed:
                np.savez_compressed(handle, **arrays)
            else:
                np.savez(handle, **arrays)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_csv_gz(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary_sibling(path)
    try:
        with gzip.open(temp, "wt", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_pickle(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = _temporary_sibling(path)
    try:
        with temp.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def manifest_is_complete(
    directory: Path,
    run_hash: str | Iterable[str],
    required: Iterable[str],
) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    accepted_hashes = (
        {run_hash}
        if isinstance(run_hash, str)
        else {str(value) for value in run_hash}
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("run_sha256") not in accepted_hashes
    ):
        return False
    recorded = {
        os.path.normpath(str(row.get("path", ""))): row.get("sha256")
        for row in manifest.get("outputs", [])
    }
    for relative in required:
        normalized = os.path.normpath(str(relative))
        path = directory / normalized
        if not path.exists() or normalized not in recorded:
            return False
        if recorded[normalized] != sha256_file(path):
            return False
    # A completed downstream result is reusable only while every recorded input
    # still exists and has the exact content used to create that result. This is
    # essential for checkpointed connection matrices: changing an upstream
    # checkpoint must invalidate baseline/familiarization outputs.
    for row in manifest.get("inputs", []):
        recorded_path = Path(str(row.get("path", "")))
        path = recorded_path if recorded_path.is_absolute() else directory / recorded_path
        if not path.is_file() or row.get("sha256") != sha256_file(path):
            return False
    return True


def complete_manifest(
    directory: Path,
    run_hash: str,
    stage: str,
    identity: dict[str, Any],
    seeds: dict[str, int],
    outputs: Iterable[Path],
    inputs: Iterable[Path] = (),
    notes: dict[str, Any] | None = None,
) -> None:
    output_rows = []
    for path in outputs:
        output_rows.append({
            "path": os.path.relpath(path, directory),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    input_rows = []
    for path in inputs:
        input_rows.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    atomic_json(directory / "manifest.json", {
        "status": "complete",
        "stage": stage,
        "run_sha256": run_hash,
        "identity": identity,
        "seeds": seeds,
        "inputs": input_rows,
        "outputs": output_rows,
        "notes": notes or {},
    })

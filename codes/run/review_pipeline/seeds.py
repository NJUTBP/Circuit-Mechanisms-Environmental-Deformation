from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def derive_seed(master_seed: int, *labels: object) -> int:
    """Derive an order-independent uint32 seed from a stable semantic key."""
    message = "\x1f".join([str(int(master_seed)), *(str(label) for label in labels)])
    digest = hashlib.blake2s(message.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


@dataclass(frozen=True)
class SeedRecord:
    seed: int
    purpose: str
    network_id: int | None = None
    environment: str | None = None
    trajectory_id: int | None = None
    session: int | None = None
    branch: int | None = None
    detail: str | None = None

    def as_row(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "purpose": self.purpose,
            "network_id": "" if self.network_id is None else self.network_id,
            "environment": self.environment or "",
            "trajectory_id": "" if self.trajectory_id is None else self.trajectory_id,
            "session": "" if self.session is None else self.session,
            "branch": "" if self.branch is None else self.branch,
            "detail": self.detail or "",
        }


class SeedBook:
    def __init__(self, master_seed: int):
        self.master_seed = int(master_seed)

    def seed(self, purpose: str, **parts: object) -> int:
        labels: list[object] = [purpose]
        for key in sorted(parts):
            labels.extend((key, parts[key]))
        return derive_seed(self.master_seed, *labels)

    def record(self, purpose: str, **parts: object) -> SeedRecord:
        return SeedRecord(seed=self.seed(purpose, **parts), purpose=purpose, **parts)


SEED_COLUMNS = [
    "seed", "purpose", "network_id", "environment", "trajectory_id",
    "session", "branch", "detail",
]


def write_seed_ledger(path: Path, records: Iterable[SeedRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    rows = sorted(
        (record.as_row() for record in records),
        key=lambda row: tuple(str(row[key]) for key in SEED_COLUMNS[1:]),
    )
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


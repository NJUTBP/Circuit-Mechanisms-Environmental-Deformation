"""Run one flattened LSF task from a prepared sensitivity manifest."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from review_pipeline.config import PipelineConfig
from review_pipeline.plan import task_for_index
from review_pipeline.stages import (
    audit_initialization,
    run_evaluation_group,
    run_familiarization,
    run_initialization,
)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ready":
        raise SystemExit(f"Sensitivity manifest is not ready: {path}")
    return payload


def rows_for(payload: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    try:
        rows = payload["tasks"][stage]
    except KeyError as exc:
        raise SystemExit(f"Stage {stage!r} is not present in this sweep") from exc
    return list(rows)


def run_row(stage: str, row: dict[str, Any]) -> None:
    config = PipelineConfig(row["config"])
    if stage == "audit_init":
        audit_initialization(config)
        return
    local_index = int(row["local_task_index"])
    task = task_for_index(config, stage, local_index)
    if task != row.get("task"):
        raise RuntimeError(
            f"Prepared task mapping drifted for {row['condition_id']} {stage} "
            f"index {local_index}: {row.get('task')} != {task}"
        )
    if stage == "init":
        run_initialization(config, int(task["network_id"]))
    elif stage in {"baseline", "initial_exposure"}:
        run_evaluation_group(
            config,
            stage=stage,
            network_id=int(task["network_id"]),
            environment=str(task["environment"]),
        )
    elif stage == "familiarization":
        run_familiarization(
            config,
            network_id=int(task["network_id"]),
            environment=str(task["environment"]),
            variant=str(task["variant"]),
        )
    else:
        raise SystemExit(f"Unsupported stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    count = subparsers.add_parser("count")
    count.add_argument("--stage", required=True)
    parallel = subparsers.add_parser("parallel")
    parallel.add_argument("--stage", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--stage", required=True)
    run.add_argument("--task-index", type=int, required=True)
    subparsers.add_parser("describe")
    args = parser.parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    payload = load_manifest(manifest_path)
    if args.command == "describe":
        print(json.dumps({
            "sweep_name": payload["sweep_name"],
            "job_cores": payload["job_cores"],
            "trajectory_ids": payload["trajectory_ids"],
            "retain_raw_activity": payload["retain_raw_activity"],
            "task_counts": payload["task_counts"],
            "max_parallel": payload["max_parallel"],
        }, indent=2))
        return 0
    rows = rows_for(payload, args.stage)
    if args.command == "count":
        print(len(rows))
        return 0
    if args.command == "parallel":
        print(min(len(rows), int(payload["max_parallel"][args.stage])))
        return 0
    if args.task_index < 0 or args.task_index >= len(rows):
        raise SystemExit(
            f"--task-index must be in 0..{len(rows) - 1}; got {args.task_index}"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    row = rows[args.task_index]
    logging.getLogger(__name__).info(
        "sensitivity_task sweep=%s stage=%s condition=%s local_index=%s",
        payload["sweep_name"], args.stage, row["condition_id"],
        row["local_task_index"],
    )
    run_row(args.stage, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

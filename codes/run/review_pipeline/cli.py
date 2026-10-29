from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .aggregate import aggregate_outputs
from .config import PipelineConfig
from .exposure_summary import create_post_exposure_summary
from .plan import STAGES, count_tasks, prepare_experiment, print_plan, task_for_index
from .stages import (
    audit_initialization,
    run_evaluation_group,
    run_familiarization,
    run_initialization,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent-network rescaling revision pipeline"
    )
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/config.model.json"))
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("plan")
    subparsers.add_parser("prepare")
    subparsers.add_parser("audit-init")
    count = subparsers.add_parser("count")
    count.add_argument("--stage", choices=STAGES, required=True)
    slurm_limit = subparsers.add_parser("slurm-limit")
    slurm_limit.add_argument("--stage", choices=STAGES, required=True)
    parallel_limit = subparsers.add_parser("parallel-limit")
    parallel_limit.add_argument("--stage", choices=STAGES, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--stage", choices=STAGES, required=True)
    run.add_argument("--task-index", type=int, required=True)
    subparsers.add_parser("aggregate")
    subparsers.add_parser("post-exposure-summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = PipelineConfig(args.config)
    if args.command == "validate":
        print(json.dumps({
            "status": "ok",
            "config_sha256": config.config_hash,
            "pipeline_sha256": config.pipeline_hash,
            "runtime_sha256": config.runtime_hash,
            "input_sha256": config.input_hash,
            "run_sha256": config.run_hash,
            "missing_dependencies": config.missing_dependencies,
            "data_root": str(config.data_root),
            "output_root": str(config.output_root),
        }, ensure_ascii=False))
        return 0
    if args.command == "plan":
        print_plan(config)
        return 0
    if args.command == "prepare":
        print(json.dumps(prepare_experiment(config), sort_keys=True))
        return 0
    if args.command == "audit-init":
        report = audit_initialization(config)
        print(json.dumps({
            "status": report["status"],
            "n_networks": report["n_networks"],
            "all_connection_files_unique": report["all_connection_files_unique"],
            "report": str(config.output_root / "initialization_audit.json"),
        }, ensure_ascii=False))
        return 0
    if args.command == "count":
        print(count_tasks(config, args.stage))
        return 0
    if args.command in {"slurm-limit", "parallel-limit"}:
        print(int(config.data["slurm"][f"max_parallel_{args.stage}"]))
        return 0
    if args.command == "aggregate":
        outputs = aggregate_outputs(config)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False))
        return 0
    if args.command == "post-exposure-summary":
        outputs = create_post_exposure_summary(config)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False))
        return 0
    if args.command == "run":
        logging.getLogger(__name__).info(
            "run_context config=%s experiment=%s output_root=%s run_sha256=%s",
            config.path,
            config.data["experiment_name"],
            config.output_root,
            config.hash,
        )
        task = task_for_index(config, args.stage, args.task_index)
        logging.getLogger(__name__).info("task=%s", task)
        if args.stage == "init":
            run_initialization(config, int(task["network_id"]))
        elif args.stage in {"baseline", "initial_exposure"}:
            run_evaluation_group(
                config,
                stage=args.stage,
                network_id=int(task["network_id"]),
                environment=str(task["environment"]),
            )
        elif args.stage == "familiarization":
            run_familiarization(
                config,
                network_id=int(task["network_id"]),
                environment=str(task["environment"]),
                variant=str(task["variant"]),
            )
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())

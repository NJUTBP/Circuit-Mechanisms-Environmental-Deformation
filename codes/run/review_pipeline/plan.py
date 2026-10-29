from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .seeds import SeedBook, SeedRecord, write_seed_ledger
from .storage import atomic_json


STAGES = ("init", "baseline", "initial_exposure", "familiarization")
UPSTREAM_COMPATIBILITY_FILE = "upstream_compatibility.json"
UPSTREAM_REUSE_ALLOWED_DIFFERENCES = (
    ("initial_exposure", "retain_raw_activity"),
    ("storage", "raw_activity_selection", "stages", "initial_exposure", "environments"),
    ("familiarization", "trajectory_mode"),
    ("familiarization", "trajectory_ids"),
    ("slurm", "max_parallel_initial_exposure"),
    ("slurm", "max_parallel_familiarization"),
)


def _upstream_comparable_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove downstream-only settings that do not affect init/baseline."""

    comparable = json.loads(json.dumps(payload))
    comparable.pop("_resolved", None)
    comparable.pop("upstream_reuse", None)
    for keys in UPSTREAM_REUSE_ALLOWED_DIFFERENCES:
        value = comparable
        for key in keys[:-1]:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, dict):
            value.pop(keys[-1], None)
    return comparable


def prepare_upstream_compatibility(config: PipelineConfig) -> Path | None:
    """Verify and record reuse of completed init/baseline from the same v3 run.

    No simulation artifact or old manifest is rewritten.  Reuse is enabled only
    when the previous resolved configuration differs exclusively in the
    familiarization replication plan, LSF concurrency, or initial-exposure
    trace retention, and when the runtime and input fingerprints are unchanged.
    """

    reuse = config.data.get("upstream_reuse", {})
    if not bool(reuse.get("enabled", False)):
        return None
    output = config.output_root
    compatibility_path = output / UPSTREAM_COMPATIBILITY_FILE
    existing: dict[str, Any] = {}
    if compatibility_path.is_file():
        try:
            existing = json.loads(compatibility_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        if (
            existing.get("status") == "verified"
            and existing.get("target_run_sha256") == config.run_hash
        ):
            return compatibility_path

    resolved_path = output / "config.resolved.json"
    if not resolved_path.is_file():
        return None
    try:
        previous = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot verify the previous resolved configuration: {resolved_path}"
        ) from exc
    previous_resolved = previous.get("_resolved", {})
    source_run_hash = previous_resolved.get("run_sha256")
    if source_run_hash == config.run_hash:
        return None
    if not isinstance(source_run_hash, str) or len(source_run_hash) != 64:
        raise RuntimeError(
            f"Previous resolved configuration has no valid run hash: {resolved_path}"
        )
    if _upstream_comparable_config(previous) != _upstream_comparable_config(config.data):
        raise RuntimeError(
            "Existing v3 outputs cannot be reused: the previous configuration "
            "differs outside familiarization trajectory replication, LSF "
            "concurrency, and initial-exposure trace retention. Preserve the "
            "existing directory and use a new output_root."
        )
    if previous_resolved.get("runtime_sha256") != config.runtime_hash:
        raise RuntimeError(
            "Existing v3 outputs cannot be reused because the Python/package "
            "runtime fingerprint changed."
        )
    if previous_resolved.get("input_sha256") != config.input_hash:
        raise RuntimeError(
            "Existing v3 outputs cannot be reused because an external input "
            "fingerprint changed."
        )
    # Preserve a previously verified init/baseline lineage. A second downstream
    # revision must not discard the original checkpoint hash from the first.
    stage_sources = {
        stage: [source_run_hash] for stage in reuse["stages"]
    }
    if (
        existing.get("status") == "verified"
        and existing.get("target_run_sha256") == source_run_hash
    ):
        previous_stage_sources = existing.get("source_run_sha256_by_stage", {})
        for stage in reuse["stages"]:
            if stage not in existing.get("stages", []):
                continue
            candidates = [existing.get("source_run_sha256")]
            if isinstance(previous_stage_sources, dict):
                prior = previous_stage_sources.get(stage, [])
                if isinstance(prior, list):
                    candidates.extend(prior)
            stage_sources[stage] = sorted({
                source_run_hash,
                *(
                    value for value in candidates
                    if isinstance(value, str) and len(value) == 64
                ),
            })
    atomic_json(
        compatibility_path,
        {
            "status": "verified",
            "source_run_sha256": source_run_hash,
            "source_run_sha256_by_stage": stage_sources,
            "source_config_sha256": previous_resolved.get("config_sha256"),
            "source_pipeline_sha256": previous_resolved.get("pipeline_sha256"),
            "target_run_sha256": config.run_hash,
            "target_config_sha256": config.config_hash,
            "target_pipeline_sha256": config.pipeline_hash,
            "runtime_sha256": config.runtime_hash,
            "input_sha256": config.input_hash,
            "stages": list(reuse["stages"]),
            "allowed_configuration_differences": [
                ".".join(value) for value in UPSTREAM_REUSE_ALLOWED_DIFFERENCES
            ],
            "reason": reuse.get("reason", ""),
        },
    )
    return compatibility_path


def build_tasks(config: PipelineConfig) -> dict[str, list[dict[str, Any]]]:
    tasks: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    variants = config.data["familiarization"]["plasticity_variants"]
    for network_id in config.network_ids:
        tasks["init"].append({"network_id": network_id})
        tasks["baseline"].append({
            "network_id": network_id,
            "environment": config.baseline_environment["id"],
        })
        for env_id in config.stage_environments("initial_exposure"):
            tasks["initial_exposure"].append({
                "network_id": network_id,
                "environment": env_id,
            })
        for env_id in config.stage_environments("familiarization"):
            for variant in variants:
                tasks["familiarization"].append({
                    "network_id": network_id,
                    "environment": env_id,
                    "variant": variant["id"],
                })
    return tasks


def build_seed_records(config: PipelineConfig) -> list[SeedRecord]:
    book = SeedBook(config.master_seed)
    records: list[SeedRecord] = [book.record("grid_cell_sample", detail="shared_across_networks")]
    init_sessions = int(config.data["initialization"]["sessions"])
    fam_sessions = int(config.data["familiarization"]["sessions"])
    reuse_fam_trajectories = bool(
        config.data["familiarization"].get("reuse_trajectories_across_sessions", True)
    )
    reuse_initial_exposure = bool(
        config.data["familiarization"].get(
            "reuse_initial_exposure_trajectories", True
        )
    )
    te_cfg = config.data["analysis"]["te"]
    te_enabled = bool(te_cfg["enabled"])
    te_envs = set(te_cfg.get("environments", []))
    te_fam_sessions = {int(value) for value in te_cfg.get("familiarization_sessions", [])}
    variants = [item["id"] for item in config.data["familiarization"]["plasticity_variants"]]
    baseline = config.baseline_environment["id"]
    eval_envs = {baseline, *config.stage_environments("initial_exposure")}
    eval_envs.update(config.stage_environments("familiarization"))
    for network_id in config.network_ids:
        if config.data["model"]["bvc_tuning"]["source"] == "generated_per_network":
            records.append(book.record("bvc_distance", network_id=network_id))
            records.append(book.record("bvc_angle", network_id=network_id))
            records.append(book.record("bvc_angular_width", network_id=network_id))
        records.append(book.record("network_weights", network_id=network_id))
        activity_record = book.record(
            "initial_grid_activity", network_id=network_id
        )
        records.append(activity_record)
        if config.data["model"].get("grid_warmup_velocity_mode") == "seeded_small":
            records.append(SeedRecord(
                seed=activity_record.seed,
                purpose="initial_grid_warmup_velocity",
                network_id=network_id,
            ))
        for session in range(1, init_sessions + 1):
            records.append(book.record(
                "trajectory", network_id=network_id, environment=baseline,
                session=session, detail="initialization",
            ))
        for env_id in sorted(eval_envs):
            for trajectory_id in config.trajectory_ids:
                records.append(book.record(
                    "trajectory", network_id=network_id, environment=env_id,
                    trajectory_id=trajectory_id, detail="evaluation",
                ))
                records.append(book.record(
                    "analysis_bootstrap", network_id=network_id, environment=env_id,
                    trajectory_id=trajectory_id, session=0,
                ))
            if te_enabled and env_id in te_envs:
                records.append(book.record(
                    "te_analysis", network_id=network_id, environment=env_id, session=0,
                ))
            if env_id in config.stage_environments("familiarization"):
                for trajectory_id in config.familiarization_trajectory_ids:
                    seed_identity = {"trajectory_id": trajectory_id}
                    if config.data["familiarization"]["trajectory_mode"] == "branches":
                        seed_identity["branch"] = trajectory_id
                    if reuse_fam_trajectories and not reuse_initial_exposure:
                        records.append(book.record(
                            "trajectory", network_id=network_id, environment=env_id,
                            **seed_identity,
                            detail="familiarization",
                        ))
                    for session in range(1, fam_sessions + 1):
                        if not reuse_fam_trajectories and not reuse_initial_exposure:
                            records.append(book.record(
                                "trajectory", network_id=network_id, environment=env_id,
                                **seed_identity, session=session,
                                detail="familiarization",
                            ))
                        records.append(book.record(
                            "analysis_bootstrap", network_id=network_id, environment=env_id,
                            **seed_identity, session=session,
                        ))
                if te_enabled and env_id in te_envs:
                    for session in sorted(te_fam_sessions):
                        for variant in variants:
                            records.append(book.record(
                                "te_analysis", network_id=network_id,
                                environment=env_id, session=session, detail=variant,
                            ))
    return records


def prepare_experiment(config: PipelineConfig) -> dict[str, int]:
    output = config.output_root
    output.mkdir(parents=True, exist_ok=True)
    # This must run before replacing the previous resolved snapshot.
    prepare_upstream_compatibility(config)
    config.snapshot(output / "config.resolved.json")
    tasks = build_tasks(config)
    task_dir = output / "task_maps"
    task_dir.mkdir(parents=True, exist_ok=True)
    for stage, rows in tasks.items():
        atomic_json(task_dir / f"{stage}.json", rows)
    write_seed_ledger(output / "seed_ledger.csv", build_seed_records(config))
    # Create this one shared artifact before Slurm arrays start. Atomic writes are
    # still process-unique, but eager creation also makes the sampling choice auditable.
    from .simulation import grid_sample_indices

    grid_sample_indices(config)
    counts = {stage: len(rows) for stage, rows in tasks.items()}
    atomic_json(output / "task_counts.json", counts)
    return counts


def task_for_index(config: PipelineConfig, stage: str, index: int) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage: {stage}")
    tasks = build_tasks(config)[stage]
    if index < 0 or index >= len(tasks):
        raise IndexError(f"Task index {index} outside 0..{len(tasks)-1} for {stage}")
    return tasks[index]


def count_tasks(config: PipelineConfig, stage: str) -> int:
    return len(build_tasks(config)[stage])


def print_plan(config: PipelineConfig) -> None:
    tasks = build_tasks(config)
    print(f"experiment={config.data['experiment_name']}")
    print(f"config_sha256={config.config_hash}")
    print(f"pipeline_sha256={config.pipeline_hash}")
    print(f"runtime_sha256={config.runtime_hash}")
    print(f"input_sha256={config.input_hash}")
    print(f"run_sha256={config.run_hash}")
    print(f"output_root={config.output_root}")
    for stage in STAGES:
        print(f"{stage}: {len(tasks[stage])} tasks")

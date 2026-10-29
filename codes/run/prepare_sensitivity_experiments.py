"""Generate reproducible configs and flat task maps for reviewer sweeps."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from review_pipeline.config import PipelineConfig
from review_pipeline.plan import build_tasks, prepare_experiment
from review_pipeline.storage import atomic_json


PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_STAGES = ("init", "baseline", "initial_exposure", "familiarization")


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _condition_payload(
    base: dict[str, Any],
    base_config: PipelineConfig,
    spec: dict[str, Any],
    condition: dict[str, Any],
    spec_dir: Path,
) -> dict[str, Any]:
    payload = copy.deepcopy(base)
    payload.pop("_resolved", None)
    condition_id = str(condition["id"])
    output_root = resolve(spec_dir, spec["output_root"]) / condition_id
    payload["experiment_name"] = f'{spec["sweep_name"]}__{condition_id}'
    payload["project_root"] = str(base_config.project_root)
    payload["data_root"] = str(base_config.data_root)
    payload["output_root"] = str(output_root)
    payload["n_networks"] = int(spec["network_count"])
    payload["evaluation"]["trajectory_ids"] = [
        int(value) for value in spec.get("trajectory_ids", [0])
    ]
    payload["familiarization"]["trajectory_mode"] = "single"
    payload["familiarization"]["trajectory_ids"] = [
        int(value) for value in spec.get("trajectory_ids", [0])
    ]
    payload["initial_exposure"]["retain_raw_activity"] = bool(
        spec.get("retain_raw_activity", True)
    )
    payload["storage"]["raw_activity"] = (
        "all" if bool(spec.get("retain_raw_activity", True)) else "none"
    )
    payload["analysis"]["te"]["enabled"] = False
    payload["upstream_reuse"] = {
        "enabled": False,
        "stages": ["init", "baseline"],
        "reason": "Parameter-sensitivity conditions require independent initialization."
    }
    payload["slurm"]["exclusive_nodes"] = bool(spec.get("exclusive_nodes", False))
    parallel = spec["max_parallel"]
    for stage in PIPELINE_STAGES:
        if stage in parallel:
            payload["slurm"][f"max_parallel_{stage}"] = int(parallel[stage])

    kind = str(condition["kind"])
    if kind == "place_cells":
        n_place = int(condition["n_place"])
        inhibition = float(condition["place_global_inhibition"])
        if n_place <= 0 or inhibition <= 0:
            raise ValueError(f"Invalid place-cell condition: {condition}")
        if condition.get("inhibition_rule") == "1.5_times_100_over_n_place":
            expected = 1.5 * 100.0 / n_place
            if abs(inhibition - expected) > 1e-12:
                raise ValueError(
                    f"{condition_id}: scaled inhibition should be {expected}"
                )
        payload["model"]["n_place"] = n_place
        payload["model"]["place_global_inhibition"] = inhibition
    elif kind == "grid_velocity_gains":
        gains = [float(value) for value in condition["grid_velocity_gains"]]
        if len(gains) != 3 or min(gains) <= 0:
            raise ValueError(f"Invalid grid velocity gains: {condition}")
        alpha_1 = float(condition["alpha_1"])
        expected = [alpha_1, 0.7 * alpha_1, 0.5 * alpha_1]
        if any(abs(value - target) > 1e-12 for value, target in zip(gains, expected)):
            raise ValueError(
                f"{condition_id}: gains must follow [alpha1, 0.7*alpha1, 0.5*alpha1]"
            )
        payload["model"]["grid_velocity_gains"] = gains
    else:
        raise ValueError(f"Unknown sensitivity condition kind: {kind}")
    return payload


def make_payloads(
    spec_path: Path,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    base_path = resolve(spec_path.parent, spec["base_config"])
    base_config = PipelineConfig(base_path)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    conditions = spec.get("conditions", [])
    identities = [str(condition["id"]) for condition in conditions]
    if not identities or len(identities) != len(set(identities)):
        raise ValueError("Sensitivity condition IDs must be non-empty and unique")
    payloads = [
        (
            condition,
            _condition_payload(
                base, base_config, spec, condition, spec_path.parent
            ),
        )
        for condition in conditions
    ]
    return spec, payloads


def prepare(spec_path: Path) -> Path:
    spec, payloads = make_payloads(spec_path)
    root = resolve(spec_path.parent, spec["output_root"])
    config_dir = root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    selected_stages = [str(value) for value in spec["stages"]]
    unknown = set(selected_stages) - set(PIPELINE_STAGES)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")

    flattened: dict[str, list[dict[str, Any]]] = {
        stage: [] for stage in PIPELINE_STAGES
    }
    audit_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    for condition, payload in payloads:
        condition_id = str(condition["id"])
        config_path = config_dir / f"{condition_id}.json"
        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous != payload:
                raise RuntimeError(
                    f"Existing generated config differs: {config_path}; "
                    "use a new output_root or remove only the unstarted sweep."
                )
        else:
            atomic_json(config_path, payload)
        config = PipelineConfig(config_path)
        prepare_experiment(config)
        tasks = build_tasks(config)
        counts = {stage: len(tasks[stage]) for stage in PIPELINE_STAGES}
        condition_rows.append({
            "condition_id": condition_id,
            "kind": condition["kind"],
            "parameters": condition,
            "config": str(config_path),
            "output_root": str(config.output_root),
            "task_counts": counts,
        })
        audit_rows.append({
            "condition_id": condition_id,
            "config": str(config_path),
            "local_task_index": 0,
        })
        for stage in selected_stages:
            for local_index, task in enumerate(tasks[stage]):
                flattened[stage].append({
                    "condition_id": condition_id,
                    "config": str(config_path),
                    "local_task_index": local_index,
                    "task": task,
                })

    task_maps = {
        stage: rows for stage, rows in flattened.items() if stage in selected_stages
    }
    task_maps["audit_init"] = audit_rows
    manifest_path = root / "sensitivity_manifest.json"
    atomic_json(manifest_path, {
        "status": "ready",
        "spec": str(spec_path),
        "sweep_name": spec["sweep_name"],
        "job_cores": int(spec.get("job_cores", 2)),
        "exclusive_nodes": bool(spec.get("exclusive_nodes", False)),
        "stages": selected_stages,
        "max_parallel": {
            key: int(value) for key, value in spec["max_parallel"].items()
        },
        "trajectory_ids": [int(value) for value in spec.get("trajectory_ids", [0])],
        "retain_raw_activity": bool(spec.get("retain_raw_activity", True)),
        "conditions": condition_rows,
        "tasks": task_maps,
        "task_counts": {stage: len(rows) for stage, rows in task_maps.items()},
    })
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    spec_path = args.spec.expanduser().resolve()
    if args.plan:
        spec, payloads = make_payloads(spec_path)
        print(json.dumps({
            "sweep_name": spec["sweep_name"],
            "conditions": [
                {
                    "id": condition["id"],
                    "n_networks": payload["n_networks"],
                    "trajectory_ids": payload["evaluation"]["trajectory_ids"],
                    "n_place": payload["model"]["n_place"],
                    "place_global_inhibition": payload["model"]["place_global_inhibition"],
                    "grid_velocity_gains": payload["model"]["grid_velocity_gains"],
                }
                for condition, payload in payloads
            ],
        }, indent=2))
    else:
        print(prepare(spec_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

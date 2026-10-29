"""Prepare 0x learning-rate controls from completed fixed-delta place-cell runs.

The existing learning-rate preparer copies source BVC tuning, baseline metrics,
initial-exposure checkpoints, and *exact* familiarization trajectories. Source
simulation outputs are read only. Three generated configs share one flat LSF
task map, so all 72 familiarization tasks can be submitted in one array.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare_learning_rate_control import PATHWAYS, prepare as prepare_one
from review_pipeline.config import PipelineConfig
from review_pipeline.plan import build_tasks
from review_pipeline.storage import atomic_json


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SPEC = PROJECT_DIR / "configs/config.place_cell_zero_rates.json"
EXPECTED_COUNTS = (200, 300, 400)
EXPECTED_ENVIRONMENTS = ("compressed_80", "elongated_120")


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    counts = tuple(int(value) for value in spec["place_cell_counts"])
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Expected place-cell counts {EXPECTED_COUNTS}; got {counts}")
    if int(spec["network_count"]) != 4:
        raise ValueError("This experiment requires four networks per place-cell count")
    if tuple(spec["environments"]) != EXPECTED_ENVIRONMENTS:
        raise ValueError("This experiment requires the 80 and 120 cm environments")
    if int(spec["trajectory_id"]) != 0 or list(spec["multipliers"]) != [0.0]:
        raise ValueError("Only trajectory 0 and the three 0x pathway variants are allowed")
    if int(spec["job_cores"]) != 2 or int(spec["max_parallel"]) != 72:
        raise ValueError("This experiment requests 72 concurrent two-core tasks")
    if bool(spec["exclusive_nodes"]) or not bool(spec["retain_raw_activity"]):
        raise ValueError("Exclusive nodes must be off and raw activity retained")
    if bool(spec.get("inline_te", False)):
        raise ValueError("Compute TE later from retained activity, not during simulation")
    return spec


def condition_specs(spec: dict[str, Any], spec_dir: Path) -> list[dict[str, Any]]:
    source_root = resolve(spec_dir, str(spec["source_output_root"]))
    destination_root = resolve(spec_dir, str(spec["destination_output_root"]))
    if source_root == destination_root or source_root in destination_root.parents or destination_root in source_root.parents:
        raise ValueError("Source and destination output roots must be disjoint")
    conditions = []
    for n_place in EXPECTED_COUNTS:
        condition_id = f"npc{n_place}_delta_fixed"
        conditions.append({
            "condition_id": condition_id,
            "n_place": n_place,
            "source_root": source_root / condition_id,
            "destination_root": destination_root / condition_id,
            "experiment_name": f'{spec["experiment_name"]}__{condition_id}',
        })
    return conditions


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"Existing prepared file differs; preserve it: {path}")
    else:
        atomic_json(path, payload)


def _validate_source(condition: dict[str, Any], spec: dict[str, Any]) -> None:
    source_root = Path(condition["source_root"])
    snapshot = source_root / "config.resolved.json"
    if not snapshot.is_file():
        raise RuntimeError(f"Missing fixed-delta source config: {snapshot}")
    source = json.loads(snapshot.read_text(encoding="utf-8"))
    model = source["model"]
    if int(model["n_place"]) != int(condition["n_place"]):
        raise RuntimeError(f"Wrong n_place in source config: {snapshot}")
    if abs(float(model["place_global_inhibition"]) - 1.5) > 1e-12:
        raise RuntimeError(f"Source is not the delta=1.5 fixed condition: {snapshot}")
    if int(source["n_networks"]) != int(spec["network_count"]):
        raise RuntimeError(f"Wrong network count in source config: {snapshot}")
    if list(source["familiarization"]["trajectory_ids"]) != [0]:
        raise RuntimeError(f"Expected a single familiarization trajectory 0: {snapshot}")
    variants = source["familiarization"]["plasticity_variants"]
    if len(variants) != 1 or variants[0]["id"] != "intact":
        raise RuntimeError(f"Expected the existing intact 1x control: {snapshot}")
    for pathway in PATHWAYS.values():
        if float(variants[0]["multipliers"][pathway]) != 1.0:
            raise RuntimeError(f"Intact control is not 1x for {pathway}: {snapshot}")
    if int(spec["trajectory_id"]) not in source["evaluation"]["trajectory_ids"]:
        raise RuntimeError(f"Missing evaluation trajectory 0: {snapshot}")


def prepare(spec_path: Path) -> Path:
    spec_path = spec_path.expanduser().resolve()
    spec = load_spec(spec_path)
    conditions = condition_specs(spec, spec_path.parent)
    # Check every source snapshot before creating new experiment files.
    for condition in conditions:
        _validate_source(condition, spec)

    rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        derived_spec = {
            "source_output_root": str(condition["source_root"]),
            "destination_output_root": str(condition["destination_root"]),
            "experiment_name": condition["experiment_name"],
            "network_count": int(spec["network_count"]),
            "environments": list(spec["environments"]),
            "trajectory_id": int(spec["trajectory_id"]),
            "multipliers": list(spec["multipliers"]),
            "max_parallel": int(spec["max_parallel"]),
            "job_cores": int(spec["job_cores"]),
            "exclusive_nodes": bool(spec["exclusive_nodes"]),
            "retain_raw_activity": bool(spec["retain_raw_activity"]),
            "inline_te": False,
        }
        # Keep the derived spec beside the existing project configs so that
        # prepare_learning_rate_control resolves project_root exactly as before.
        derived_spec_path = PROJECT_DIR / f"config.place_cell_lr_zero_{condition_id}.generated.json"
        _write_once(derived_spec_path, derived_spec)
        generated_config = prepare_one(derived_spec_path)
        config = PipelineConfig(generated_config)
        tasks = build_tasks(config)["familiarization"]
        if len(tasks) != 24 or int(config.data["model"]["n_place"]) != condition["n_place"]:
            raise RuntimeError(f"Unexpected task plan for {condition_id}")
        for local_index, task in enumerate(tasks):
            rows.append({
                "condition_id": condition_id,
                "config": str(generated_config),
                "local_task_index": local_index,
                "task": task,
            })
        condition_rows.append({
            "condition_id": condition_id,
            "n_place": condition["n_place"],
            "config": str(generated_config),
            "source_output_root": str(condition["source_root"]),
            "output_root": str(condition["destination_root"]),
            "reference_variant": "intact",
            "task_count": len(tasks),
        })

    expected = len(EXPECTED_COUNTS) * int(spec["network_count"]) * len(EXPECTED_ENVIRONMENTS) * len(PATHWAYS)
    if len(rows) != expected or expected != 72:
        raise RuntimeError(f"Expected 72 familiarization tasks; got {len(rows)}")
    root = resolve(spec_path.parent, str(spec["destination_output_root"]))
    manifest = root / "place_cell_lr_zero_manifest.json"
    payload = {
        "status": "ready",
        "sweep_name": spec["experiment_name"],
        "spec": str(spec_path),
        "job_cores": 2,
        "exclusive_nodes": False,
        "trajectory_ids": [0],
        "retain_raw_activity": True,
        "max_parallel": {"familiarization": 72},
        "task_counts": {"familiarization": len(rows)},
        "conditions": condition_rows,
        "tasks": {"familiarization": rows},
        "notes": {
            "source_outputs_unmodified": True,
            "one_pathway_zero_at_a_time": True,
            "source_intact_familiarization_is_1x_reference": True,
            "source_trajectories_copied_exactly": True,
            "each_condition_starts_from_its_own_initial_exposure_checkpoint": True,
        },
    }
    _write_once(manifest, payload)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    spec_path = args.config.expanduser().resolve()
    spec = load_spec(spec_path)
    if args.plan:
        conditions = condition_specs(spec, spec_path.parent)
        print(json.dumps({
            "task_count": 72,
            "max_parallel": 72,
            "cores_per_task": 2,
            "maximum_requested_cores": 144,
            "source_conditions": [str(row["source_root"]) for row in conditions],
            "output_conditions": [str(row["destination_root"]) for row in conditions],
            "variants": [f"{pathway}_0p0x" for pathway in PATHWAYS],
            "environments": list(spec["environments"]),
            "trajectory_id": int(spec["trajectory_id"]),
            "sessions": 9,
        }, indent=2))
    else:
        print(prepare(spec_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

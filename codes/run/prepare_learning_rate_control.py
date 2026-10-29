"""Prepare independent one-pathway-at-a-time familiarization controls."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np

from review_pipeline.config import PipelineConfig
from review_pipeline.plan import build_tasks, prepare_experiment
from review_pipeline.storage import (
    atomic_json, complete_manifest, manifest_is_complete, sha256_file,
)


PATHWAYS = {
    "W_bp": "bvc_to_place",
    "W_pg": "place_to_grid",
    "W_gp": "grid_to_place",
}


def resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def make_payload(base: dict, spec: dict, config_dir: Path) -> dict:
    payload = copy.deepcopy(base)
    payload.pop("_resolved", None)
    payload["project_root"] = str(config_dir.parent.resolve())
    payload["data_root"] = str(resolve(config_dir, base["data_root"]))
    payload["output_root"] = str(resolve(config_dir, spec["destination_output_root"]))
    payload["experiment_name"] = str(spec["experiment_name"])
    payload["n_networks"] = int(spec["network_count"])
    payload["evaluation"]["trajectory_ids"] = [int(spec["trajectory_id"])]
    environments = [str(value) for value in spec["environments"]]
    payload["initial_exposure"]["environments"] = environments
    fam = payload["familiarization"]
    fam["environments"] = environments
    fam["trajectory_mode"] = "single"
    fam["trajectory_ids"] = [int(spec["trajectory_id"])]
    variants = []
    for short_name, pathway in PATHWAYS.items():
        for value in spec["multipliers"]:
            multipliers = {
                "bvc_to_place": 1.0, "place_to_grid": 1.0,
                "grid_to_place": 1.0, "place_recurrent": 0.0,
            }
            multipliers[pathway] = float(value)
            label = str(float(value)).replace(".", "p")
            variants.append({"id": f"{short_name}_{label}x", "multipliers": multipliers})
    fam["plasticity_variants"] = variants
    payload["storage"]["raw_activity"] = (
        "all" if bool(spec.get("retain_raw_activity", True)) else "none"
    )
    payload["analysis"]["te"]["enabled"] = bool(spec.get("inline_te", False))
    payload["slurm"]["max_parallel_familiarization"] = int(spec["max_parallel"])
    payload["upstream_reuse"] = {"enabled": False}
    return payload


def checked_manifest(directory: Path, required: list[str]) -> str:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    digest = str(manifest.get("run_sha256", ""))
    if len(digest) != 64 or not manifest_is_complete(directory, {digest}, required):
        raise RuntimeError(f"Incomplete/changed source files: {directory}")
    return digest


def copy_unchanged(source: Path, target: Path) -> None:
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise RuntimeError(f"Existing copied upstream file differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def prepare(spec_path: Path) -> Path:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    source_root = resolve(spec_path.parent, spec["source_output_root"])
    destination = resolve(spec_path.parent, spec["destination_output_root"])
    if source_root == destination or source_root in destination.parents or destination in source_root.parents:
        raise ValueError("Source and destination must be disjoint output roots")
    snapshot = source_root / "config.resolved.json"
    base = json.loads(snapshot.read_text(encoding="utf-8"))
    if int(base["n_networks"]) < int(spec["network_count"]):
        raise ValueError("Source experiment has too few networks")
    payload = make_payload(base, spec, spec_path.parent)
    config_path = destination / "config.learning_rate_control.generated.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError("Existing control config differs; use a new destination_output_root")
    atomic_json(config_path, payload)
    config = PipelineConfig(config_path)
    tasks = build_tasks(config)["familiarization"]
    variant_count = len(PATHWAYS) * len(spec["multipliers"])
    expected_tasks = (
        int(spec["network_count"]) * len(spec["environments"]) * variant_count
    )
    if (
        len(tasks) != expected_tasks
        or len(payload["familiarization"]["plasticity_variants"]) != variant_count
    ):
        raise ValueError(
            f"Expected {expected_tasks} tasks / {variant_count} conditions, "
            f"got {len(tasks)} tasks"
        )
    if payload["model"] != base["model"]:
        raise RuntimeError("Base model or default learning rates changed")
    if int(spec["trajectory_id"]) not in {
        int(value) for value in base["evaluation"]["trajectory_ids"]
    }:
        raise RuntimeError("Requested trajectory is absent from the source run")

    copied_inputs = []
    baseline_id = config.baseline_environment["id"]
    environments = [str(value) for value in spec["environments"]]
    for network_id in config.network_ids:
        source_network = source_root / "networks" / f"network_{network_id:03d}"
        target_network = destination / "networks" / f"network_{network_id:03d}"
        bvc = source_network / "init" / "bvc_tuning.npz"
        metadata = json.loads((bvc.with_suffix(".json")).read_text(encoding="utf-8"))
        if metadata.get("sha256") != sha256_file(bvc):
            raise RuntimeError(f"BVC tuning changed: {bvc}")
        target_bvc = target_network / "init" / "bvc_tuning.npz"
        copy_unchanged(bvc, target_bvc)
        source_bvc_hash = str(metadata["run_sha256"])
        metadata["run_sha256"] = config.run_hash
        metadata["copied_source_run_sha256"] = source_bvc_hash
        atomic_json(target_bvc.with_suffix(".json"), metadata)
        copied_inputs.append({"path": str(bvc), "sha256": sha256_file(bvc)})
        baseline = source_network / "baseline" / baseline_id
        required = ["network_summary.csv.gz"]
        for trajectory_id in config.trajectory_ids:
            required.extend([
                f"trajectory_{trajectory_id:02d}/cell_metrics.csv.gz",
                f"trajectory_{trajectory_id:02d}/rate_maps.npz",
            ])
        source_hash = checked_manifest(baseline, required)
        target_baseline = target_network / "baseline" / baseline_id
        outputs, inputs = [], []
        for relative in required:
            original, copied = baseline / relative, target_baseline / relative
            copy_unchanged(original, copied)
            outputs.append(copied)
            inputs.append(original)
        complete_manifest(
            target_baseline, config.run_hash, "baseline",
            {"network_id": network_id, "environment": baseline_id}, {},
            outputs, inputs,
            notes={"copied_source_run_sha256": source_hash,
                   "purpose": "Baseline references for learning-rate controls"},
        )
        copied_inputs.extend({"path": str(p), "sha256": sha256_file(p)} for p in inputs)

        # The familiarization loader validates every configured evaluation
        # checkpoint. Copy both production trajectories even though only
        # trajectory 0 is continued by the sensitivity experiment.
        for environment in environments:
            exposure = source_network / "initial_exposure" / environment
            target = target_network / "initial_exposure" / environment
            required = ["network_summary.csv.gz", *[
                f"trajectory_{i:02d}/checkpoint.npz" for i in config.trajectory_ids
            ]]
            source_hash = checked_manifest(exposure, required)
            outputs, inputs = [], []
            for relative in required:
                original, copied = exposure / relative, target / relative
                copy_unchanged(original, copied)
                outputs.append(copied)
                inputs.append(original)
            complete_manifest(
                target, config.run_hash, "initial_exposure",
                {"network_id": network_id, "environment": environment}, {},
                outputs, inputs,
                notes={"copied_source_run_sha256": source_hash,
                       "purpose": "Session-0 state for learning-rate sensitivity"},
            )
            copied_inputs.extend(
                {"path": str(p), "sha256": sha256_file(p)} for p in inputs
            )

    prepare_experiment(config)
    # Pre-copy the production familiarization trajectories. This prevents
    # concurrent variants from racing to create the same file, and guarantees
    # that every multiplier uses the exact source positions in every session.
    trajectory_id = int(spec["trajectory_id"])
    sessions = int(payload["familiarization"]["sessions"])
    for network_id in config.network_ids:
        for environment in environments:
            for session in range(1, sessions + 1):
                relative = Path("trajectories") / f"network_{network_id:03d}" / environment \
                    / "familiarization" / f"session_{session:02d}__trajectory_{trajectory_id:02d}" \
                    / "trajectory.npz"
                original = source_root / relative
                if not original.exists():
                    raise RuntimeError(
                        "Required production familiarization trajectory is not ready: "
                        f"{original}. Wait until that network has generated this session, "
                        "then run the submission command again."
                    )
                source_hash = checked_manifest(original.parent, [original.name])
                copied = destination / relative
                copy_unchanged(original, copied)
                complete_manifest(
                    copied.parent, config.run_hash, "trajectory",
                    {"network_id": network_id, "environment": environment,
                     "purpose": "familiarization", "trajectory_id": trajectory_id,
                     "session": session, "branch": None}, {}, [copied], [original],
                    notes={"copied_source_run_sha256": source_hash,
                           "purpose": "Exact source trajectory shared by all variants"},
                )
                copied_inputs.append(
                    {"path": str(original), "sha256": sha256_file(original)}
                )
    with np.load(source_root / "grid_sample_indices.npz", allow_pickle=False) as original:
        source_indices = original["indices"]
    with np.load(destination / "grid_sample_indices.npz", allow_pickle=False) as prepared:
        if not np.array_equal(source_indices, prepared["indices"]):
            raise RuntimeError("GC sample identities differ from original experiment")
    atomic_json(destination / "learning_rate_control_prepared.json", {
        "status": "ready", "config": str(config_path),
        "source_output_root": str(source_root), "task_count": len(tasks),
        "environments": environments, "trajectory_id": trajectory_id,
        "max_parallel": int(spec["max_parallel"]),
        "exclusive_nodes": bool(spec.get("exclusive_nodes", False)),
        "base_learning_rates": payload["model"]["learning_rates"],
        "variants": payload["familiarization"]["plasticity_variants"],
        "copied_source_inputs": copied_inputs,
    })
    return config_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "configs/config.learning_rates.json")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    spec_path = args.config.resolve()
    if args.plan:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        source_root = resolve(spec_path.parent, spec["source_output_root"])
        base = json.loads(
            (source_root / "config.resolved.json").read_text(encoding="utf-8")
        )
        payload = make_payload(base, spec, spec_path.parent)
        print(json.dumps({"task_count": int(spec["network_count"])
                          * len(spec["environments"])
                          * len(PATHWAYS) * len(spec["multipliers"]),
                          "sessions": payload["familiarization"]["sessions"],
                          "base_learning_rates": payload["model"]["learning_rates"],
                          "variants": payload["familiarization"]["plasticity_variants"]}, indent=2))
    else:
        print(prepare(spec_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

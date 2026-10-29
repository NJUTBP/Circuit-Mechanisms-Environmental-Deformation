from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from .bvc import (
    bvc_tuning_path,
    bvc_tuning_seeds,
    ensure_bvc_tuning,
    load_bvc_tuning,
)
from .config import PipelineConfig
from .metrics import (
    COMPARISON_FIELDS,
    SUMMARY_FIELDS,
    analyze_result,
    compare_to_baseline,
    read_csv_gz,
    summarize_cells,
    summarize_comparisons,
)
from .model import NetworkState, initialize_network
from .paths import (
    evaluation_group_dir,
    evaluation_run_dir,
    familiarization_root,
    familiarization_session_dir,
    init_checkpoint,
    init_session_dir,
    network_root,
)
from .qc_figures import (
    EVALUATION_QC_FILENAMES,
    write_evaluation_rate_map_atlases,
    write_initialization_grid_qc,
)
from .seeds import SeedBook
from .simulation import SimulationResult, run_session
from .storage import (
    atomic_csv_gz,
    atomic_json,
    complete_manifest,
    manifest_is_complete,
    sha256_file,
)
from .te import ProjectionBundle, compare_te_to_baseline, compute_te_group
from .trajectory import ensure_trajectory


LOGGER = logging.getLogger(__name__)

INITIALIZATION_SESSION_REQUIRED = [
    "checkpoint.npz",
    "weight_qc.json",
    "qc_grid_activity.npz",
    "qc_grid_fields.png",
    "qc_grid_fields.pdf",
    "qc_grid_fields.svg",
]


def _weight_qc(state: NetworkState) -> dict[str, Any]:
    matrices = {
        "bvc_to_place": state.w_bvc_to_place,
        "place_recurrent": state.w_place_recurrent,
        **{f"grid_to_place_m{i+1}": value for i, value in enumerate(state.w_grid_to_place)},
        **{f"place_to_grid_m{i+1}": value for i, value in enumerate(state.w_place_to_grid)},
    }
    return {
        name: {
            "shape": list(matrix.shape),
            "minimum": float(np.min(matrix)),
            "maximum": float(np.max(matrix)),
            "mean": float(np.mean(matrix)),
            "frobenius_norm": float(np.linalg.norm(matrix)),
            "column_sum_min": float(np.min(matrix.sum(axis=0))),
            "column_sum_max": float(np.max(matrix.sum(axis=0))),
            "row_sum_min": float(np.min(matrix.sum(axis=1))),
            "row_sum_max": float(np.max(matrix.sum(axis=1))),
        }
        for name, matrix in matrices.items()
    }


def _activity_qc(state: NetworkState) -> dict[str, Any]:
    if state.place_activity is None or state.grid_activity is None:
        return {"present": False}
    activities = {
        "place": state.place_activity,
        **{
            f"grid_m{index + 1}": activity
            for index, activity in enumerate(state.grid_activity)
        },
    }
    return {
        "present": True,
        **{
            name: {
                "shape": list(activity.shape),
                "minimum": float(np.min(activity)),
                "maximum": float(np.max(activity)),
                "mean": float(np.mean(activity)),
            }
            for name, activity in activities.items()
        },
    }
def _keep_raw(
    config: PipelineConfig,
    network_id: int,
    stage: str,
    environment: str,
    session: int = 0,
) -> bool:
    # Preserve initial-exposure traces for later rebinning and phase/TE debugging.
    if stage == "initial_exposure" and config.data["initial_exposure"].get(
        "retain_raw_activity", False
    ):
        return True
    policy = config.data["storage"]["raw_activity"]
    if policy == "all":
        return True
    if policy == "qa":
        return network_id in [int(value) for value in config.data["storage"].get("qa_network_ids", [0])]
    if policy == "selective":
        selection = config.data["storage"].get("raw_activity_selection", {})
        selected_networks = selection.get("network_ids", "all")
        if selected_networks != "all" and network_id not in {
            int(value) for value in selected_networks
        }:
            return False
        stage_rule = selection.get("stages", {}).get(stage)
        if not isinstance(stage_rule, dict):
            return False
        environments = stage_rule.get("environments", "all")
        if environments != "all" and environment not in environments:
            return False
        sessions = stage_rule.get("sessions", "all")
        if sessions != "all" and session not in {int(value) for value in sessions}:
            return False
        return True
    return False


def _run_te_for(
    config: PipelineConfig, stage: str, environment: str, session: int = 0
) -> bool:
    te_cfg = config.data["analysis"]["te"]
    if not bool(te_cfg["enabled"]):
        return False
    if environment not in te_cfg.get("environments", [environment]):
        return False
    if stage == "familiarization":
        sessions = [int(value) for value in te_cfg.get("familiarization_sessions", [])]
        return session in sessions
    return True


def _load_final_initialization(
    config: PipelineConfig, network_id: int
) -> tuple[Path, NetworkState]:
    final_session = int(config.data["initialization"]["sessions"])
    directory = init_session_dir(config, network_id, final_session)
    checkpoint = directory / "checkpoint.npz"
    if not manifest_is_complete(
        directory,
        config.accepted_run_hashes("init"),
        INITIALIZATION_SESSION_REQUIRED,
    ):
        raise RuntimeError(
            f"Initialization checkpoint/manifest is incomplete or has changed: {checkpoint}"
        )
    state = NetworkState.load(checkpoint)
    state.validate(
        config.data["model"],
        label=f"network {network_id} final initialization checkpoint",
        require_activity=True,
    )
    return checkpoint, state


def _baseline_reference_paths(
    config: PipelineConfig, network_id: int
) -> dict[int, tuple[Path, Path]]:
    environment = config.baseline_environment["id"]
    group = evaluation_group_dir(config, "baseline", network_id, environment)
    required = ["network_summary.csv.gz"]
    references: dict[int, tuple[Path, Path]] = {}
    for trajectory_id in config.trajectory_ids:
        run_dir = evaluation_run_dir(
            config, "baseline", network_id, environment, trajectory_id
        )
        cells = run_dir / "cell_metrics.csv.gz"
        maps = run_dir / "rate_maps.npz"
        references[trajectory_id] = (cells, maps)
        required.extend([
            os.path.relpath(cells, group),
            os.path.relpath(maps, group),
        ])
    if not manifest_is_complete(
        group, config.accepted_run_hashes("baseline"), required
    ):
        raise RuntimeError(
            "Baseline reference metrics are incomplete or have changed for "
            f"network {network_id}; run/repair baseline before downstream stages"
        )
    return references


def _initial_exposure_checkpoint_paths(
    config: PipelineConfig, network_id: int, environment: str
) -> dict[int, Path]:
    group = evaluation_group_dir(
        config, "initial_exposure", network_id, environment
    )
    checkpoints = {
        trajectory_id: evaluation_run_dir(
            config, "initial_exposure", network_id, environment, trajectory_id
        ) / "checkpoint.npz"
        for trajectory_id in config.trajectory_ids
    }
    required = [
        "network_summary.csv.gz",
        *[os.path.relpath(path, group) for path in checkpoints.values()],
    ]
    if not manifest_is_complete(group, config.hash, required):
        raise RuntimeError(
            "Initial-exposure activity/checkpoint state is incomplete or has changed "
            f"for network {network_id}, environment {environment}"
        )
    return checkpoints


def audit_initialization(config: PipelineConfig) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for network_id in config.network_ids:
        load_bvc_tuning(config, network_id)
        root = network_root(config, network_id) / "init"
        initial_path = root / "initial_weights.npz"
        metadata_path = root / "initial_weights.json"
        if not initial_path.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Missing initial weights for network {network_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_seed = SeedBook(config.master_seed).seed(
            "network_weights", network_id=network_id
        )
        initial_sha = sha256_file(initial_path)
        if (
            metadata.get("run_sha256") not in config.accepted_run_hashes("init")
            or metadata.get("seed") != expected_seed
            or metadata.get("sha256") != initial_sha
        ):
            raise RuntimeError(
                f"Initial weight metadata/hash mismatch for network {network_id}"
            )
        initial_state = NetworkState.load(initial_path)
        initial_state.validate(
            config.data["model"], label=f"network {network_id} initial weights"
        )
        session_checkpoints: list[dict[str, Any]] = []
        for session in range(1, int(config.data["initialization"]["sessions"]) + 1):
            session_directory = init_session_dir(config, network_id, session)
            session_checkpoint = session_directory / "checkpoint.npz"
            if not manifest_is_complete(
                session_directory,
                config.accepted_run_hashes("init"),
                INITIALIZATION_SESSION_REQUIRED,
            ):
                raise RuntimeError(
                    "Initialization session checkpoint/manifest is incomplete or "
                    f"has changed: {session_checkpoint}"
                )
            session_state = NetworkState.load(session_checkpoint)
            session_state.validate(
                config.data["model"],
                label=f"network {network_id} initialization session {session}",
                require_activity=True,
            )
            session_checkpoints.append({
                "session": session,
                "path": str(session_checkpoint),
                "sha256": sha256_file(session_checkpoint),
            })
        checkpoint, final_state = _load_final_initialization(config, network_id)
        records.append({
            "network_id": network_id,
            "network_weights_seed": expected_seed,
            "initial_weights_sha256": initial_sha,
            "session_checkpoints": session_checkpoints,
            "final_checkpoint": str(checkpoint),
            "final_checkpoint_sha256": sha256_file(checkpoint),
            "bvc_tuning_sha256": sha256_file(bvc_tuning_path(config, network_id)),
            "weight_qc": _weight_qc(final_state),
            "activity_qc": _activity_qc(final_state),
        })
    unique_keys = ["initial_weights_sha256", "final_checkpoint_sha256"]
    if config.data["model"]["bvc_tuning"]["source"] == "generated_per_network":
        unique_keys.append("bvc_tuning_sha256")
    for key in unique_keys:
        values = [record[key] for record in records]
        if len(values) != len(set(values)):
            raise RuntimeError(f"Independent networks contain duplicate {key}")
    report = {
        "status": "ok",
        "run_sha256": config.hash,
        "n_networks": len(records),
        "all_connection_files_unique": True,
        "networks": records,
    }
    atomic_json(config.output_root / "initialization_audit.json", report)
    return report


def run_initialization(config: PipelineConfig, network_id: int) -> None:
    book = SeedBook(config.master_seed)
    root = network_root(config, network_id) / "init"
    root.mkdir(parents=True, exist_ok=True)
    ensure_bvc_tuning(config, network_id)
    tuning_path = bvc_tuning_path(config, network_id)
    tuning_seeds = bvc_tuning_seeds(config, network_id)
    initial_path = root / "initial_weights.npz"
    initial_manifest = root / "initial_weights.json"
    weights_seed = book.seed("network_weights", network_id=network_id)
    if initial_path.exists() and initial_manifest.exists():
        metadata = json.loads(initial_manifest.read_text(encoding="utf-8"))
        if (
            metadata.get("run_sha256") != config.hash
            or metadata.get("seed") != weights_seed
            or metadata.get("sha256") != sha256_file(initial_path)
        ):
            raise RuntimeError(
                f"Existing initial weights do not match this config/seed: {initial_path}"
            )
        state = NetworkState.load(initial_path)
    else:
        state = initialize_network(config.data["model"], weights_seed)
        state.save(initial_path, compressed=False)
        atomic_json(initial_manifest, {
            "run_sha256": config.hash,
            "seed": weights_seed,
            "bytes": initial_path.stat().st_size,
            "sha256": sha256_file(initial_path),
            "weight_qc": _weight_qc(state),
        })
    state.validate(config.data["model"], label=f"network {network_id} initial weights")

    env = config.baseline_environment
    n_sessions = int(config.data["initialization"]["sessions"])
    for session in range(1, n_sessions + 1):
        directory = init_session_dir(config, network_id, session)
        checkpoint = directory / "checkpoint.npz"
        required = INITIALIZATION_SESSION_REQUIRED
        if manifest_is_complete(directory, config.hash, required):
            LOGGER.info("skip complete init network=%d session=%d", network_id, session)
            state = NetworkState.load(checkpoint)
            state.validate(
                config.data["model"],
                label=f"network {network_id} initialization session {session}",
                require_activity=True,
            )
            continue
        if session > 1:
            previous = init_checkpoint(config, network_id, session - 1)
            if not previous.exists():
                raise RuntimeError(f"Missing previous initialization checkpoint: {previous}")
            state = NetworkState.load(previous)
            state.validate(
                config.data["model"],
                label=f"network {network_id} initialization session {session - 1}",
                require_activity=True,
            )
        trajectory_seed = book.seed(
            "trajectory",
            network_id=network_id,
            environment=env["id"],
            session=session,
            detail="initialization",
        )
        activity_seed = (
            book.seed("initial_grid_activity", network_id=network_id)
            if session == 1
            else None
        )
        trajectory_file, positions = ensure_trajectory(
            config,
            env,
            trajectory_seed,
            network_id=network_id,
            purpose="initialization",
            session=session,
        )
        result = run_session(
            config,
            env,
            positions,
            state,
            network_id=network_id,
            activity_seed=activity_seed,
            plasticity_enabled=True,
            record=True,
            grid_record_indices=np.array([
                (int(config.data["model"]["n_grid"]) // 2)
                * int(config.data["model"]["n_grid"])
                + int(config.data["model"]["n_grid"]) // 2
            ]),
            record_place=False,
        )
        state = result.state
        state.validate(
            config.data["model"],
            label=f"network {network_id} initialization session {session}",
            require_activity=True,
        )
        state.save(checkpoint, compressed=False)
        qc_path = directory / "weight_qc.json"
        atomic_json(qc_path, {
            "weights": _weight_qc(state),
            "activity": _activity_qc(state),
        })
        centre = int(config.data["model"]["n_grid"]) // 2
        grid_flat_indices = np.array(
            [centre * int(config.data["model"]["n_grid"]) + centre],
            dtype=np.int64,
        )
        grid_qc_outputs = write_initialization_grid_qc(
            config,
            result,
            env,
            directory,
            network_id=network_id,
            session=session,
            grid_flat_indices=grid_flat_indices,
        )
        inputs = [
            tuning_path,
            trajectory_file,
            initial_path if session == 1 else init_checkpoint(config, network_id, session - 1),
        ]
        manifest_seeds = {
            "network_weights": weights_seed,
            **tuning_seeds,
            "trajectory": trajectory_seed,
        }
        if activity_seed is not None:
            manifest_seeds["initial_grid_activity"] = activity_seed
            if config.data["model"].get("grid_warmup_velocity_mode") == "seeded_small":
                manifest_seeds["initial_grid_warmup_velocity"] = activity_seed
        complete_manifest(
            directory,
            config.hash,
            "init",
            {"network_id": network_id, "session": session, "environment": env["id"]},
            manifest_seeds,
            [checkpoint, qc_path, *grid_qc_outputs],
            inputs,
            {
                "plasticity": config.data["initialization"]["plasticity"],
                "grid_qc": {
                    "one_centre_sheet_cell_per_module": True,
                    "grid_flat_indices": grid_flat_indices.tolist(),
                    "record_every_steps": int(
                        config.data["model"]["record_every_steps"]
                    ),
                },
            },
        )


def _work_is_reusable(
    directory: Path,
    run_hash: str,
    state_input_path: Path,
    phase_reference_cells_path: Path | None = None,
) -> bool:
    metadata_path = directory / "work_manifest.json"
    transient = directory / "transient_activity.npz"
    cells = directory / "cell_metrics.csv.gz"
    maps = directory / "rate_maps.npz"
    if not all(path.exists() for path in (metadata_path, transient, cells, maps)):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("run_sha256") != run_hash:
        return False
    if (
        not state_input_path.is_file()
        or metadata.get("state_input_sha256") != sha256_file(state_input_path)
    ):
        return False
    expected_phase_hash = metadata.get("phase_reference_cells_sha256")
    if phase_reference_cells_path is None:
        if expected_phase_hash is not None:
            return False
    elif (
        not phase_reference_cells_path.is_file()
        or expected_phase_hash != sha256_file(phase_reference_cells_path)
    ):
        return False
    expected_hashes = metadata.get("output_sha256", {})
    paths = [transient, cells, maps]
    checkpoint = directory / "checkpoint.npz"
    if not checkpoint.exists():
        return False
    paths.append(checkpoint)
    return all(expected_hashes.get(path.name) == sha256_file(path) for path in paths)


def _evaluate_one(
    config: PipelineConfig,
    *,
    stage: str,
    network_id: int,
    env: dict[str, Any],
    trajectory_id: int,
    state: NetworkState,
    state_input_path: Path,
    output_dir: Path,
    trajectory_seed: int,
    activity_seed: int | None,
    analysis_seed: int,
    plasticity_enabled: bool,
    variant: str | None = None,
    session: int = 0,
    branch: int | None = None,
    purpose: str = "evaluation",
    phase_reference_cells_path: Path | None = None,
) -> tuple[SimulationResult, list[dict[str, Any]], Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    transient = output_dir / "transient_activity.npz"
    maps_path = output_dir / "rate_maps.npz"
    cells_path = output_dir / "cell_metrics.csv.gz"
    checkpoint = output_dir / "checkpoint.npz"
    if _work_is_reusable(
        output_dir, config.hash, state_input_path, phase_reference_cells_path
    ):
        result_state = NetworkState.load(checkpoint) if checkpoint.exists() else state
        result = SimulationResult.load_transient(transient, result_state)
        return result, read_csv_gz(cells_path), maps_path, cells_path, transient, checkpoint

    trajectory_file, positions = ensure_trajectory(
        config,
        env,
        trajectory_seed,
        network_id=network_id,
        purpose=purpose,
        trajectory_id=trajectory_id,
        session=(
            session
            if purpose == "familiarization"
            and not bool(
                config.data["familiarization"].get(
                    "reuse_trajectories_across_sessions", True
                )
            )
            else None
        ),
        branch=None if purpose == "evaluation" else branch,
    )
    result = run_session(
        config,
        env,
        positions,
        state,
        network_id=network_id,
        activity_seed=activity_seed,
        plasticity_enabled=plasticity_enabled,
        variant_id=variant,
        record=True,
    )
    result.save_transient(transient)
    identity = {
        "network_id": network_id,
        "stage": stage,
        "environment": env["id"],
        "trajectory_id": trajectory_id,
        "branch": "" if branch is None else branch,
        "session": session,
    }
    maps_path, cells_path, rows = analyze_result(
        config,
        result,
        env,
        output_dir,
        identity,
        analysis_seed,
        phase_reference_cells_path=phase_reference_cells_path,
    )
    result.state.save(checkpoint, compressed=False)
    work_outputs = [transient, maps_path, cells_path, checkpoint]
    atomic_json(output_dir / "work_manifest.json", {
        "run_sha256": config.hash,
        "trajectory_seed": trajectory_seed,
        "activity_seed": activity_seed,
        "analysis_seed": analysis_seed,
        "trajectory_path": str(trajectory_file),
        "state_input_path": str(state_input_path),
        "state_input_sha256": sha256_file(state_input_path),
        "phase_reference_cells_path": (
            None if phase_reference_cells_path is None
            else str(phase_reference_cells_path)
        ),
        "phase_reference_cells_sha256": (
            None if phase_reference_cells_path is None
            else sha256_file(phase_reference_cells_path)
        ),
        "plasticity_enabled": plasticity_enabled,
        "variant": variant,
        "output_sha256": {path.name: sha256_file(path) for path in work_outputs},
    })
    return result, rows, maps_path, cells_path, transient, checkpoint


def run_evaluation_group(
    config: PipelineConfig,
    *,
    stage: str,
    network_id: int,
    environment: str,
) -> None:
    if stage not in {"baseline", "initial_exposure"}:
        raise ValueError(f"Invalid evaluation stage: {stage}")
    # Validate the immutable asset even when completed activity work is reused.
    load_bvc_tuning(config, network_id)
    book = SeedBook(config.master_seed)
    env = config.environment(environment)
    group = evaluation_group_dir(config, stage, network_id, environment)
    te_enabled = _run_te_for(config, stage, environment, session=0)
    keep_raw = _keep_raw(config, network_id, stage, environment, session=0)
    required = ["network_summary.csv.gz", *EVALUATION_QC_FILENAMES]
    if te_enabled:
        required.append("te_metrics.csv.gz")
        if stage == "baseline":
            required.append("projection_basis.npz")
            if "umap" in config.data["analysis"]["te"]["embeddings"]:
                required.append("umap_models.pkl")
        else:
            required.append("te_delta.csv.gz")
    for trajectory_id in config.trajectory_ids:
        required.extend([
            f"trajectory_{trajectory_id:02d}/cell_metrics.csv.gz",
            f"trajectory_{trajectory_id:02d}/rate_maps.npz",
            f"trajectory_{trajectory_id:02d}/checkpoint.npz",
        ])
        if keep_raw:
            required.append(f"trajectory_{trajectory_id:02d}/transient_activity.npz")
        if stage == "initial_exposure":
            required.append(f"trajectory_{trajectory_id:02d}/comparison_metrics.csv.gz")
    if manifest_is_complete(group, config.hash, required):
        LOGGER.info("skip complete %s network=%d env=%s", stage, network_id, environment)
        return
    checkpoint, state = _load_final_initialization(config, network_id)
    baseline_references = (
        _baseline_reference_paths(config, network_id)
        if stage == "initial_exposure"
        else {}
    )
    results: list[SimulationResult] = []
    identities: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    all_comparison: list[dict[str, Any]] = []
    outputs: list[Path] = []
    rate_map_paths: list[Path] = []
    transient_paths: list[Path] = []
    trajectory_inputs: list[Path] = []

    for trajectory_id in config.trajectory_ids:
        trajectory_seed = book.seed(
            "trajectory",
            network_id=network_id,
            environment=environment,
            trajectory_id=trajectory_id,
            detail="evaluation",
        )
        activity_seed = None
        analysis_seed = book.seed(
            "analysis_bootstrap",
            network_id=network_id,
            environment=environment,
            trajectory_id=trajectory_id,
            session=0,
        )
        run_dir = evaluation_run_dir(
            config, stage, network_id, environment, trajectory_id
        )
        if stage == "initial_exposure":
            baseline_cells, baseline_maps = baseline_references[trajectory_id]
            phase_reference_cells = baseline_cells
        else:
            baseline_cells = baseline_maps = None
            phase_reference_cells = None
        result, rows, maps_path, cells_path, transient, output_checkpoint = _evaluate_one(
            config,
            stage=stage,
            network_id=network_id,
            env=env,
            trajectory_id=trajectory_id,
            state=state,
            state_input_path=checkpoint,
            output_dir=run_dir,
            trajectory_seed=trajectory_seed,
            activity_seed=activity_seed,
            analysis_seed=analysis_seed,
            plasticity_enabled=False,
            phase_reference_cells_path=phase_reference_cells,
        )
        results.append(result)
        all_cells.extend(rows)
        identities.append({
            "network_id": network_id,
            "stage": stage,
            "environment": environment,
            "trajectory_id": trajectory_id,
            "branch": "",
            "session": 0,
        })
        outputs.extend([maps_path, cells_path, output_checkpoint])
        rate_map_paths.append(maps_path)
        transient_paths.append(transient)
        trajectory_inputs.append(
            Path(json.loads((run_dir / "work_manifest.json").read_text(encoding="utf-8"))["trajectory_path"])
        )
        if stage == "initial_exposure":
            assert baseline_cells is not None and baseline_maps is not None
            comparison_path, comparison = compare_to_baseline(
                config,
                env,
                run_dir,
                identities[-1],
                rows,
                baseline_cells,
                baseline_maps,
                maps_path,
            )
            all_comparison.extend(comparison)
            outputs.append(comparison_path)

    outputs.extend(
        write_evaluation_rate_map_atlases(
            config,
            rate_map_paths,
            group,
            stage=stage,
            network_id=network_id,
            env=env,
        )
    )

    projection_dir = evaluation_group_dir(
        config, "baseline", network_id, config.baseline_environment["id"]
    )
    if te_enabled:
        te_seed = book.seed("te_analysis", network_id=network_id, environment=environment, session=0)
        if stage == "baseline":
            projection = ProjectionBundle.fit(config, results, env, te_seed)
            outputs.extend(projection.save(group))
        else:
            projection = ProjectionBundle.load(projection_dir)
        te_path = group / "te_metrics.csv.gz"
        te_rows = compute_te_group(
            config, results, env, identities, projection, te_seed, te_path
        )
        outputs.append(te_path)
        if stage == "initial_exposure":
            baseline_te = read_csv_gz(projection_dir / "te_metrics.csv.gz")
            te_delta_path = group / "te_delta.csv.gz"
            compare_te_to_baseline(te_rows, baseline_te, te_delta_path)
            outputs.append(te_delta_path)

    if stage == "baseline":
        summaries = summarize_cells(
            all_cells,
            network_id=network_id,
            stage=stage,
            environment=environment,
            session=0,
        )
    else:
        summaries = summarize_comparisons(
            all_comparison,
            network_id=network_id,
            stage=stage,
            environment=environment,
            session=0,
        )
    summary_path = group / "network_summary.csv.gz"
    atomic_csv_gz(summary_path, summaries, SUMMARY_FIELDS)
    outputs.append(summary_path)
    if keep_raw:
        outputs.extend(transient_paths)
    manifest_seeds = {
        "network_weights": book.seed("network_weights", network_id=network_id),
        **bvc_tuning_seeds(config, network_id),
    }
    if te_enabled:
        manifest_seeds["te_analysis"] = te_seed
    complete_manifest(
        group,
        config.hash,
        stage,
        {"network_id": network_id, "environment": environment},
        manifest_seeds,
        outputs,
        [
            checkpoint,
            bvc_tuning_path(config, network_id),
            *trajectory_inputs,
            *(
                path
                for pair in baseline_references.values()
                for path in pair
            ),
        ],
        {"raw_activity_retained": keep_raw, "statistical_unit": "network"},
    )
    if not keep_raw:
        for transient in transient_paths:
            transient.unlink(missing_ok=True)


def run_familiarization(
    config: PipelineConfig,
    *,
    network_id: int,
    environment: str,
    variant: str,
) -> None:
    # Validate the immutable asset before loading/reusing any trajectory outputs.
    load_bvc_tuning(config, network_id)
    book = SeedBook(config.master_seed)
    env = config.environment(environment)
    root = familiarization_root(config, network_id, environment, variant)
    projection_dir = evaluation_group_dir(
        config, "baseline", network_id, config.baseline_environment["id"]
    )
    baseline_references = _baseline_reference_paths(config, network_id)
    trajectory_ids = config.familiarization_trajectory_ids
    trajectory_mode = config.data["familiarization"]["trajectory_mode"]
    exposure_checkpoints = _initial_exposure_checkpoint_paths(
        config, network_id, environment
    )
    states = {
        trajectory_id: NetworkState.load(exposure_checkpoints[trajectory_id])
        for trajectory_id in trajectory_ids
    }
    for trajectory_id, state in states.items():
        state.validate(
            config.data["model"],
            label=(
                f"network {network_id} familiarization trajectory {trajectory_id} "
                "initial checkpoint"
            ),
            require_activity=True,
        )
    state_input_paths = {
        trajectory_id: exposure_checkpoints[trajectory_id]
        for trajectory_id in trajectory_ids
    }
    n_sessions = int(config.data["familiarization"]["sessions"])
    projection = (
        ProjectionBundle.load(projection_dir)
        if any(
            _run_te_for(config, "familiarization", environment, session=session)
            for session in range(1, n_sessions + 1)
        )
        else None
    )
    for session in range(1, n_sessions + 1):
        session_group = root / "session_outputs" / f"session_{session:02d}"
        checkpoint_paths = [
            familiarization_session_dir(
                config, network_id, environment, variant, trajectory_id, session
            ) / "checkpoint.npz"
            for trajectory_id in trajectory_ids
        ]
        required = [
            "network_summary.csv.gz",
            *EVALUATION_QC_FILENAMES,
        ]
        for trajectory_id in trajectory_ids:
            run_dir = familiarization_session_dir(
                config, network_id, environment, variant, trajectory_id, session
            )
            required.extend([
                os.path.relpath(run_dir / "checkpoint.npz", session_group),
                os.path.relpath(run_dir / "cell_metrics.csv.gz", session_group),
                os.path.relpath(run_dir / "rate_maps.npz", session_group),
                os.path.relpath(run_dir / "comparison_metrics.csv.gz", session_group),
            ])
        session_te_enabled = _run_te_for(
            config, "familiarization", environment, session=session
        )
        keep_raw = _keep_raw(
            config, network_id, "familiarization", environment, session=session
        )
        if keep_raw:
            required.extend([
                os.path.relpath(
                    familiarization_session_dir(
                        config, network_id, environment, variant, trajectory_id, session
                    ) / "transient_activity.npz",
                    session_group,
                )
                for trajectory_id in trajectory_ids
            ])
        if session_te_enabled:
            required.extend(["te_metrics.csv.gz", "te_delta.csv.gz"])
        if manifest_is_complete(session_group, config.hash, required):
            for trajectory_id, checkpoint_path in zip(
                trajectory_ids, checkpoint_paths
            ):
                states[trajectory_id] = NetworkState.load(checkpoint_path)
                states[trajectory_id].validate(
                    config.data["model"],
                    label=(
                        f"network {network_id} familiarization trajectory "
                        f"{trajectory_id} "
                        f"session {session}"
                    ),
                    require_activity=True,
                )
                state_input_paths[trajectory_id] = checkpoint_path
            LOGGER.info(
                "skip complete familiarization network=%d env=%s variant=%s session=%d",
                network_id, environment, variant, session,
            )
            continue

        results: list[SimulationResult] = []
        identities: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        outputs: list[Path] = []
        transient_paths: list[Path] = []
        inputs: list[Path] = []
        for trajectory_id in trajectory_ids:
            branch = trajectory_id if trajectory_mode == "branches" else None
            run_dir = familiarization_session_dir(
                config, network_id, environment, variant, trajectory_id, session
            )
            reuse_initial_exposure = bool(
                config.data["familiarization"].get(
                    "reuse_initial_exposure_trajectories", True
                )
            )
            if reuse_initial_exposure:
                trajectory_seed_parts: dict[str, Any] = {
                    "network_id": network_id,
                    "environment": environment,
                    "trajectory_id": trajectory_id,
                    "detail": "evaluation",
                }
                trajectory_purpose = "evaluation"
            else:
                trajectory_seed_parts = {
                    "network_id": network_id,
                    "environment": environment,
                    "trajectory_id": trajectory_id,
                    "detail": "familiarization",
                }
                if branch is not None:
                    trajectory_seed_parts["branch"] = branch
                trajectory_purpose = "familiarization"
                if not bool(
                    config.data["familiarization"].get(
                        "reuse_trajectories_across_sessions", True
                    )
                ):
                    trajectory_seed_parts["session"] = session
            trajectory_seed = book.seed("trajectory", **trajectory_seed_parts)
            activity_seed = None
            analysis_seed_parts: dict[str, Any] = {
                "network_id": network_id,
                "environment": environment,
                "trajectory_id": trajectory_id,
                "session": session,
            }
            if branch is not None:
                analysis_seed_parts["branch"] = branch
            analysis_seed = book.seed("analysis_bootstrap", **analysis_seed_parts)
            baseline_cells, baseline_maps = baseline_references[trajectory_id]
            result, rows, maps_path, cells_path, transient, checkpoint_path = _evaluate_one(
                config,
                stage="familiarization",
                network_id=network_id,
                env=env,
                trajectory_id=trajectory_id,
                state=states[trajectory_id],
                state_input_path=state_input_paths[trajectory_id],
                output_dir=run_dir,
                trajectory_seed=trajectory_seed,
                activity_seed=activity_seed,
                analysis_seed=analysis_seed,
                plasticity_enabled=True,
                variant=variant,
                session=session,
                branch=branch,
                purpose=trajectory_purpose,
                phase_reference_cells_path=baseline_cells,
            )
            states[trajectory_id] = result.state
            states[trajectory_id].validate(
                config.data["model"],
                label=(
                    f"network {network_id} familiarization trajectory "
                    f"{trajectory_id} "
                    f"session {session}"
                ),
                require_activity=True,
            )
            inputs.append(state_input_paths[trajectory_id])
            state_input_paths[trajectory_id] = checkpoint_path
            results.append(result)
            identity = {
                "network_id": network_id,
                "stage": "familiarization",
                "environment": environment,
                "trajectory_id": trajectory_id,
                "branch": "" if branch is None else branch,
                "session": session,
            }
            identities.append(identity)
            comparison_path, comparison = compare_to_baseline(
                config,
                env,
                run_dir,
                identity,
                rows,
                baseline_cells,
                baseline_maps,
                maps_path,
            )
            comparisons.extend(comparison)
            outputs.extend([maps_path, cells_path, comparison_path, checkpoint_path])
            transient_paths.append(transient)
            work = json.loads((run_dir / "work_manifest.json").read_text(encoding="utf-8"))
            inputs.append(Path(work["trajectory_path"]))
            inputs.extend([baseline_cells, baseline_maps])

        outputs.extend(
            write_evaluation_rate_map_atlases(
                config,
                [
                    familiarization_session_dir(
                        config, network_id, environment, variant,
                        trajectory_id, session,
                    ) / "rate_maps.npz"
                    for trajectory_id in trajectory_ids
                ],
                session_group,
                stage="familiarization",
                network_id=network_id,
                env=env,
                trajectory_ids=trajectory_ids,
                variant=variant,
                session=session,
            )
        )

        if projection is not None and session_te_enabled:
            te_seed = book.seed(
                "te_analysis",
                network_id=network_id,
                environment=environment,
                session=session,
                detail=variant,
            )
            te_path = session_group / "te_metrics.csv.gz"
            te_rows = compute_te_group(
                config, results, env, identities, projection, te_seed, te_path
            )
            baseline_te = read_csv_gz(projection_dir / "te_metrics.csv.gz")
            delta_path = session_group / "te_delta.csv.gz"
            compare_te_to_baseline(te_rows, baseline_te, delta_path)
            outputs.extend([te_path, delta_path])
        summaries = summarize_comparisons(
            comparisons,
            network_id=network_id,
            stage="familiarization",
            environment=environment,
            variant=variant,
            session=session,
        )
        summary_path = session_group / "network_summary.csv.gz"
        atomic_csv_gz(summary_path, summaries, SUMMARY_FIELDS)
        outputs.append(summary_path)
        if keep_raw:
            outputs.extend(transient_paths)
        manifest_seeds = {
            "network_weights": book.seed("network_weights", network_id=network_id),
            **bvc_tuning_seeds(config, network_id),
        }
        if session_te_enabled:
            manifest_seeds["te_analysis"] = te_seed
        complete_manifest(
            session_group,
            config.hash,
            "familiarization",
            {
                "network_id": network_id,
                "environment": environment,
                "variant": variant,
                "session": session,
            },
            manifest_seeds,
            outputs,
            [bvc_tuning_path(config, network_id), *inputs],
            {
                "trajectory_mode": trajectory_mode,
                "trajectory_ids": trajectory_ids,
                "raw_activity_retained": keep_raw,
                "statistical_unit": "network",
            },
        )
        if not keep_raw:
            for transient in transient_paths:
                transient.unlink(missing_ok=True)

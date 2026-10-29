from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig


def network_root(config: PipelineConfig, network_id: int) -> Path:
    return config.output_root / "networks" / f"network_{network_id:03d}"


def init_session_dir(config: PipelineConfig, network_id: int, session: int) -> Path:
    return network_root(config, network_id) / "init" / f"session_{session:02d}"


def init_checkpoint(config: PipelineConfig, network_id: int, session: int | None = None) -> Path:
    if session is None:
        session = int(config.data["initialization"]["sessions"])
    return init_session_dir(config, network_id, session) / "checkpoint.npz"


def evaluation_group_dir(
    config: PipelineConfig, stage: str, network_id: int, environment: str
) -> Path:
    return network_root(config, network_id) / stage / environment


def evaluation_run_dir(
    config: PipelineConfig,
    stage: str,
    network_id: int,
    environment: str,
    trajectory_id: int,
) -> Path:
    return evaluation_group_dir(config, stage, network_id, environment) / f"trajectory_{trajectory_id:02d}"


def familiarization_root(
    config: PipelineConfig, network_id: int, environment: str, variant: str
) -> Path:
    return network_root(config, network_id) / "familiarization" / environment / variant


def familiarization_session_dir(
    config: PipelineConfig,
    network_id: int,
    environment: str,
    variant: str,
    trajectory_id: int,
    session: int,
) -> Path:
    trajectory_mode = config.data["familiarization"]["trajectory_mode"]
    trajectory_directory = (
        f"trajectory_{trajectory_id:02d}"
        if trajectory_mode == "single"
        else f"branch_{trajectory_id:02d}"
    )
    return (
        familiarization_root(config, network_id, environment, variant)
        / trajectory_directory
        / f"session_{session:02d}"
    )


def trajectory_path(
    config: PipelineConfig,
    network_id: int,
    environment: str,
    purpose: str,
    trajectory_id: int | None = None,
    session: int | None = None,
    branch: int | None = None,
) -> Path:
    root = config.output_root / "trajectories" / f"network_{network_id:03d}" / environment / purpose
    parts = []
    if branch is not None:
        parts.append(f"branch_{branch:02d}")
    if session is not None:
        parts.append(f"session_{session:02d}")
    if trajectory_id is not None:
        parts.append(f"trajectory_{trajectory_id:02d}")
    if not parts:
        raise ValueError("A trajectory path requires a session or trajectory_id")
    return root / "__".join(parts) / "trajectory.npz"

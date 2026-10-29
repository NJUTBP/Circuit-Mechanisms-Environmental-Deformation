from __future__ import annotations

import contextlib
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import PipelineConfig
from .paths import trajectory_path
from .storage import atomic_npz, complete_manifest, manifest_is_complete


@contextlib.contextmanager
def _legacy_numpy_seed(seed: int) -> Iterator[None]:
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    np.random.seed(int(seed))
    random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _load_existing(path: Path) -> tuple[np.ndarray, float]:
    value = np.load(path, allow_pickle=True)
    if value.shape == ():
        payload = value.item()
        positions = np.asarray(payload["pos"], dtype=np.float32)
        times = np.asarray(payload.get("t", []), dtype=float)
        dt = float(np.median(np.diff(times))) if times.size > 1 else np.nan
    else:
        positions = np.asarray(value, dtype=np.float32)
        dt = np.nan
    return positions, dt


def _generate_positions(
    env: dict[str, Any], model_cfg: dict[str, Any], trajectory_cfg: dict[str, Any], seed: int
) -> np.ndarray:
    try:
        from ratinabox.Agent import Agent
        from ratinabox.Environment import Environment
    except ImportError as exc:
        raise RuntimeError(
            "RatInABox is required to generate trajectories; install requirements.txt "
            "or configure trajectory.source='existing'."
        ) from exc

    width = float(env["width_m"])
    height = float(env["height_m"])
    dt = float(model_cfg["dt_s"])
    n_steps = int(round(float(model_cfg["duration_s"]) / dt))
    with _legacy_numpy_seed(seed):
        environment = Environment(params={"scale": height, "aspect": width / height})
        agent = Agent(environment)
        agent.dt = dt
        agent.speed_mean = float(trajectory_cfg["mean_speed_m_s"])
        agent.pos = np.array([width / 2.0, height / 2.0])
        for _ in range(n_steps):
            agent.update()
        positions = np.asarray(agent.get_history_arrays()["pos"], dtype=np.float32)
    return positions


def ensure_trajectory(
    config: PipelineConfig,
    env: dict[str, Any],
    seed: int,
    *,
    network_id: int,
    purpose: str,
    trajectory_id: int | None = None,
    session: int | None = None,
    branch: int | None = None,
) -> tuple[Path, np.ndarray]:
    target = trajectory_path(
        config,
        network_id,
        env["id"],
        purpose,
        trajectory_id=trajectory_id,
        session=session,
        branch=branch,
    )
    required = [target.name]
    if manifest_is_complete(target.parent, config.hash, required):
        with np.load(target, allow_pickle=False) as data:
            return target, data["positions"].astype(np.float32, copy=False)

    trajectory_cfg = config.data["trajectory"]
    source = trajectory_cfg["source"]
    source_path: Path | None = None
    if source == "generated":
        positions = _generate_positions(env, config.data["model"], trajectory_cfg, seed)
        source_dt = float(config.data["model"]["dt_s"])
    elif source == "existing":
        pattern = env.get("trajectory_pattern")
        if not pattern:
            raise ValueError(f"Environment {env['id']} has no trajectory_pattern")
        formatted = pattern.format(
            network_id=network_id,
            trajectory_id=0 if trajectory_id is None else trajectory_id,
            session=0 if session is None else session,
            branch=0 if branch is None else branch,
        )
        source_path = config.resolve_path(formatted, base=config.data_root)
        positions, source_dt = _load_existing(source_path)
    else:
        raise ValueError(f"Unknown trajectory.source: {source}")

    expected_dt = float(config.data["model"]["dt_s"])
    if np.isfinite(source_dt) and not np.isclose(source_dt, expected_dt, rtol=0, atol=1e-9):
        raise ValueError(f"Trajectory dt={source_dt} does not match model dt={expected_dt}: {source_path}")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"Trajectory positions must have shape (time, 2), got {positions.shape}")
    minimum_steps = int(float(config.data["model"]["duration_s"]) / expected_dt) - 2
    if len(positions) < minimum_steps:
        raise ValueError(f"Trajectory is too short: {len(positions)} < {minimum_steps}")
    width = float(env["width_m"])
    height = float(env["height_m"])
    tolerance = 1e-5
    if (
        positions[:, 0].min() < -tolerance
        or positions[:, 1].min() < -tolerance
        or positions[:, 0].max() > width + tolerance
        or positions[:, 1].max() > height + tolerance
    ):
        raise ValueError(f"Trajectory leaves environment {env['id']}")

    atomic_npz(
        target,
        compressed=bool(config.data["storage"].get("compress_trajectories", True)),
        positions=positions.astype(np.float32, copy=False),
        dt_s=np.array(expected_dt),
        seed=np.array(seed, dtype=np.uint32),
    )
    complete_manifest(
        target.parent,
        config.hash,
        "trajectory",
        {
            "network_id": network_id,
            "environment": env["id"],
            "purpose": purpose,
            "trajectory_id": trajectory_id,
            "session": session,
            "branch": branch,
        },
        {"trajectory": seed},
        [target],
        [] if source_path is None else [source_path],
    )
    return target, positions

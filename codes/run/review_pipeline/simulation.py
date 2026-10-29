from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bvc import load_bvc_tuning
from .config import PipelineConfig
from .model import (
    GridAttractor,
    NetworkState,
    apply_hebbian_update,
    initial_grid_activity,
    relu,
    target_sparsity_inhibition,
)
from .storage import atomic_npz


LOGGER = logging.getLogger(__name__)


def _peak_normalized_von_mises(
    theta: np.ndarray,
    mu: np.ndarray | float,
    sigma: np.ndarray,
    norm: float | np.ndarray = 1.0,
) -> np.ndarray:
    """Evaluate a peak-normalized von Mises function without overflow.

    RatInABox 1.15.3 evaluates ``exp(kappa*cos(delta))`` and
    ``norm/exp(kappa)`` separately.  The Methods distribution permits angular
    widths below about 2.15 degrees, for which those two terms become ``inf``
    and zero even though their product is finite.  Combining the exponents is
    algebraically identical and keeps the Methods distribution unchanged.
    """

    theta_array = np.asarray(theta, dtype=np.float64)
    mu_array = np.asarray(mu, dtype=np.float64)
    sigma_array = np.asarray(sigma, dtype=np.float64)
    if np.any(~np.isfinite(sigma_array)) or np.any(sigma_array <= 0.0):
        raise ValueError("von Mises angular widths must be finite and positive")
    cosine = np.clip(np.cos(theta_array - mu_array), -1.0, 1.0)
    exponent = (cosine - 1.0) / np.square(sigma_array)
    return np.asarray(norm, dtype=np.float64) * np.exp(exponent)


def _install_ratinabox_peak_normalization(rib_utils: Any) -> None:
    """Patch only RatInABox's explicitly peak-normalized von Mises calls."""

    current = rib_utils.von_mises
    if getattr(current, "_rescale_review_stable", False):
        return

    def stable_von_mises(theta, mu, sigma, norm=None):
        if norm is None:
            return current(theta, mu, sigma, norm=norm)
        return _peak_normalized_von_mises(theta, mu, sigma, norm)

    stable_von_mises._rescale_review_stable = True
    rib_utils.von_mises = stable_von_mises


@dataclass
class SimulationResult:
    positions: np.ndarray
    place_activity: np.ndarray
    grid_activity: tuple[np.ndarray, np.ndarray, np.ndarray]
    record_dt_s: float
    state: NetworkState

    def save_transient(self, path: Path) -> None:
        atomic_npz(
            path,
            compressed=False,
            positions=self.positions.astype(np.float32, copy=False),
            place=self.place_activity.astype(np.float32, copy=False),
            grid_m1=self.grid_activity[0].astype(np.float32, copy=False),
            grid_m2=self.grid_activity[1].astype(np.float32, copy=False),
            grid_m3=self.grid_activity[2].astype(np.float32, copy=False),
            record_dt_s=np.array(self.record_dt_s),
        )

    @classmethod
    def load_transient(cls, path: Path, state: NetworkState) -> "SimulationResult":
        with np.load(path, allow_pickle=False) as data:
            return cls(
                positions=data["positions"].astype(np.float32, copy=False),
                place_activity=data["place"].astype(np.float32, copy=False),
                grid_activity=tuple(
                    data[f"grid_m{module}"].astype(np.float32, copy=False)
                    for module in (1, 2, 3)
                ),
                record_dt_s=float(data["record_dt_s"]),
                state=state,
            )


def grid_sample_indices(config: PipelineConfig) -> np.ndarray:
    path = config.output_root / "grid_sample_indices.npz"
    mode = config.data["analysis"]["grid_sample"]["mode"]
    count = int(config.data["analysis"]["grid_sample"]["n_cells"])
    n_total = int(config.data["model"]["n_grid"]) ** 2
    if mode == "legacy_file":
        indices = np.load(config.input_path("grid_indices"), allow_pickle=False).astype(np.int64)
    elif mode == "seeded_shared":
        from .seeds import SeedBook

        seed = SeedBook(config.master_seed).seed("grid_cell_sample", detail="shared_across_networks")
        indices = np.random.default_rng(seed).choice(n_total, count, replace=False)
    else:
        raise ValueError(f"Unknown grid sample mode: {mode}")
    if len(indices) != count or len(np.unique(indices)) != count:
        raise ValueError("Grid sample indices are not unique or have the wrong length")
    if indices.min() < 0 or indices.max() >= n_total:
        raise ValueError("Grid sample index is outside the grid sheet")
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as data:
                stored = data["indices"].astype(np.int64, copy=False)
                stored_run_hash = str(data["run_sha256"].item())
            if stored_run_hash == config.hash and np.array_equal(stored, indices):
                return stored
        except (KeyError, OSError, ValueError):
            pass
    atomic_npz(
        path,
        indices=indices.astype(np.int32),
        run_sha256=np.array(config.hash),
        mode=np.array(mode),
    )
    return indices


def _build_bvc(config: PipelineConfig, env: dict[str, Any], network_id: int):
    try:
        from ratinabox import utils as rib_utils
        from ratinabox.Agent import Agent
        from ratinabox.Environment import Environment
        from ratinabox.Neurons import BoundaryVectorCells
    except ImportError as exc:
        raise RuntimeError("RatInABox is required for network simulation") from exc

    width = float(env["width_m"])
    height = float(env["height_m"])
    environment = Environment(params={"scale": height, "aspect": width / height})
    agent = Agent(environment)
    agent.dt = float(config.data["model"]["dt_s"])
    tuning = load_bvc_tuning(config, network_id)
    radial = config.data["model"]["bvc_tuning"]["radial_width"]
    sigma_distance_m = (
        float(radial["xi_m"])
        + tuning.distance_m / float(radial["beta"])
    )

    # RatInABox 1.15.3 accepts sigma_angle in create_random_assembly(), but a
    # typo in VectorCells.default_params causes a misleading "unexpected key"
    # warning.  Supplying the exact four arrays through the documented custom
    # arrangement path avoids that warning and prevents any internal resampling.
    def exact_arrangement(**_kwargs):
        return (
            tuning.distance_m.copy(),
            np.deg2rad(tuning.angle_deg),
            sigma_distance_m,
            np.deg2rad(tuning.sigma_angle_deg),
        )

    _install_ratinabox_peak_normalization(rib_utils)
    cells = BoundaryVectorCells(
        agent,
        params={
            "n": len(tuning.distance_m),
            "cell_arrangement": exact_arrangement,
        },
    )
    if not np.all(np.isfinite(cells.cell_fr_norm)):
        raise FloatingPointError(
            "RatInABox BVC normalization is non-finite; "
            f"angular width range={tuning.sigma_angle_deg.min():.6g}.."
            f"{tuning.sigma_angle_deg.max():.6g} degrees"
        )
    return agent, cells


def _plasticity_multipliers(config: PipelineConfig, variant_id: str | None, enabled: bool) -> dict[str, float]:
    pathways = ("bvc_to_place", "grid_to_place", "place_to_grid", "place_recurrent")
    if not enabled:
        return {pathway: 0.0 for pathway in pathways}
    if variant_id is None:
        source = config.data["initialization"]["plasticity"]
    else:
        variants = config.data["familiarization"]["plasticity_variants"]
        matches = [variant for variant in variants if variant["id"] == variant_id]
        if len(matches) != 1:
            raise ValueError(f"Unknown plasticity variant: {variant_id}")
        source = matches[0]["multipliers"]
    return {pathway: float(source[pathway]) for pathway in pathways}


def grid_warmup_velocities(
    model_cfg: dict[str, Any], seed: int | None = None
) -> np.ndarray:
    """Return one warm-up velocity per grid module.

    Older configurations contain one shared ``warmup_velocity``. The optional
    three-row form reproduces the warm-up in ``main_run_prev/plas_init.py``.
    """

    if model_cfg.get("grid_warmup_velocity_mode") == "seeded_small":
        if seed is None:
            raise ValueError("A seed is required for seeded grid warm-up velocities")
        maximum = float(model_cfg["grid_warmup_velocity_max"])
        if not 0.0 < maximum <= 0.01:
            raise ValueError("grid_warmup_velocity_max must be in (0, 0.01]")
        # A separate stream keeps velocity sampling independent of the
        # background grid-activity draws while retaining network replay.
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
        velocities = rng.uniform(-maximum, maximum, size=(3, 2))
    elif "grid_warmup_velocities" in model_cfg:
        velocities = np.asarray(model_cfg["grid_warmup_velocities"], dtype=float)
    else:
        shared = np.asarray(model_cfg["warmup_velocity"], dtype=float)
        velocities = np.repeat(shared[None, :], 3, axis=0)
    if velocities.shape != (3, 2) or not np.all(np.isfinite(velocities)):
        raise ValueError("Grid warm-up velocities must be a finite 3 x 2 array")
    return velocities


def run_session(
    config: PipelineConfig,
    env: dict[str, Any],
    positions: np.ndarray,
    state: NetworkState,
    *,
    network_id: int,
    activity_seed: int | None,
    plasticity_enabled: bool,
    variant_id: str | None = None,
    record: bool = True,
    grid_record_indices: np.ndarray | None = None,
    record_place: bool = True,
) -> SimulationResult:
    cfg = config.data["model"]
    state = state.copy()
    grid_model = GridAttractor(cfg)
    if state.place_activity is None or state.grid_activity is None:
        if activity_seed is None:
            raise ValueError("An activity seed is required for the first network session")
        grids = list(initial_grid_activity(cfg, activity_seed))
        place = np.zeros(int(cfg["n_place"]), dtype=np.float64)
        warmup_steps = int(cfg["warmup_steps"])
        warmup_velocities = grid_warmup_velocities(cfg, activity_seed)
        for _ in range(warmup_steps):
            for module in range(3):
                grids[module] = grid_model.update(
                    grids[module], warmup_velocities[module], module, 0.0
                )
    else:
        grids = [activity.copy() for activity in state.grid_activity]
        place = state.place_activity.copy()

    dt = float(cfg["dt_s"])
    duration_steps = int(round(float(cfg["duration_s"]) / dt))
    n_steps = min(duration_steps - 2, len(positions) - 1)
    velocity = np.diff(positions[: n_steps + 1].astype(np.float64), axis=0) / dt
    record_every = int(cfg["record_every_steps"])
    n_records = (n_steps + record_every - 1) // record_every
    n_place = int(cfg["n_place"])
    if record:
        sampled = (
            grid_sample_indices(config)
            if grid_record_indices is None
            else np.asarray(grid_record_indices, dtype=np.int64).reshape(-1)
        )
        n_grid_cells = int(cfg["n_grid"]) ** 2
        if (
            sampled.size < 1
            or np.any(sampled < 0)
            or np.any(sampled >= n_grid_cells)
            or len(np.unique(sampled)) != len(sampled)
        ):
            raise ValueError("grid_record_indices must be unique valid grid-sheet indices")
        place_record = np.empty(
            (n_place if record_place else 0, n_records), dtype=np.float32
        )
        grid_record = tuple(
            np.empty((len(sampled), n_records), dtype=np.float32) for _ in range(3)
        )
        position_record = np.empty((n_records, 2), dtype=np.float32)
    else:
        sampled = np.empty(0, dtype=np.int64)
        place_record = np.empty((n_place, 0), dtype=np.float32)
        grid_record = tuple(np.empty((len(sampled), 0), dtype=np.float32) for _ in range(3))
        position_record = np.empty((0, 2), dtype=np.float32)

    tau_place = float(cfg["tau_place_s"])
    place_dtau = dt / tau_place
    inhibition = 0.0
    agent, bvc_cells = _build_bvc(config, env, network_id)
    bvc_activity = np.zeros(int(cfg["n_boundary"]), dtype=np.float64)
    bvc_to_place = np.zeros(n_place, dtype=np.float64)
    plasticity = _plasticity_multipliers(
        config, variant_id, enabled=plasticity_enabled
    )
    update_every = int(cfg["plasticity_every_steps"])
    progress_every = int(cfg.get("progress_every_steps", 100000))
    record_index = 0

    for step in range(n_steps):
        if step % record_every == 0:
            agent.pos = positions[step]
            bvc_cells.update()
            bvc_activity = np.asarray(bvc_cells.get_state(), dtype=float).reshape(-1)
            if not np.all(np.isfinite(bvc_activity)):
                raise FloatingPointError(
                    f"Non-finite BVC activity at step {step} in {env['id']}"
                )
            bvc_activity *= float(cfg.get("bvc_drive_gain", 1.0))
            bvc_to_place = bvc_activity @ state.w_bvc_to_place

            # The legacy evaluation scripts record the current PC/GC state at
            # position[step] before integrating velocity[step] toward the next
            # position. Keeping that order avoids a one-step activity/position
            # offset, which is especially important for lagged TE analyses.
            if record:
                if record_place:
                    place_record[:, record_index] = place
                for module in range(3):
                    grid_record[module][:, record_index] = grids[module].ravel()[sampled]
                position_record[record_index] = positions[step]
                record_index += 1

        place_to_grid_inputs = [
            state.w_place_to_grid[module] @ place for module in range(3)
        ]
        for module in range(3):
            grids[module] = grid_model.update(
                grids[module], velocity[step], module, place_to_grid_inputs[module].reshape(grids[module].shape)
            )
        grid_to_place = sum(
            grids[module].ravel() @ state.w_grid_to_place[module]
            for module in range(3)
        )
        recurrent_place = place @ state.w_place_recurrent
        total_without_inhibition = (
            bvc_to_place
            + float(cfg["grid_to_place_gain"]) * grid_to_place
            + recurrent_place
            - float(cfg["place_threshold"])
        )
        inhibition_mode = cfg.get("place_inhibition_mode", "linear")
        if inhibition_mode == "linear":
            inhibition = -float(cfg["place_global_inhibition"]) * float(place.sum())
        elif inhibition_mode == "target_sparsity":
            inhibition = target_sparsity_inhibition(
                total_without_inhibition,
                inhibition,
                float(cfg["target_place_sparsity"]),
            )
        else:
            raise ValueError(f"Unknown place_inhibition_mode: {inhibition_mode}")
        drive = relu(total_without_inhibition + inhibition)
        place += place_dtau * (drive - place)
        place[place < float(cfg["place_activity_floor"])] = 0.0

        if plasticity_enabled and step % update_every == 0:
            apply_hebbian_update(
                state,
                bvc_activity,
                place,
                tuple(grids),  # type: ignore[arg-type]
                cfg,
                plasticity,
            )

        if progress_every and step and step % progress_every == 0:
            LOGGER.info(
                "simulation progress env=%s step=%d/%d mean_place=%.6g",
                env["id"], step, n_steps, float(place.mean()),
            )

    state.place_activity = place.copy()
    state.grid_activity = tuple(grid.copy() for grid in grids)
    state.validate(cfg, label="session output state", require_activity=True)
    return SimulationResult(
        positions=position_record[:, :record_index],
        place_activity=place_record[:, :record_index],
        grid_activity=tuple(array[:, :record_index] for array in grid_record),
        record_dt_s=dt * record_every,
        state=state,
    )

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .storage import atomic_npz


def relu(value: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, value)


def rotate_velocity(velocity: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    return rotation @ np.asarray(velocity, dtype=np.float64)


class GridAttractor:
    """Refactored numerical core of ``GC_module_v.py`` without global state."""

    def __init__(self, model_cfg: dict[str, Any]):
        self.n = int(model_cfg["n_grid"])
        self.shift = int(model_cfg["grid_kernel_shift"])
        self.velocity_gains = np.asarray(model_cfg["grid_velocity_gains"], dtype=float)
        self.orientations_deg = np.asarray(
            model_cfg["grid_orientations_deg"], dtype=float
        )
        # Explicit velocity rotations control the numerical update.  The
        # orientation list is descriptive when this separate key is present;
        # an observed firing-field angle must still be measured from rate maps.
        self.velocity_rotations_deg = np.asarray(
            model_cfg.get("grid_velocity_rotations_deg", -self.orientations_deg),
            dtype=float,
        )
        self.dtau = float(model_cfg["dt_s"]) / float(model_cfg["tau_grid_s"])
        wavelength = float(model_cfg["grid_kernel_wavelength"])
        gamma_ratio = float(model_cfg["grid_kernel_gamma_ratio"])
        centre_gain = float(model_cfg["grid_kernel_centre_gain"])
        kernels = self._setup_kernels(wavelength, gamma_ratio, centre_gain)
        self.kernel_ffts = tuple(np.fft.fft2(kernel) for kernel in kernels)

        half = self.n // 2
        self.type_left = np.tile([[1.0, 0.0], [0.0, 0.0]], (half, half))
        self.type_right = np.tile([[0.0, 0.0], [0.0, 1.0]], (half, half))
        self.type_up = np.tile([[0.0, 1.0], [0.0, 0.0]], (half, half))
        self.type_down = np.tile([[0.0, 0.0], [1.0, 0.0]], (half, half))

    def _setup_kernels(
        self, wavelength: float, gamma_ratio: float, centre_gain: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        beta = 3.0 / wavelength**2
        gamma = gamma_ratio * beta
        coordinates = np.arange(self.n, dtype=float)
        coordinates = np.where(coordinates <= self.n / 2, coordinates, coordinates - self.n)
        xx, yy = np.meshgrid(coordinates, coordinates)
        base = centre_gain * np.exp(-gamma * (xx**2 + yy**2)) - np.exp(
            -beta * (xx**2 + yy**2)
        )
        left = np.roll(base, -self.shift, axis=1)
        right = np.roll(base, self.shift, axis=1)
        up = np.roll(base, -self.shift, axis=0)
        down = np.roll(base, self.shift, axis=0)
        return left, right, up, down

    def update(
        self,
        activity: np.ndarray,
        velocity: np.ndarray,
        module_index: int,
        place_input: np.ndarray | float,
    ) -> np.ndarray:
        idx = int(module_index)
        velocity = rotate_velocity(velocity, float(self.velocity_rotations_deg[idx]))
        gain = float(self.velocity_gains[idx])
        velocity_field = (
            (1.0 - gain * velocity[0]) * self.type_left
            + (1.0 + gain * velocity[0]) * self.type_right
            + (1.0 - gain * velocity[1]) * self.type_down
            + (1.0 + gain * velocity[1]) * self.type_up
        )
        fft_left, fft_right, fft_up, fft_down = self.kernel_ffts
        total_fft = (
            np.fft.fft2(activity * self.type_left) * fft_left
            + np.fft.fft2(activity * self.type_right) * fft_right
            + np.fft.fft2(activity * self.type_up) * fft_up
            + np.fft.fft2(activity * self.type_down) * fft_down
        )
        recurrent = np.real(np.fft.ifft2(total_fft))
        drive = relu(recurrent + velocity_field + place_input)
        predictor = activity + self.dtau * (drive - activity)
        return activity + 0.5 * self.dtau * (2.0 * drive - activity - predictor)


@dataclass
class NetworkState:
    w_bvc_to_place: np.ndarray
    w_place_recurrent: np.ndarray
    w_grid_to_place: tuple[np.ndarray, np.ndarray, np.ndarray]
    w_place_to_grid: tuple[np.ndarray, np.ndarray, np.ndarray]
    place_activity: np.ndarray | None = None
    grid_activity: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def copy(self) -> "NetworkState":
        return NetworkState(
            self.w_bvc_to_place.copy(),
            self.w_place_recurrent.copy(),
            tuple(matrix.copy() for matrix in self.w_grid_to_place),
            tuple(matrix.copy() for matrix in self.w_place_to_grid),
            None if self.place_activity is None else self.place_activity.copy(),
            (
                None
                if self.grid_activity is None
                else tuple(activity.copy() for activity in self.grid_activity)
            ),
        )

    def validate(
        self,
        model_cfg: dict[str, Any],
        *,
        label: str = "network state",
        require_activity: bool = False,
    ) -> None:
        n_boundary = int(model_cfg["n_boundary"])
        n_place = int(model_cfg["n_place"])
        n_grid_cells = int(model_cfg["n_grid"]) ** 2
        expected = {
            "w_bvc_to_place": (self.w_bvc_to_place, (n_boundary, n_place)),
            "w_place_recurrent": (self.w_place_recurrent, (n_place, n_place)),
        }
        if len(self.w_grid_to_place) != 3 or len(self.w_place_to_grid) != 3:
            raise ValueError(f"{label} must contain exactly three grid modules")
        expected.update({
            f"w_grid_to_place_m{index + 1}": (matrix, (n_grid_cells, n_place))
            for index, matrix in enumerate(self.w_grid_to_place)
        })
        expected.update({
            f"w_place_to_grid_m{index + 1}": (matrix, (n_grid_cells, n_place))
            for index, matrix in enumerate(self.w_place_to_grid)
        })
        for name, (matrix, shape) in expected.items():
            if matrix.shape != shape:
                raise ValueError(
                    f"{label} {name} has shape {matrix.shape}, expected {shape}"
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"{label} {name} contains non-finite values")
            if np.any(matrix < 0.0):
                raise ValueError(f"{label} {name} contains negative weights")
        if not np.allclose(np.diag(self.w_place_recurrent), 0.0, rtol=0.0, atol=1e-14):
            raise ValueError(f"{label} w_place_recurrent diagonal must be zero")
        if (self.place_activity is None) != (self.grid_activity is None):
            raise ValueError(f"{label} must contain both PC and GC activity or neither")
        if require_activity and self.place_activity is None:
            raise ValueError(f"{label} is missing the final PC/GC activity state")
        if self.place_activity is not None and self.grid_activity is not None:
            if self.place_activity.shape != (n_place,):
                raise ValueError(
                    f"{label} place activity has shape {self.place_activity.shape}, "
                    f"expected {(n_place,)}"
                )
            if len(self.grid_activity) != 3 or any(
                activity.shape != (int(model_cfg["n_grid"]), int(model_cfg["n_grid"]))
                for activity in self.grid_activity
            ):
                raise ValueError(f"{label} has invalid grid activity shapes")
            activities = (self.place_activity, *self.grid_activity)
            if any(not np.all(np.isfinite(activity)) for activity in activities):
                raise ValueError(f"{label} contains non-finite activity")
            if any(np.any(activity < 0.0) for activity in activities):
                raise ValueError(f"{label} contains negative activity")

    def save(self, path: Path, compressed: bool = False) -> None:
        arrays: dict[str, np.ndarray] = {
            "w_bvc_to_place": self.w_bvc_to_place,
            "w_place_recurrent": self.w_place_recurrent,
            "w_grid_to_place_m1": self.w_grid_to_place[0],
            "w_grid_to_place_m2": self.w_grid_to_place[1],
            "w_grid_to_place_m3": self.w_grid_to_place[2],
            "w_place_to_grid_m1": self.w_place_to_grid[0],
            "w_place_to_grid_m2": self.w_place_to_grid[1],
            "w_place_to_grid_m3": self.w_place_to_grid[2],
        }
        if self.place_activity is not None and self.grid_activity is not None:
            arrays.update({
                "place_activity": self.place_activity,
                "grid_activity_m1": self.grid_activity[0],
                "grid_activity_m2": self.grid_activity[1],
                "grid_activity_m3": self.grid_activity[2],
            })
        atomic_npz(path, compressed=compressed, **arrays)

    @classmethod
    def load(cls, path: Path) -> "NetworkState":
        with np.load(path, allow_pickle=False) as data:
            has_activity = "place_activity" in data.files
            return cls(
                data["w_bvc_to_place"].astype(np.float64, copy=True),
                data["w_place_recurrent"].astype(np.float64, copy=True),
                tuple(
                    data[f"w_grid_to_place_m{idx}"].astype(np.float64, copy=True)
                    for idx in (1, 2, 3)
                ),
                tuple(
                    data[f"w_place_to_grid_m{idx}"].astype(np.float64, copy=True)
                    for idx in (1, 2, 3)
                ),
                (
                    data["place_activity"].astype(np.float64, copy=True)
                    if has_activity else None
                ),
                (
                    tuple(
                        data[f"grid_activity_m{idx}"].astype(np.float64, copy=True)
                        for idx in (1, 2, 3)
                    )
                    if has_activity else None
                ),
            )


def _balanced_bvc_to_place(
    rng: np.random.Generator, n_boundary: int, n_place: int, inputs_per_place: int
) -> np.ndarray:
    """Reproduce the first batch's whole-graph rejection sampler.

    Start every attempt from the same ordered multiset of BVC stubs, rather
    than shuffling the previous attempt in place. Keep the generator shared
    with the subsequent Gamma weight draws to preserve the first batch's
    random-number sequence. No alternative graph sampler is used here.
    """

    total = n_place * inputs_per_place
    if total % n_boundary:
        raise ValueError(
            "n_place * inputs_per_place must be divisible by n_boundary for balanced connectivity"
        )
    if inputs_per_place > n_boundary:
        raise ValueError("inputs_per_place cannot exceed n_boundary in a simple graph")

    stubs = np.repeat(np.arange(n_boundary), total // n_boundary)
    for _ in range(1000):
        neighbours = rng.permutation(stubs).reshape(n_place, inputs_per_place)
        ordered = np.sort(neighbours, axis=1)
        if np.all(ordered[:, 1:] != ordered[:, :-1]):
            weights = np.zeros((n_boundary, n_place), dtype=np.float64)
            weights[neighbours, np.arange(n_place)[:, None]] = 1.0
            return weights
    raise RuntimeError(
        "Could not create a simple balanced BVC-to-place graph with the "
        "first-batch rejection sampler after 1000 attempts"
    )


def initialize_network(model_cfg: dict[str, Any], seed: int) -> NetworkState:
    rng = np.random.default_rng(int(seed))
    n_boundary = int(model_cfg["n_boundary"])
    n_place = int(model_cfg["n_place"])
    n_grid_cells = int(model_cfg["n_grid"]) ** 2
    inputs_per_place = int(model_cfg["bvc_inputs_per_place"])

    w_bvc_to_place = _balanced_bvc_to_place(
        rng, n_boundary, n_place, inputs_per_place
    )
    gamma_cfg = model_cfg["initial_weight_gamma"]
    multipliers = model_cfg["initial_weight_multipliers"]
    gamma_shape = float(gamma_cfg["shape"])
    gamma_scale = float(gamma_cfg["scale"])
    w_place_recurrent = float(multipliers["place_recurrent"]) * rng.gamma(
        shape=gamma_shape,
        scale=gamma_scale,
        size=(n_place, n_place),
    )
    np.fill_diagonal(w_place_recurrent, 0.0)
    w_grid_to_place = tuple(
        float(multipliers["grid_to_place"]) * rng.gamma(
            shape=gamma_shape,
            scale=gamma_scale,
            size=(n_grid_cells, n_place),
        )
        for _ in range(3)
    )
    w_place_to_grid = tuple(
        float(multipliers["place_to_grid"]) * rng.gamma(
            shape=gamma_shape,
            scale=gamma_scale,
            size=(n_grid_cells, n_place),
        )
        for _ in range(3)
    )
    state = NetworkState(
        w_bvc_to_place,
        w_place_recurrent,
        w_grid_to_place,
        w_place_to_grid,
    )
    state.validate(model_cfg, label="initialized network")
    return state


def initial_grid_activity(
    model_cfg: dict[str, Any], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize all three sheets with the single reviewer activation template.

    On each 64 x 64 sheet, only (32, 15), (32, 32), and (32, 49) start at
    activity 1. All remaining cells have independent seeded background activity
    below 0.01. This is the initial state, not a pattern imposed during updates.
    """
    n = int(model_cfg["n_grid"])
    mode = model_cfg["grid_initialization"]
    if mode != "three_point_seeded_background":
        raise ValueError(
            "Only model.grid_initialization='three_point_seeded_background' "
            "is supported in the reviewer code"
        )
    if n != 64:
        raise ValueError(
            "grid_initialization='three_point_seeded_background' requires model.n_grid=64"
        )
    maximum = float(model_cfg["initial_grid_background_max"])
    if not np.isfinite(maximum) or not 0.0 < maximum <= 0.01:
        raise ValueError("initial_grid_background_max must be in (0, 0.01]")
    rng = np.random.default_rng(int(seed))
    grids = [rng.uniform(0.0, maximum, size=(n, n)) for _ in range(3)]
    rows = np.array([32, 32, 32])
    cols = np.array([15, 32, 49])
    for grid in grids:
        grid[rows, cols] = 1.0
    return tuple(grids)  # type: ignore[return-value]


def target_sparsity_inhibition(
    total_input: np.ndarray,
    previous_inhibition: float,
    target_sparsity: float,
    step: float = 0.001,
) -> float:
    """Generalized version of the legacy search; no fixed 100-cell divisor."""

    def sparsity(values: np.ndarray) -> float:
        denominator = np.mean(values * values)
        return 0.0 if denominator <= 0 else float(np.mean(values) ** 2 / denominator)

    upper = 0.0
    lower = -float(np.max(total_input)) + 1e-12
    current = float(np.clip(previous_inhibition, lower, upper))
    current_delta = sparsity(relu(total_input + current)) - target_sparsity
    if abs(current_delta) < target_sparsity / 50.0:
        return current
    for _ in range(100000):
        positive = min(upper, current + step)
        negative = max(lower, current - step)
        positive_delta = sparsity(relu(total_input + positive)) - target_sparsity
        negative_delta = sparsity(relu(total_input + negative)) - target_sparsity
        if positive_delta * current_delta <= 0:
            return positive
        if negative_delta * current_delta <= 0:
            return negative
        if positive == upper and negative == lower:
            break
        step *= 1.05
    return current


def _normalize_columns(matrix: np.ndarray, target: float) -> None:
    sums = matrix.sum(axis=0, keepdims=True)
    np.divide(target, sums, out=sums, where=sums > 0)
    matrix *= sums


def _normalize_rows(matrix: np.ndarray, target: float) -> None:
    sums = matrix.sum(axis=1, keepdims=True)
    np.divide(target, sums, out=sums, where=sums > 0)
    matrix *= sums


def apply_hebbian_update(
    state: NetworkState,
    bvc_activity: np.ndarray,
    place_activity: np.ndarray,
    grid_activity: tuple[np.ndarray, np.ndarray, np.ndarray],
    model_cfg: dict[str, Any],
    plasticity: dict[str, float],
) -> None:
    """Update complete population vectors (fixes the legacy ``s_p[0]`` bug)."""
    rates = model_cfg["learning_rates"]
    norms = model_cfg["normalization"]

    multiplier = float(plasticity.get("bvc_to_place", 0.0))
    if multiplier:
        state.w_bvc_to_place += (
            float(rates["bvc_to_place"])
            * multiplier
            * np.outer(bvc_activity, place_activity)
        )
        _normalize_columns(state.w_bvc_to_place, float(norms["bvc_to_place_incoming"]))

    multiplier = float(plasticity.get("place_recurrent", 0.0))
    if multiplier:
        state.w_place_recurrent += (
            float(rates["place_recurrent"])
            * multiplier
            * np.outer(place_activity, place_activity)
        )
        np.fill_diagonal(state.w_place_recurrent, 0.0)
        _normalize_columns(
            state.w_place_recurrent, float(norms["place_recurrent_incoming"])
        )
        maximum = norms.get("place_recurrent_max_single")
        if maximum is not None:
            np.minimum(
                state.w_place_recurrent,
                float(maximum),
                out=state.w_place_recurrent,
            )

    grid_to_place_multiplier = float(plasticity.get("grid_to_place", 0.0))
    place_to_grid_multiplier = float(plasticity.get("place_to_grid", 0.0))
    for module in range(3):
        coactivity = np.outer(grid_activity[module].ravel(), place_activity)
        if grid_to_place_multiplier:
            grid_to_place_matrix = state.w_grid_to_place[module]
            grid_to_place_matrix[...] += (
                float(rates["grid_to_place"])
                * grid_to_place_multiplier
                * coactivity
            )
            _normalize_columns(
                grid_to_place_matrix,
                float(norms["grid_to_place_incoming"]),
            )
        if place_to_grid_multiplier:
            place_to_grid_matrix = state.w_place_to_grid[module]
            place_to_grid_matrix[...] += (
                float(rates["place_to_grid"])
                * place_to_grid_multiplier
                * coactivity
            )
            maximum = norms.get("place_to_grid_max_outgoing")
            if maximum is not None:
                outgoing = place_to_grid_matrix.sum(axis=0, keepdims=True)
                factors = np.ones_like(outgoing)
                np.divide(
                    float(maximum),
                    outgoing,
                    out=factors,
                    where=outgoing > float(maximum),
                )
                place_to_grid_matrix *= np.minimum(factors, 1.0)
            _normalize_rows(
                place_to_grid_matrix,
                float(norms["place_to_grid_incoming"]),
            )

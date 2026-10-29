"""Baseline-fitted, frozen population PCA implemented with NumPy only."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import Config
from .io import TrajectoryData


def _as_time_major_2d(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (n_time, n_features)")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} has insufficient samples or features: {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _deterministic_component_signs(components: np.ndarray) -> np.ndarray:
    result = components.copy()
    pivots = np.argmax(np.abs(result), axis=1)
    signs = np.sign(result[np.arange(result.shape[0]), pivots])
    signs[signs == 0] = 1.0
    result *= signs[:, None]
    return result


@dataclass(frozen=True)
class PCAProjector:
    """Immutable PCA transform with deterministic component signs."""

    mean_: np.ndarray
    components_: np.ndarray
    explained_variance_: np.ndarray
    explained_variance_ratio_: np.ndarray
    n_samples_seen_: int

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean_, dtype=np.float64)
        components = np.asarray(self.components_, dtype=np.float64)
        variance = np.asarray(self.explained_variance_, dtype=np.float64)
        ratio = np.asarray(self.explained_variance_ratio_, dtype=np.float64)
        if mean.ndim != 1:
            raise ValueError("PCA mean_ must be one-dimensional")
        if components.ndim != 2 or components.shape[1] != mean.size:
            raise ValueError("PCA components_ must have shape (components, features)")
        if variance.shape != (components.shape[0],) or ratio.shape != variance.shape:
            raise ValueError("PCA variance arrays do not match components_")
        if self.n_samples_seen_ < 2:
            raise ValueError("PCA requires at least two fitted samples")
        if not all(np.all(np.isfinite(a)) for a in (mean, components, variance, ratio)):
            raise ValueError("PCA parameters must be finite")
        for array in (mean, components, variance, ratio):
            array.setflags(write=False)
        object.__setattr__(self, "mean_", mean)
        object.__setattr__(self, "components_", components)
        object.__setattr__(self, "explained_variance_", variance)
        object.__setattr__(self, "explained_variance_ratio_", ratio)

    @property
    def n_components(self) -> int:
        return int(self.components_.shape[0])

    @property
    def n_features_in(self) -> int:
        return int(self.components_.shape[1])

    @classmethod
    def fit(cls, values: np.ndarray, n_components: int) -> "PCAProjector":
        matrix = _as_time_major_2d(values, "PCA input")
        maximum = min(matrix.shape[0] - 1, matrix.shape[1])
        if isinstance(n_components, bool) or not isinstance(n_components, int):
            raise TypeError("n_components must be an integer")
        if n_components < 1 or n_components > maximum:
            raise ValueError(
                f"n_components={n_components} exceeds the fitted rank limit {maximum}"
            )
        mean = matrix.mean(axis=0)
        centered = matrix - mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        components = _deterministic_component_signs(vt[:n_components])
        explained = singular_values[:n_components] ** 2 / (matrix.shape[0] - 1)
        total_variance = np.var(matrix, axis=0, ddof=1).sum()
        if not np.isfinite(total_variance) or total_variance <= 0:
            raise ValueError("PCA cannot be fitted to a population with zero total variance")
        ratio = explained / total_variance
        return cls(
            mean_=mean,
            components_=components,
            explained_variance_=explained,
            explained_variance_ratio_=ratio,
            n_samples_seen_=int(matrix.shape[0]),
        )

    def transform(self, values: np.ndarray, n_components: int | None = None) -> np.ndarray:
        matrix = _as_time_major_2d(values, "PCA transform input")
        if matrix.shape[1] != self.n_features_in:
            raise ValueError(
                f"PCA expected {self.n_features_in} features, received {matrix.shape[1]}"
            )
        count = self.n_components if n_components is None else int(n_components)
        if count < 1 or count > self.n_components:
            raise ValueError(
                f"Requested {count} components; fitted model contains {self.n_components}"
            )
        return (matrix - self.mean_) @ self.components_[:count].T

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.transform(values)


@dataclass(frozen=True)
class PopulationScores:
    """Symmetric PCA scores for one loaded trajectory."""

    place: np.ndarray
    grid: dict[str, np.ndarray]
    n_components: int


@dataclass(frozen=True)
class FrozenPopulationPCA:
    """Place and per-module grid projections fitted for one network baseline."""

    network_id: str
    place: PCAProjector
    grid: dict[str, PCAProjector]
    baseline: dict[str, Any]
    trajectory_ids: tuple[str, ...]
    fitted_sample_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.grid:
            raise ValueError("A population PCA model requires at least one grid module")
        counts = {self.place.n_components, *(model.n_components for model in self.grid.values())}
        if len(counts) != 1:
            raise ValueError("Place and grid projections must contain symmetric dimensions")

    @property
    def max_components(self) -> int:
        return self.place.n_components

    def transform(
        self, trajectory: TrajectoryData, n_components: int | None = None
    ) -> PopulationScores:
        if str(trajectory.network_id) != str(self.network_id):
            raise ValueError(
                f"PCA for network {self.network_id!r} cannot transform network "
                f"{trajectory.network_id!r}"
            )
        if set(trajectory.grid_activity) != set(self.grid):
            raise ValueError(
                "Grid module mismatch between fitted PCA and trajectory: "
                f"{sorted(self.grid)} vs {sorted(trajectory.grid_activity)}"
            )
        count = self.max_components if n_components is None else int(n_components)
        return PopulationScores(
            place=self.place.transform(trajectory.place_activity, count),
            grid={
                module: self.grid[module].transform(trajectory.grid_activity[module], count)
                for module in sorted(self.grid)
            },
            n_components=count,
        )

    def save(self, path: str | Path) -> None:
        """Atomically save the complete frozen projection in one safe NPZ file."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        modules = sorted(self.grid)
        payload: dict[str, np.ndarray] = {}

        def add(prefix: str, model: PCAProjector) -> None:
            payload[f"{prefix}_mean"] = model.mean_
            payload[f"{prefix}_components"] = model.components_
            payload[f"{prefix}_explained_variance"] = model.explained_variance_
            payload[f"{prefix}_explained_variance_ratio"] = model.explained_variance_ratio_
            payload[f"{prefix}_n_samples"] = np.asarray(model.n_samples_seen_, dtype=np.int64)

        add("place", self.place)
        for index, module in enumerate(modules):
            add(f"grid_{index}", self.grid[module])
        metadata = {
            "format_version": 1,
            "network_id": self.network_id,
            "modules": modules,
            "baseline": self.baseline,
            "trajectory_ids": list(self.trajectory_ids),
            "fitted_sample_count": self.fitted_sample_count,
            "metadata": self.metadata,
        }
        payload["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez_compressed(handle, **payload)
            os.replace(temporary_name, destination)
        except Exception:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path) -> "FrozenPopulationPCA":
        source = Path(path)
        with np.load(source, allow_pickle=False) as archive:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            if metadata.get("format_version") != 1:
                raise ValueError("Unsupported frozen PCA format")

            def read(prefix: str) -> PCAProjector:
                return PCAProjector(
                    mean_=np.asarray(archive[f"{prefix}_mean"]),
                    components_=np.asarray(archive[f"{prefix}_components"]),
                    explained_variance_=np.asarray(
                        archive[f"{prefix}_explained_variance"]
                    ),
                    explained_variance_ratio_=np.asarray(
                        archive[f"{prefix}_explained_variance_ratio"]
                    ),
                    n_samples_seen_=int(np.asarray(archive[f"{prefix}_n_samples"]).item()),
                )

            place = read("place")
            grid = {
                module: read(f"grid_{index}")
                for index, module in enumerate(metadata["modules"])
            }
        return cls(
            network_id=str(metadata["network_id"]),
            place=place,
            grid=grid,
            baseline=dict(metadata["baseline"]),
            trajectory_ids=tuple(str(v) for v in metadata["trajectory_ids"]),
            fitted_sample_count=int(metadata["fitted_sample_count"]),
            metadata=dict(metadata.get("metadata", {})),
        )


def _sample_indices(
    lengths: Sequence[int], max_samples: int | None, random_seed: int
) -> list[np.ndarray]:
    total = int(sum(lengths))
    if max_samples is None or max_samples >= total:
        return [np.arange(length, dtype=np.int64) for length in lengths]
    if max_samples < 2:
        raise ValueError("pca_fit_max_samples must be at least two")
    rng = np.random.default_rng(random_seed)
    global_indices = np.sort(rng.choice(total, size=max_samples, replace=False))
    offsets = np.cumsum([0, *lengths])
    return [
        global_indices[(global_indices >= start) & (global_indices < stop)] - start
        for start, stop in zip(offsets[:-1], offsets[1:])
    ]


def _stack_selected(
    arrays: Sequence[np.ndarray], indices: Sequence[np.ndarray], name: str
) -> np.ndarray:
    if len(arrays) != len(indices):
        raise ValueError("Array/index collection length mismatch")
    selected = [array[index] for array, index in zip(arrays, indices) if index.size]
    if not selected:
        raise ValueError(f"No baseline samples selected for {name}")
    return _as_time_major_2d(np.concatenate(selected, axis=0), name)


def fit_population_pca(
    trajectories: Sequence[TrajectoryData],
    *,
    n_components: int,
    max_samples: int | None,
    random_seed: int,
    baseline: Mapping[str, Any],
) -> FrozenPopulationPCA:
    """Fit one network's symmetric PCA projections from declared baseline data."""

    if not trajectories:
        raise ValueError("No baseline trajectories supplied")
    networks = {str(item.network_id) for item in trajectories}
    if len(networks) != 1:
        raise ValueError("fit_population_pca accepts exactly one network")
    expected = {
        "stage": str(baseline["stage"]),
        "environment": str(baseline["environment"]),
        "session": str(baseline["session"]),
    }
    for item in trajectories:
        observed = {
            "stage": str(item.stage),
            "environment": str(item.environment),
            "session": str(item.session),
        }
        if observed != expected:
            raise ValueError(
                "PCA fitting received non-baseline data; expected "
                f"{expected}, observed {observed}"
            )
    modules = set(trajectories[0].grid_activity)
    if any(set(item.grid_activity) != modules for item in trajectories):
        raise ValueError("Grid module names must be identical across baseline trajectories")
    lengths = [item.n_time for item in trajectories]
    indices = _sample_indices(lengths, max_samples, random_seed)
    place_matrix = _stack_selected(
        [item.place_activity for item in trajectories], indices, "place baseline"
    )
    place = PCAProjector.fit(place_matrix, n_components)
    grid = {
        module: PCAProjector.fit(
            _stack_selected(
                [item.grid_activity[module] for item in trajectories],
                indices,
                f"grid baseline {module}",
            ),
            n_components,
        )
        for module in sorted(modules)
    }
    return FrozenPopulationPCA(
        network_id=next(iter(networks)),
        place=place,
        grid=grid,
        baseline=dict(baseline),
        trajectory_ids=tuple(str(item.trajectory_id) for item in trajectories),
        fitted_sample_count=int(place_matrix.shape[0]),
        metadata={
            "fit_policy": "familiar_baseline_only",
            "sampling_policy": "shared_uniform_without_replacement",
            "random_seed": int(random_seed),
        },
    )


def fit_baseline_pcas(
    trajectories: Iterable[TrajectoryData], config: Config
) -> dict[str, FrozenPopulationPCA]:
    """Fit and freeze one symmetric population projection per independent network."""

    baseline = config.baseline
    selected: list[TrajectoryData] = []
    all_networks: set[str] = set()
    for item in trajectories:
        all_networks.add(str(item.network_id))
        if (
            str(item.stage) == str(baseline["stage"])
            and str(item.environment) == str(baseline["environment"])
            and str(item.session) == str(baseline["session"])
        ):
            selected.append(item)
    if not selected:
        raise ValueError("No loaded trajectory matches the configured familiar baseline")
    baseline_networks = {str(item.network_id) for item in selected}
    missing = sorted(all_networks.difference(baseline_networks))
    if missing:
        raise ValueError(f"Loaded networks without a familiar baseline: {missing}")
    primary = int(config.representation["primary_components"])
    sensitivity = [int(v) for v in config.representation.get("sensitivity_components", [])]
    fitted_components = max([primary, *sensitivity])
    grouped: dict[str, list[TrajectoryData]] = {}
    for item in selected:
        grouped.setdefault(str(item.network_id), []).append(item)
    return {
        network: fit_population_pca(
            sorted(group, key=lambda item: str(item.trajectory_id)),
            n_components=fitted_components,
            max_samples=config.representation["pca_fit_max_samples"],
            random_seed=int(config.random_seed),
            baseline=baseline,
        )
        for network, group in sorted(grouped.items())
    }


def save_pca_collection(
    models: Mapping[str, FrozenPopulationPCA], directory: str | Path
) -> dict[str, Path]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for index, (network, model) in enumerate(sorted(models.items())):
        path = destination / f"network_{index:04d}_population_pca.npz"
        model.save(path)
        written[str(network)] = path
    index_path = destination / "pca_index.json"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".pca_index.", suffix=".tmp", dir=destination
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {network: path.name for network, path in written.items()},
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        os.replace(temporary_name, index_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return written


def load_pca_collection(directory: str | Path) -> dict[str, FrozenPopulationPCA]:
    source = Path(directory)
    with (source / "pca_index.json").open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if not isinstance(index, dict) or not index:
        raise ValueError("pca_index.json must contain a non-empty object")
    result = {
        str(network): FrozenPopulationPCA.load(source / filename)
        for network, filename in index.items()
    }
    for network, model in result.items():
        if str(model.network_id) != network:
            raise ValueError(f"PCA index/model network mismatch for {network!r}")
    return result

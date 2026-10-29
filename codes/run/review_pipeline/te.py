from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import PipelineConfig
from .simulation import SimulationResult
from .storage import atomic_csv_gz, atomic_npz, atomic_pickle


TE_FIELDS = [
    "network_id", "stage", "environment", "trajectory_id", "branch", "session",
    "module", "embedding", "residualized", "pairing", "behavior", "wall_axis",
    "lag_frames", "lag_ms", "source_history", "target_history", "te_grid_to_place_raw",
    "te_place_to_grid_raw", "te_grid_to_place_corrected", "te_place_to_grid_corrected",
    "directionality_index", "n_segments", "n_scalar_pairs", "mean_segment_frames",
]

TE_DELTA_FIELDS = TE_FIELDS + ["baseline_directionality_index", "delta_directionality_index"]


def _entropy(codes: np.ndarray) -> float:
    if len(codes) == 0:
        return np.nan
    _, counts = np.unique(codes, return_counts=True)
    probability = counts.astype(float) / counts.sum()
    return float(-np.sum(probability * np.log2(probability)))


def _encode_columns(columns: list[np.ndarray], alphabets: list[int]) -> np.ndarray:
    code = np.zeros(len(columns[0]), dtype=np.int64)
    multiplier = 1
    for column, alphabet in zip(columns, alphabets):
        code += column.astype(np.int64) * multiplier
        multiplier *= int(alphabet)
    return code


def discretize_equal_count(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < max(10, n_bins * 2):
        return np.zeros(len(values), dtype=np.int16), 1
    quantiles = np.quantile(values[finite], np.linspace(0, 1, n_bins + 1)[1:-1])
    edges = np.unique(quantiles)
    codes = np.digitize(values, edges, right=False).astype(np.int16)
    codes[~finite] = 0
    return codes, len(edges) + 1


def transfer_entropy_plugin(
    source_codes: np.ndarray,
    target_codes: np.ndarray,
    source_alphabet: int,
    target_alphabet: int,
    *,
    source_history: int,
    target_history: int,
    lag: int,
) -> float:
    length = min(len(source_codes), len(target_codes))
    start = max(target_history, lag + source_history - 1)
    if length - start < 20:
        return np.nan
    time = np.arange(start, length)
    future = target_codes[time]
    target_past = [target_codes[time - offset] for offset in range(1, target_history + 1)]
    source_past = [source_codes[time - lag - offset] for offset in range(source_history)]
    y_past_code = _encode_columns(target_past, [target_alphabet] * target_history)
    x_past_code = _encode_columns(source_past, [source_alphabet] * source_history)
    h_y_past_x_past = _entropy(
        _encode_columns(
            [y_past_code, x_past_code],
            [target_alphabet**target_history, source_alphabet**source_history],
        )
    )
    h_future_y_past = _entropy(
        _encode_columns(
            [future, y_past_code],
            [target_alphabet, target_alphabet**target_history],
        )
    )
    h_y_past = _entropy(y_past_code)
    h_future_y_past_x_past = _entropy(
        _encode_columns(
            [future, y_past_code, x_past_code],
            [
                target_alphabet,
                target_alphabet**target_history,
                source_alphabet**source_history,
            ],
        )
    )
    return h_y_past_x_past + h_future_y_past - h_y_past - h_future_y_past_x_past


def _autocorrelation_time(values: np.ndarray, maximum: int | None = None) -> int:
    values = np.asarray(values, dtype=float)
    values = values - np.mean(values)
    variance = float(np.dot(values, values))
    if variance <= 0:
        return 1
    maximum = maximum or min(len(values) // 3, 1000)
    for lag in range(1, max(2, maximum + 1)):
        correlation = float(np.dot(values[:-lag], values[lag:]) / variance)
        if correlation < np.exp(-1):
            return lag
    return max(1, maximum)


def bias_corrected_te(
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_history: int,
    target_history: int,
    lag: int,
    n_bins: int,
    n_shuffles: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    source_codes, source_alphabet = discretize_equal_count(source, n_bins)
    target_codes, target_alphabet = discretize_equal_count(target, n_bins)
    raw = transfer_entropy_plugin(
        source_codes,
        target_codes,
        source_alphabet,
        target_alphabet,
        source_history=source_history,
        target_history=target_history,
        lag=lag,
    )
    if not np.isfinite(raw) or n_shuffles <= 0:
        return raw, raw
    minimum_shift = max(
        _autocorrelation_time(source),
        lag + source_history,
        target_history,
    )
    valid = np.arange(minimum_shift, len(source_codes) - minimum_shift + 1)
    if len(valid) == 0:
        return raw, np.nan
    replace = len(valid) < n_shuffles
    shifts = rng.choice(valid, size=n_shuffles, replace=replace)
    null_values = []
    for shift in shifts:
        shifted = np.roll(source_codes, int(shift))
        null_values.append(
            transfer_entropy_plugin(
                shifted,
                target_codes,
                source_alphabet,
                target_alphabet,
                source_history=source_history,
                target_history=target_history,
                lag=lag,
            )
        )
    return raw, float(raw - np.nanmean(null_values))


def behavioral_design(
    positions_m: np.ndarray, width_m: float, height_m: float, dt_s: float
) -> np.ndarray:
    position = np.asarray(positions_m, dtype=float)
    velocity = np.gradient(position, dt_s, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    heading = np.arctan2(velocity[:, 1], velocity[:, 0])
    x = position[:, 0] / width_m
    y = position[:, 1] / height_m
    design = np.column_stack(
        [
            np.ones(len(position)),
            x,
            y,
            x * x,
            y * y,
            x * y,
            speed,
            np.sin(heading),
            np.cos(heading),
            x,
            1.0 - x,
            y,
            1.0 - y,
            np.minimum.reduce([x, 1.0 - x, y, 1.0 - y]),
        ]
    )
    for column in range(1, design.shape[1]):
        mean = np.mean(design[:, column])
        standard_deviation = np.std(design[:, column])
        design[:, column] = (design[:, column] - mean) / max(standard_deviation, 1e-12)
    return design


def residualize(activity: np.ndarray, design: np.ndarray, ridge: float) -> np.ndarray:
    y = np.asarray(activity, dtype=float).T
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return (y - design @ coefficients).T.astype(np.float32)


@dataclass
class PCABasis:
    mean: np.ndarray
    scale: np.ndarray
    components: np.ndarray

    @classmethod
    def fit(cls, samples: np.ndarray, n_components: int, maximum_fit_samples: int, seed: int):
        rng = np.random.default_rng(seed)
        if len(samples) > maximum_fit_samples:
            indices = np.sort(rng.choice(len(samples), maximum_fit_samples, replace=False))
            fit_samples = samples[indices]
        else:
            fit_samples = samples
        mean = np.mean(fit_samples, axis=0)
        scale = np.std(fit_samples, axis=0)
        scale[scale < 1e-12] = 1.0
        standardized = (fit_samples - mean) / scale
        _, _, right = np.linalg.svd(standardized, full_matrices=False)
        return cls(mean, scale, right[:n_components])

    def transform(self, samples: np.ndarray) -> np.ndarray:
        return ((samples - self.mean) / self.scale) @ self.components.T


class ProjectionBundle:
    def __init__(self, bases: dict[tuple[bool, int], PCABasis], umap_models: dict[tuple[bool, int], Any]):
        self.bases = bases
        self.umap_models = umap_models

    @classmethod
    def fit(
        cls,
        config: PipelineConfig,
        results: list[SimulationResult],
        env: dict[str, Any],
        seed: int,
    ) -> "ProjectionBundle":
        te_cfg = config.data["analysis"]["te"]
        n_components = int(te_cfg["pca_components"])
        maximum = int(te_cfg["pca_fit_max_samples"])
        ridge = float(te_cfg["residual_ridge"])
        residual_grids: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for result in results:
            design = behavioral_design(
                result.positions,
                float(env["width_m"]),
                float(env["height_m"]),
                result.record_dt_s,
            )
            residual_grids.append(
                tuple(residualize(grid, design, ridge) for grid in result.grid_activity)
            )
        bases: dict[tuple[bool, int], PCABasis] = {}
        umap_models: dict[tuple[bool, int], Any] = {}
        for residualized in (False, True):
            source_results = residual_grids if residualized else [result.grid_activity for result in results]
            for module in range(3):
                samples = np.concatenate(
                    [grid_tuple[module].T for grid_tuple in source_results], axis=0
                )
                basis_seed = int(seed + 100 * int(residualized) + module)
                basis = PCABasis.fit(samples, n_components, maximum, basis_seed)
                bases[(residualized, module)] = basis
                if "umap" in te_cfg["embeddings"]:
                    try:
                        import umap
                    except ImportError as exc:
                        raise RuntimeError("umap-learn is required for the configured UMAP control") from exc
                    pca_samples = basis.transform(samples)
                    if len(pca_samples) > int(te_cfg["umap_fit_max_samples"]):
                        rng = np.random.default_rng(basis_seed)
                        indices = np.sort(
                            rng.choice(
                                len(pca_samples),
                                int(te_cfg["umap_fit_max_samples"]),
                                replace=False,
                            )
                        )
                        pca_samples = pca_samples[indices]
                    model = umap.UMAP(
                        n_components=2,
                        n_neighbors=int(te_cfg["umap_n_neighbors"]),
                        min_dist=float(te_cfg["umap_min_dist"]),
                        random_state=basis_seed,
                        transform_seed=basis_seed,
                    ).fit(pca_samples)
                    umap_models[(residualized, module)] = model
        return cls(bases, umap_models)

    def save(self, directory: Path) -> list[Path]:
        arrays: dict[str, np.ndarray] = {}
        for (residualized, module), basis in self.bases.items():
            prefix = f"{'resid' if residualized else 'raw'}_m{module + 1}"
            arrays[f"{prefix}_mean"] = basis.mean
            arrays[f"{prefix}_scale"] = basis.scale
            arrays[f"{prefix}_components"] = basis.components
        basis_path = directory / "projection_basis.npz"
        atomic_npz(basis_path, compressed=True, **arrays)
        outputs = [basis_path]
        if self.umap_models:
            umap_path = directory / "umap_models.pkl"
            atomic_pickle(umap_path, self.umap_models)
            outputs.append(umap_path)
        return outputs

    @classmethod
    def load(cls, directory: Path) -> "ProjectionBundle":
        bases: dict[tuple[bool, int], PCABasis] = {}
        with np.load(directory / "projection_basis.npz", allow_pickle=False) as data:
            for residualized in (False, True):
                for module in range(3):
                    prefix = f"{'resid' if residualized else 'raw'}_m{module + 1}"
                    bases[(residualized, module)] = PCABasis(
                        data[f"{prefix}_mean"],
                        data[f"{prefix}_scale"],
                        data[f"{prefix}_components"],
                    )
        umap_path = directory / "umap_models.pkl"
        if umap_path.exists():
            import pickle

            with umap_path.open("rb") as handle:
                umap_models = pickle.load(handle)
        else:
            umap_models = {}
        return cls(bases, umap_models)

    def transform(
        self, activity: np.ndarray, module: int, residualized: bool, embedding: str
    ) -> np.ndarray:
        pca = self.bases[(residualized, module)].transform(activity.T)
        if embedding == "pca":
            return pca.T
        if embedding == "umap":
            return np.asarray(self.umap_models[(residualized, module)].transform(pca)).T
        raise ValueError(f"Unknown embedding: {embedding}")


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    behavior: str
    wall_axis: str


def classify_segments(
    positions: np.ndarray,
    width_m: float,
    height_m: float,
    boundary_m: float,
    centre_fraction: float,
    minimum_frames: int,
) -> list[Segment]:
    x = positions[:, 0]
    y = positions[:, 1]
    margin_x = (1.0 - centre_fraction) * width_m / 2.0
    margin_y = (1.0 - centre_fraction) * height_m / 2.0
    centre = (
        (x >= margin_x) & (x <= width_m - margin_x)
        & (y >= margin_y) & (y <= height_m - margin_y)
    )
    distances = np.column_stack([x, width_m - x, y, height_m - y])
    boundary = np.min(distances, axis=1) <= boundary_m
    zone = np.zeros(len(positions), dtype=np.int8)
    zone[centre] = 1
    zone[boundary] = 2
    anchors = np.flatnonzero(zone != 0)
    segments: list[Segment] = []
    if len(anchors) < 2:
        return segments
    previous = anchors[0]
    previous_zone = zone[previous]
    for current in anchors[1:]:
        current_zone = zone[current]
        if current_zone == previous_zone:
            previous = current
            continue
        if current_zone == 2 and previous_zone == 1:
            start, end, behavior, wall_index = previous, current + 1, "approach", int(np.argmin(distances[current]))
        elif current_zone == 1 and previous_zone == 2:
            start, end, behavior, wall_index = previous, current + 1, "departure", int(np.argmin(distances[previous]))
        else:
            previous, previous_zone = current, current_zone
            continue
        if end - start >= minimum_frames:
            segments.append(
                Segment(start, end, behavior, "x" if wall_index in (0, 1) else "y")
            )
        previous, previous_zone = current, current_zone
    return segments


def _directionality(grid_to_place: float, place_to_grid: float) -> float:
    denominator = abs(grid_to_place) + abs(place_to_grid)
    return (grid_to_place - place_to_grid) / denominator if denominator > 0 else 0.0


def _segment_te(
    place: np.ndarray,
    latent: np.ndarray,
    segment: Segment,
    te_cfg: dict[str, Any],
    lag: int,
    source_history: int,
    target_history: int,
    seed: int,
) -> tuple[float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    g2p_raw: list[float] = []
    p2g_raw: list[float] = []
    g2p_corrected: list[float] = []
    p2g_corrected: list[float] = []
    for place_cell in range(place.shape[0]):
        p = place[place_cell, segment.start : segment.end]
        for latent_dimension in range(latent.shape[0]):
            g = latent[latent_dimension, segment.start : segment.end]
            raw, corrected = bias_corrected_te(
                g,
                p,
                source_history=source_history,
                target_history=target_history,
                lag=lag,
                n_bins=int(te_cfg["n_bins"]),
                n_shuffles=int(te_cfg["n_shuffles"]),
                rng=rng,
            )
            g2p_raw.append(raw)
            g2p_corrected.append(corrected)
            raw, corrected = bias_corrected_te(
                p,
                g,
                source_history=source_history,
                target_history=target_history,
                lag=lag,
                n_bins=int(te_cfg["n_bins"]),
                n_shuffles=int(te_cfg["n_shuffles"]),
                rng=rng,
            )
            p2g_raw.append(raw)
            p2g_corrected.append(corrected)
    mean_g2p_raw = float(np.nanmean(g2p_raw))
    mean_p2g_raw = float(np.nanmean(p2g_raw))
    mean_g2p_corrected = float(np.nanmean(g2p_corrected))
    mean_p2g_corrected = float(np.nanmean(p2g_corrected))
    return (
        mean_g2p_raw,
        mean_p2g_raw,
        mean_g2p_corrected,
        mean_p2g_corrected,
        _directionality(mean_g2p_corrected, mean_p2g_corrected),
    )


def compute_te_group(
    config: PipelineConfig,
    results: list[SimulationResult],
    env: dict[str, Any],
    identities: list[dict[str, Any]],
    projection: ProjectionBundle,
    seed: int,
    output_path: Path,
) -> list[dict[str, Any]]:
    te_cfg = config.data["analysis"]["te"]
    if not bool(te_cfg["enabled"]):
        return []
    residuals = []
    for result in results:
        design = behavioral_design(
            result.positions,
            float(env["width_m"]),
            float(env["height_m"]),
            result.record_dt_s,
        )
        residuals.append({
            "place": residualize(result.place_activity, design, float(te_cfg["residual_ridge"])),
            "grid": tuple(
                residualize(grid, design, float(te_cfg["residual_ridge"]))
                for grid in result.grid_activity
            ),
        })
    rows: list[dict[str, Any]] = []
    n_jobs = max(1, int(te_cfg.get("n_jobs", 1)))
    executor = ThreadPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    pairings = ["within"]
    if bool(te_cfg.get("cross_trajectory_null", True)) and len(results) >= 2:
        pairings.append("cross_trajectory")
    for trajectory_index, result in enumerate(results):
        segments = classify_segments(
            result.positions,
            float(env["width_m"]),
            float(env["height_m"]),
            float(te_cfg["boundary_zone_m"]),
            float(te_cfg["centre_fraction"]),
            int(te_cfg["minimum_segment_frames"]),
        )
        for residualized in (False, True):
            place = residuals[trajectory_index]["place"] if residualized else result.place_activity
            for module in range(3):
                own_grid = residuals[trajectory_index]["grid"][module] if residualized else result.grid_activity[module]
                for embedding in te_cfg["embeddings"]:
                    own_latent = projection.transform(own_grid, module, residualized, embedding)
                    for pairing in pairings:
                        if pairing == "within":
                            latent = own_latent
                        else:
                            other_index = (trajectory_index + 1) % len(results)
                            other_grid = (
                                residuals[other_index]["grid"][module]
                                if residualized else results[other_index].grid_activity[module]
                            )
                            latent = projection.transform(other_grid, module, residualized, embedding)
                        for lag in [int(value) for value in te_cfg["lag_frames"]]:
                            for source_history in [int(value) for value in te_cfg["source_history"]]:
                                for target_history in [int(value) for value in te_cfg["target_history"]]:
                                    for behavior in ("approach", "departure"):
                                        for wall_axis in ("x", "y"):
                                            selected = [
                                                segment for segment in segments
                                                if segment.behavior == behavior and segment.wall_axis == wall_axis
                                            ]
                                            def calculate_segment(item: tuple[int, Segment]):
                                                segment_index, segment = item
                                                return _segment_te(
                                                    place,
                                                    latent,
                                                    segment,
                                                    te_cfg,
                                                    lag,
                                                    source_history,
                                                    target_history,
                                                    seed + 1000003 * trajectory_index + 1009 * module + segment_index,
                                                )

                                            indexed_segments = list(enumerate(selected))
                                            values = (
                                                list(executor.map(calculate_segment, indexed_segments))
                                                if executor is not None
                                                else [calculate_segment(item) for item in indexed_segments]
                                            )
                                            if not values:
                                                continue
                                            array = np.asarray(values, dtype=float)
                                            means = np.nanmean(array, axis=0)
                                            rows.append({
                                                **identities[trajectory_index],
                                                "module": module + 1,
                                                "embedding": embedding,
                                                "residualized": residualized,
                                                "pairing": pairing,
                                                "behavior": behavior,
                                                "wall_axis": wall_axis,
                                                "lag_frames": lag,
                                                "lag_ms": lag * result.record_dt_s * 1000.0,
                                                "source_history": source_history,
                                                "target_history": target_history,
                                                "te_grid_to_place_raw": means[0],
                                                "te_place_to_grid_raw": means[1],
                                                "te_grid_to_place_corrected": means[2],
                                                "te_place_to_grid_corrected": means[3],
                                                "directionality_index": means[4],
                                                "n_segments": len(selected),
                                                "n_scalar_pairs": place.shape[0] * latent.shape[0],
                                                "mean_segment_frames": float(np.mean([s.end - s.start for s in selected])),
                                            })
    if executor is not None:
        executor.shutdown(wait=True)
    atomic_csv_gz(output_path, rows, TE_FIELDS)
    return rows


def compare_te_to_baseline(
    current_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, str]],
    output_path: Path,
) -> list[dict[str, Any]]:
    keys = [
        "trajectory_id", "module", "embedding", "residualized", "pairing",
        "behavior", "wall_axis", "lag_frames", "source_history", "target_history",
    ]
    baseline_by_key = {
        tuple(str(row[key]) for key in keys): row for row in baseline_rows
    }
    output: list[dict[str, Any]] = []
    for row in current_rows:
        key = tuple(str(row[key]) for key in keys)
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        baseline_di = float(baseline["directionality_index"])
        output.append({
            **row,
            "baseline_directionality_index": baseline_di,
            "delta_directionality_index": float(row["directionality_index"]) - baseline_di,
        })
    atomic_csv_gz(output_path, output, TE_DELTA_FIELDS)
    return output

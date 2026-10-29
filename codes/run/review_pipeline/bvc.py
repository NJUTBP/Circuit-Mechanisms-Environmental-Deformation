from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .paths import network_root
from .seeds import SeedBook
from .storage import atomic_json, atomic_npz, sha256_file


@dataclass(frozen=True)
class BVCTuning:
    distance_m: np.ndarray
    angle_deg: np.ndarray
    sigma_angle_deg: np.ndarray


def bvc_tuning_path(config: PipelineConfig, network_id: int) -> Path:
    return network_root(config, network_id) / "init" / "bvc_tuning.npz"


def bvc_tuning_metadata_path(config: PipelineConfig, network_id: int) -> Path:
    return network_root(config, network_id) / "init" / "bvc_tuning.json"


def bvc_tuning_seeds(config: PipelineConfig, network_id: int) -> dict[str, int]:
    """Return independent streams so BVC sampling never shifts weight draws."""

    if config.data["model"]["bvc_tuning"]["source"] != "generated_per_network":
        return {}
    book = SeedBook(config.master_seed)
    return {
        "bvc_distance": book.seed("bvc_distance", network_id=network_id),
        "bvc_angle": book.seed("bvc_angle", network_id=network_id),
        "bvc_angular_width": book.seed("bvc_angular_width", network_id=network_id),
    }


def _validate_arrays(
    config: PipelineConfig,
    distance_m: np.ndarray,
    angle_deg: np.ndarray,
    sigma_angle_deg: np.ndarray,
) -> BVCTuning:
    n_boundary = int(config.data["model"]["n_boundary"])
    distance_m = np.asarray(distance_m, dtype=np.float64).reshape(-1)
    angle_deg = np.asarray(angle_deg, dtype=np.float64).reshape(-1)
    sigma_angle_deg = np.asarray(sigma_angle_deg, dtype=np.float64).reshape(-1)
    if (
        distance_m.size != n_boundary
        or angle_deg.size != n_boundary
        or sigma_angle_deg.size != n_boundary
    ):
        raise ValueError(
            "BVC tuning arrays must each contain model.n_boundary values, "
            f"got distance={distance_m.size}, angle={angle_deg.size}, "
            f"angular_width={sigma_angle_deg.size}, "
            f"expected={n_boundary}"
        )
    if (
        not np.all(np.isfinite(distance_m))
        or not np.all(np.isfinite(angle_deg))
        or not np.all(np.isfinite(sigma_angle_deg))
    ):
        raise ValueError("BVC tuning arrays contain non-finite values")
    if np.any(distance_m < 0):
        raise ValueError("BVC tuning distances must be non-negative")
    if np.any((angle_deg < 0.0) | (angle_deg >= 360.0)):
        raise ValueError("BVC tuning angles must lie in [0, 360) degrees")
    if np.any((sigma_angle_deg <= 0.0) | (sigma_angle_deg > 180.0)):
        raise ValueError("BVC angular widths must lie in (0, 180] degrees")
    return BVCTuning(
        distance_m=distance_m,
        angle_deg=angle_deg,
        sigma_angle_deg=sigma_angle_deg,
    )


def _generate_bvc_tuning(config: PipelineConfig, network_id: int) -> BVCTuning:
    try:
        from scipy.stats import truncnorm
    except ImportError as exc:
        raise RuntimeError(
            "SciPy is required to generate per-network BVC tuning parameters"
        ) from exc

    model_cfg = config.data["model"]
    tuning_cfg = model_cfg["bvc_tuning"]
    distance_cfg = tuning_cfg["distance"]
    angle_cfg = tuning_cfg["angle"]
    width_cfg = tuning_cfg["angular_width"]
    seeds = bvc_tuning_seeds(config, network_id)
    n_boundary = int(model_cfg["n_boundary"])

    distance_rng = np.random.default_rng(seeds["bvc_distance"])
    distance_m = truncnorm.rvs(
        float(distance_cfg["a"]),
        float(distance_cfg["b"]),
        loc=float(distance_cfg["loc_m"]),
        scale=float(distance_cfg["scale_m"]),
        size=n_boundary,
        random_state=distance_rng,
    )
    distance_m = np.asarray(distance_m, dtype=np.float64)
    distance_m *= float(tuning_cfg.get("distance_scale", 1.0))

    angle_rng = np.random.default_rng(seeds["bvc_angle"])
    angle_deg = angle_rng.uniform(
        float(angle_cfg["low_deg"]),
        float(angle_cfg["high_deg"]),
        size=n_boundary,
    )
    width_rng = np.random.default_rng(seeds["bvc_angular_width"])
    low_deg = float(width_cfg["low_deg"])
    sigma_angle_deg = width_rng.uniform(
        low_deg,
        float(width_cfg["high_deg"]),
        size=n_boundary,
    )
    return _validate_arrays(config, distance_m, angle_deg, sigma_angle_deg)


def _load_fixed_bvc_tuning(config: PipelineConfig) -> BVCTuning:
    tuning_cfg = config.data["model"]["bvc_tuning"]
    distance_m = np.load(
        config.input_path("bvc_distance"), allow_pickle=False
    ).astype(np.float64, copy=False)
    distance_m = distance_m * float(tuning_cfg.get("distance_scale", 1.0))
    angle_deg = np.load(
        config.input_path("bvc_angle"), allow_pickle=False
    ).astype(np.float64, copy=False)
    if "bvc_sigma_angle" in config.data["inputs"]:
        sigma_angle_deg = np.load(
            config.input_path("bvc_sigma_angle"), allow_pickle=False
        ).astype(np.float64, copy=False)
    else:
        # Compatibility for the two-array legacy input mode. Formal Methods-based
        # runs use generated_per_network and never take this fallback.
        sigma_angle_deg = np.full(
            int(config.data["model"]["n_boundary"]),
            float(tuning_cfg.get("fixed_sigma_angle_deg", 30.0)),
        )
    return _validate_arrays(config, distance_m, angle_deg, sigma_angle_deg)


def _expected_seeds(config: PipelineConfig, network_id: int) -> dict[str, int]:
    return bvc_tuning_seeds(config, network_id)


def load_bvc_tuning(config: PipelineConfig, network_id: int) -> BVCTuning:
    path = bvc_tuning_path(config, network_id)
    metadata_path = bvc_tuning_metadata_path(config, network_id)
    if not path.is_file() or not metadata_path.is_file():
        raise RuntimeError(
            f"Missing BVC tuning asset for network {network_id}; run init first: {path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read BVC tuning metadata: {metadata_path}") from exc
    expected_seeds = _expected_seeds(config, network_id)
    if (
        metadata.get("run_sha256") not in config.accepted_run_hashes("init")
        or int(metadata.get("network_id", -1)) != int(network_id)
        or metadata.get("source") != config.data["model"]["bvc_tuning"]["source"]
        or metadata.get("seeds", {}) != expected_seeds
        or metadata.get("sha256") != sha256_file(path)
    ):
        raise RuntimeError(
            f"Existing BVC tuning does not match this run/config: {path}"
        )
    with np.load(path, allow_pickle=False) as data:
        if str(data["source"].item()) != metadata["source"]:
            raise RuntimeError(f"BVC tuning source metadata mismatch: {path}")
        if expected_seeds:
            stored_seeds = {
                "bvc_distance": int(data["bvc_distance_seed"].item()),
                "bvc_angle": int(data["bvc_angle_seed"].item()),
                "bvc_angular_width": int(data["bvc_angular_width_seed"].item()),
            }
            if stored_seeds != expected_seeds:
                raise RuntimeError(f"BVC tuning seed metadata mismatch: {path}")
        return _validate_arrays(
            config,
            data["distance_m"],
            data["angle_deg"],
            data["sigma_angle_deg"],
        )


def ensure_bvc_tuning(config: PipelineConfig, network_id: int) -> BVCTuning:
    """Create one immutable BVC population per network, then always reuse it."""

    path = bvc_tuning_path(config, network_id)
    metadata_path = bvc_tuning_metadata_path(config, network_id)
    if path.exists() and metadata_path.exists():
        return load_bvc_tuning(config, network_id)

    source = config.data["model"]["bvc_tuning"]["source"]
    if source == "generated_per_network":
        tuning = _generate_bvc_tuning(config, network_id)
    elif source == "fixed_files":
        tuning = _load_fixed_bvc_tuning(config)
    else:  # Config validation should make this unreachable.
        raise ValueError(f"Unknown BVC tuning source: {source}")

    seeds = _expected_seeds(config, network_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        path,
        compressed=False,
        distance_m=tuning.distance_m,
        angle_deg=tuning.angle_deg,
        sigma_angle_deg=tuning.sigma_angle_deg,
        source=np.array(source),
        bvc_distance_seed=np.array(seeds.get("bvc_distance", -1), dtype=np.int64),
        bvc_angle_seed=np.array(seeds.get("bvc_angle", -1), dtype=np.int64),
        bvc_angular_width_seed=np.array(
            seeds.get("bvc_angular_width", -1), dtype=np.int64
        ),
    )
    tuning_cfg: dict[str, Any] = config.data["model"]["bvc_tuning"]
    atomic_json(
        metadata_path,
        {
            "run_sha256": config.hash,
            "network_id": int(network_id),
            "source": source,
            "seeds": seeds,
            "parameters": tuning_cfg,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "summary": {
                "n_boundary": int(tuning.distance_m.size),
                "distance_min_m": float(np.min(tuning.distance_m)),
                "distance_max_m": float(np.max(tuning.distance_m)),
                "distance_mean_m": float(np.mean(tuning.distance_m)),
                "angle_min_deg": float(np.min(tuning.angle_deg)),
                "angle_max_deg": float(np.max(tuning.angle_deg)),
                "angular_width_min_deg": float(np.min(tuning.sigma_angle_deg)),
                "angular_width_max_deg": float(np.max(tuning.sigma_angle_deg)),
                "angular_width_mean_deg": float(np.mean(tuning.sigma_angle_deg)),
            },
        },
    )
    return tuning

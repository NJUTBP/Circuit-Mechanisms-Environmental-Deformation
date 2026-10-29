"""Compute familiarization source tables with the adopted grid estimator.

Session 0 is initial exposure trajectory 00. Sessions 1–9 are intact-plasticity
familiarization trajectory 00. The independent replicate is the network.
Manuscript figure layout is deliberately separate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

import familiarization_properties as methods
from initial_exposure_rescaling import load_selected_maps
from grid_rescaling import compute_recommended_rescale_batch


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "run/outputs/independent_networks/networks"
METHODS = ROOT / "outputs/familiarization_properties/metrics/cell_level_values.csv.gz"
OUTPUT = ROOT / "outputs/familiarization_rescaling"
ENVS = {"compressed_80": 80, "elongated_120": 120}
SESSIONS = tuple(range(10))
MODULES = (1, 2, 3)
CACHE_VERSION = "familiarization_grid_five_edge_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_signature(paths: list[Path]) -> str:
    records = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
    return hashlib.sha256(json.dumps(records).encode()).hexdigest()


def baseline_path(data_root: Path, network_id: int) -> Path:
    return data_root / f"network_{network_id:03d}" / "baseline/fam_100/trajectory_00/transient_activity.npz"


def session_path(data_root: Path, network_id: int, environment: str, session: int) -> Path:
    network = data_root / f"network_{network_id:03d}"
    if session == 0:
        return network / "initial_exposure" / environment / "trajectory_00/transient_activity.npz"
    return network / "familiarization" / environment / "intact/trajectory_00" / f"session_{session:02d}/transient_activity.npz"


def cache_path(output: Path, network_id: int) -> Path:
    return output / "cache" / f"network_{network_id:03d}.npz"


def compute_network(data_root: Path, output: Path, network_id: int, edge_positions_per_side: int = 5) -> str:
    source_paths = [baseline_path(data_root, network_id)] + [
        session_path(data_root, network_id, environment, session)
        for environment in ENVS for session in SESSIONS
    ]
    missing = [path for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    signature = source_signature(source_paths)
    cache = cache_path(output, network_id)
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved["cache_version"]) == f"{CACHE_VERSION}_edge{edge_positions_per_side}" and str(saved["source_signature"]) == signature:
                    return str(cache)
        except (OSError, ValueError, KeyError):
            pass

    baseline = load_selected_maps(source_paths[0], 100)
    arrays: dict[str, np.ndarray] = {
        "cache_version": np.array(f"{CACHE_VERSION}_edge{edge_positions_per_side}"),
        "source_signature": np.array(signature),
    }
    for environment, width_cm in ENVS.items():
        for session in SESSIONS:
            current = load_selected_maps(session_path(data_root, network_id, environment, session), width_cm)
            for module in MODULES:
                factor, correlation = compute_recommended_rescale_batch(
                    baseline[module], current[module], width_cm,
                    edge_positions_per_side=edge_positions_per_side, bin_cm=2.5,
                )
                prefix = f"{environment}_s{session:02d}_m{module}"
                arrays[f"factor_{prefix}"] = factor
                arrays[f"corr_{prefix}"] = correlation
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f".{cache.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, cache)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(cache)


def read_grid_cells(output: Path, network_ids: list[int], edge_positions_per_side: int = 5) -> pd.DataFrame:
    rows = []
    for network_id in network_ids:
        with np.load(cache_path(output, network_id), allow_pickle=False) as saved:
            for environment, width_cm in ENVS.items():
                for session in SESSIONS:
                    for module in MODULES:
                        prefix = f"{environment}_s{session:02d}_m{module}"
                        factors = np.asarray(saved[f"factor_{prefix}"], float)
                        correlations = np.asarray(saved[f"corr_{prefix}"], float)
                        if factors.shape != (100,) or correlations.shape != (100,):
                            raise ValueError(f"Wrong cell count in {network_id}/{prefix}")
                        for cell_id, (factor, correlation) in enumerate(zip(factors, correlations)):
                            rows.append({
                                "network_id": network_id, "environment": environment,
                                "length_cm": width_cm, "session": session,
                                "module": module, "cell_id": cell_id,
                                f"rescale_factor_edge{edge_positions_per_side}m": factor,
                                f"rescale_best_correlation_edge{edge_positions_per_side}m": correlation,
                            })
    return pd.DataFrame(rows)


def merge_metrics(source: Path, grid: pd.DataFrame, network_ids: list[int], edge_positions_per_side: int = 5) -> pd.DataFrame:
    cells = pd.read_csv(source)
    cells = cells.loc[cells.network_id.isin(network_ids)].copy()
    keys = ["network_id", "environment", "session", "module", "cell_id"]
    if cells.duplicated(keys).any() or grid.duplicated(keys).any():
        raise ValueError("Cell keys are not unique")
    if len(cells) != len(network_ids) * len(ENVS) * len(SESSIONS) * 400:
        raise ValueError("Methods source has unexpected cell count")
    merged = cells.merge(grid.drop(columns="length_cm"), on=keys, how="left", validate="one_to_one")
    is_grid = merged.population.eq("grid").to_numpy()
    factor_column = f"rescale_factor_edge{edge_positions_per_side}m"
    correlation_column = f"rescale_best_correlation_edge{edge_positions_per_side}m"
    factor = merged[factor_column].to_numpy(float)
    correlation = merged[correlation_column].to_numpy(float)
    if len(grid) != int(is_grid.sum()) or not np.isfinite(factor[is_grid]).all():
        raise ValueError(f"Edge{edge_positions_per_side}-M rescale coverage incomplete")
    if np.isfinite(factor[~is_grid]).any():
        raise ValueError("Grid estimator unexpectedly joined to place cells")
    merged.loc[is_grid, "rescale_factor"] = factor[is_grid]
    merged.loc[is_grid, "rescale_best_correlation"] = correlation[is_grid]
    return merged.drop(columns=[factor_column, correlation_column])


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(edge_positions_per_side: int = 5) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--methods-cell-source", type=Path, default=METHODS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--network-ids", type=int, nargs="*", default=list(range(8)))
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    network_ids = sorted(set(args.network_ids))
    data_root = args.data_root.resolve()
    if edge_positions_per_side != 5:
        raise ValueError("This release includes only the adopted five-edge estimator")
    alias = f"Edge{edge_positions_per_side}-M"
    output = args.output.resolve()
    source = args.methods_cell_source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(compute_network, data_root, output, network_id, edge_positions_per_side) for network_id in network_ids]
        for index, future in enumerate(as_completed(futures), 1):
            print(f"{alias} familiarization cache {index}/{len(futures)}: {future.result()}", flush=True)

    grid = read_grid_cells(output, network_ids, edge_positions_per_side)
    cells = merge_metrics(source, grid, network_ids, edge_positions_per_side)
    network, group, counts = methods.aggregate(cells)
    differences, tests = methods.endpoint_statistics(network)
    metrics = output / "metrics"
    save_table(grid, metrics / f"grid_edge{edge_positions_per_side}m_cell_level.csv.gz")
    save_table(cells, metrics / "cell_level_values.csv.gz")
    save_table(network, metrics / "network_level_values.csv")
    save_table(group, metrics / "group_summary_network_unit.csv")
    save_table(counts, metrics / "finite_cell_counts.csv")
    save_table(differences, metrics / "endpoint_network_differences.csv")
    save_table(tests, metrics / "endpoint_statistics.csv")
    provenance = {
        "alias": alias, "analysis": "v4 familiarization Figure 5",
        "network_ids": network_ids, "n_networks": len(network_ids),
        "environments": ENVS, "sessions": list(SESSIONS),
        "session_0": "initial exposure trajectory 00",
        "sessions_1_to_9": "intact-plasticity familiarization trajectory 00",
        "matched_baseline": "same network/cell familiar 100-cm trajectory 00",
        "grid_rescaling": f"2.5-cm occupancy-normalized maps, normalized-mask Gaussian sigma=1.5 bins, pairwise finite, Methods physical scale candidates, first and last {edge_positions_per_side} complete-overlap x positions",
        "unchanged_metrics": "Grid phase and place field/phase values retained from Methods-profile per-cell source",
        "place_phase_direction": "Methods opposing-movement absolute lag is nondirectional; plotted delta is current magnitude minus matched familiar magnitude",
        "aggregation": "one trajectory per session; cells within network, then network mean and 95% t CI across networks",
        "endpoint_statistics": "session 9 minus session 0 within each network; two-sided paired-network t tests and BH within figure",
        "data_root": str(data_root), "methods_cell_source": str(source),
        "methods_cell_source_sha256": sha256(source),
        "cell_rows": len(cells), "grid_rows": len(grid),
        "python": sys.executable, "python_version": platform.python_version(),
        "scipy_version": __import__("scipy").__version__,
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), Path(methods.__file__),
            ROOT / "local_network_analysis/initial_exposure_rescaling.py",
            ROOT / "local_network_analysis/grid_rescaling.py",
        )},
    }
    (output / "analysis_qa_and_provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{alias} familiarization analysis complete: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

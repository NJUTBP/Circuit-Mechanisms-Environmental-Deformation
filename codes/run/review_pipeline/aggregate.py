from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .paths import (
    evaluation_group_dir,
    familiarization_root,
    familiarization_session_dir,
)
from .seeds import derive_seed
from .storage import manifest_is_complete


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    order = finite_indices[np.argsort(p_values[finite_indices])]
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def _completed_summary_files(config: PipelineConfig) -> list[Path]:
    """Require every configured task before producing a cross-network aggregate."""

    from .stages import audit_initialization

    audit_initialization(config)
    summaries: list[Path] = []
    baseline_environment = config.baseline_environment["id"]
    te_cfg = config.data["analysis"]["te"]
    te_enabled = bool(te_cfg["enabled"])
    te_environments = set(te_cfg.get("environments", []))

    for network_id in config.network_ids:
        baseline_group = evaluation_group_dir(
            config, "baseline", network_id, baseline_environment
        )
        baseline_required = ["network_summary.csv.gz"]
        for trajectory_id in config.trajectory_ids:
            baseline_required.extend([
                f"trajectory_{trajectory_id:02d}/cell_metrics.csv.gz",
                f"trajectory_{trajectory_id:02d}/rate_maps.npz",
                f"trajectory_{trajectory_id:02d}/checkpoint.npz",
            ])
        if te_enabled and baseline_environment in te_environments:
            baseline_required.extend(["te_metrics.csv.gz", "projection_basis.npz"])
            if "umap" in te_cfg["embeddings"]:
                baseline_required.append("umap_models.pkl")
        if not manifest_is_complete(
            baseline_group,
            config.accepted_run_hashes("baseline"),
            baseline_required,
        ):
            raise RuntimeError(
                f"Baseline task is incomplete or corrupted: {baseline_group}"
            )
        summaries.append(baseline_group / "network_summary.csv.gz")

        for environment in config.stage_environments("initial_exposure"):
            group = evaluation_group_dir(
                config, "initial_exposure", network_id, environment
            )
            required = ["network_summary.csv.gz"]
            for trajectory_id in config.trajectory_ids:
                required.extend([
                    f"trajectory_{trajectory_id:02d}/cell_metrics.csv.gz",
                    f"trajectory_{trajectory_id:02d}/rate_maps.npz",
                    f"trajectory_{trajectory_id:02d}/comparison_metrics.csv.gz",
                    f"trajectory_{trajectory_id:02d}/checkpoint.npz",
                ])
            if te_enabled and environment in te_environments:
                required.extend(["te_metrics.csv.gz", "te_delta.csv.gz"])
            if not manifest_is_complete(group, config.hash, required):
                raise RuntimeError(
                    f"Initial-exposure task is incomplete or corrupted: {group}"
                )
            summaries.append(group / "network_summary.csv.gz")

        for environment in config.stage_environments("familiarization"):
            for variant in config.data["familiarization"]["plasticity_variants"]:
                root = familiarization_root(
                    config, network_id, environment, str(variant["id"])
                )
                for session in range(
                    1, int(config.data["familiarization"]["sessions"]) + 1
                ):
                    session_group = root / "session_outputs" / f"session_{session:02d}"
                    required = ["network_summary.csv.gz"]
                    for trajectory_id in config.familiarization_trajectory_ids:
                        run_dir = familiarization_session_dir(
                            config,
                            network_id,
                            environment,
                            str(variant["id"]),
                            trajectory_id,
                            session,
                        )
                        required.extend([
                            os.path.relpath(run_dir / "checkpoint.npz", session_group),
                            os.path.relpath(run_dir / "cell_metrics.csv.gz", session_group),
                            os.path.relpath(run_dir / "rate_maps.npz", session_group),
                            os.path.relpath(
                                run_dir / "comparison_metrics.csv.gz", session_group
                            ),
                        ])
                    if (
                        te_enabled
                        and environment in te_environments
                        and session
                        in {
                            int(value)
                            for value in te_cfg.get("familiarization_sessions", [])
                        }
                    ):
                        required.extend(["te_metrics.csv.gz", "te_delta.csv.gz"])
                    if not manifest_is_complete(
                        session_group, config.hash, required
                    ):
                        raise RuntimeError(
                            "Familiarization session is incomplete or corrupted: "
                            f"{session_group}"
                        )
                    summaries.append(session_group / "network_summary.csv.gz")
    return summaries


def aggregate_outputs(config: PipelineConfig) -> dict[str, Path]:
    try:
        import pandas as pd
        from scipy import stats
    except ImportError as exc:
        raise RuntimeError("pandas and scipy are required for aggregation") from exc

    result_dir = config.output_root / "aggregate"
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_files = _completed_summary_files(config)
    summaries = pd.concat((pd.read_csv(path) for path in summary_files), ignore_index=True)
    summaries = summaries.drop_duplicates(
        subset=[
            "network_id", "stage", "environment", "variant", "session",
            "population", "module", "metric",
        ],
        keep="last",
    )
    network_metrics_path = result_dir / "network_metrics.csv.gz"
    summaries.to_csv(network_metrics_path, index=False, compression="gzip")

    group_columns = [
        "stage", "environment", "variant", "session", "population", "module", "metric"
    ]
    tests: list[dict[str, Any]] = []
    ratio_metrics = {"rescale_factor", "peak_rate_ratio", "mean_rate_ratio"}
    for keys, group in summaries.groupby(group_columns, dropna=False):
        values = group["mean"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 2:
            continue
        null = 1.0 if keys[-1] in ratio_metrics else 0.0
        t_result = stats.ttest_1samp(values, popmean=null)
        try:
            w_result = stats.wilcoxon(values - null, alternative="two-sided")
            wilcoxon_p = float(w_result.pvalue)
        except ValueError:
            wilcoxon_p = np.nan
        bootstrap_seed = derive_seed(
            config.master_seed, "aggregate_bootstrap", *[str(value) for value in keys]
        )
        ci_low, ci_high = _bootstrap_ci(
            values,
            seed=bootstrap_seed,
            draws=int(config.data["statistics"]["bootstrap_draws"]),
        )
        tests.append({
            **dict(zip(group_columns, keys)),
            "null_value": null,
            "n_networks": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "sem": float(stats.sem(values)),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "bootstrap_seed": bootstrap_seed,
            "t": float(t_result.statistic),
            "p_t_raw": float(t_result.pvalue),
            "p_wilcoxon_raw": wilcoxon_p,
            "effect_dz": float((np.mean(values) - null) / np.std(values, ddof=1))
            if np.std(values, ddof=1) > 0 else np.nan,
        })
    test_frame = pd.DataFrame(tests)
    if not test_frame.empty:
        test_frame["p_t_holm_global"] = holm_adjust(test_frame["p_t_raw"].to_numpy())
        test_frame["p_wilcoxon_holm_global"] = holm_adjust(
            test_frame["p_wilcoxon_raw"].to_numpy()
        )
    network_stats_path = result_dir / "network_statistics_exploratory.csv"
    test_frame.to_csv(network_stats_path, index=False)

    te_files = sorted(config.output_root.glob("networks/**/te_delta.csv.gz"))
    te_network_path = result_dir / "te_network_metrics.csv.gz"
    te_stats_path = result_dir / "te_network_statistics_exploratory.csv"
    if te_files:
        te = pd.concat((pd.read_csv(path) for path in te_files), ignore_index=True)
        te_group_columns = [
            "network_id", "stage", "environment", "session", "module", "embedding",
            "residualized", "pairing", "behavior", "wall_axis", "lag_frames",
            "lag_ms", "source_history", "target_history",
        ]
        te_network = (
            te.groupby(te_group_columns, dropna=False)["delta_directionality_index"]
            .mean()
            .reset_index()
        )
        te_network.to_csv(te_network_path, index=False, compression="gzip")
        infer_columns = te_group_columns[1:]
        te_tests = []
        for keys, group in te_network.groupby(infer_columns, dropna=False):
            values = group["delta_directionality_index"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(values) < 2:
                continue
            result = stats.ttest_1samp(values, popmean=0.0)
            bootstrap_seed = derive_seed(
                config.master_seed,
                "aggregate_te_bootstrap",
                *[str(value) for value in keys],
            )
            ci_low, ci_high = _bootstrap_ci(
                values,
                seed=bootstrap_seed,
                draws=int(config.data["statistics"]["bootstrap_draws"]),
            )
            te_tests.append({
                **dict(zip(infer_columns, keys)),
                "n_networks": len(values),
                "mean_delta_directionality": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "bootstrap_seed": bootstrap_seed,
                "t": float(result.statistic),
                "p_raw": float(result.pvalue),
                "effect_dz": float(np.mean(values) / np.std(values, ddof=1))
                if np.std(values, ddof=1) > 0 else np.nan,
            })
        te_test_frame = pd.DataFrame(te_tests)
        if not te_test_frame.empty:
            te_test_frame["p_holm_global"] = holm_adjust(te_test_frame["p_raw"].to_numpy())
        te_test_frame.to_csv(te_stats_path, index=False)
    else:
        pd.DataFrame().to_csv(te_network_path, index=False, compression="gzip")
        pd.DataFrame().to_csv(te_stats_path, index=False)

    return {
        "network_metrics": network_metrics_path,
        "network_statistics": network_stats_path,
        "te_network_metrics": te_network_path,
        "te_network_statistics": te_stats_path,
    }

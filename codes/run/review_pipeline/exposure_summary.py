from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .paths import evaluation_group_dir
from .seeds import derive_seed
from .storage import complete_manifest, manifest_is_complete


METRICS = (
    ("rescale_factor", "Rescaling factor"),
    ("r_phase", "Normalized phase difference"),
)
MODULE_COLORS = {1: "#3B6FB6", 2: "#C47A32", 3: "#56875D"}


def _atomic_dataframe(frame: Any, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_figure(figure: Any, path: Path, *, dpi: int = 600) -> None:
    temporary = path.with_name(
        f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{path.suffix}"
    )
    try:
        figure.savefig(
            temporary, dpi=dpi, bbox_inches="tight", facecolor="white",
            format=path.suffix.lstrip("."),
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _bootstrap_interval(values: np.ndarray, seed: int, draws: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1 or np.all(values == values[0]):
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _required_evaluation_outputs(config: PipelineConfig, stage: str) -> list[str]:
    required = ["network_summary.csv.gz"]
    for trajectory_id in config.trajectory_ids:
        required.extend([
            f"trajectory_{trajectory_id:02d}/cell_metrics.csv.gz",
            f"trajectory_{trajectory_id:02d}/rate_maps.npz",
            f"trajectory_{trajectory_id:02d}/checkpoint.npz",
        ])
        if stage == "initial_exposure":
            required.append(f"trajectory_{trajectory_id:02d}/comparison_metrics.csv.gz")
    return required


def _load_network_values(config: PipelineConfig) -> tuple[Any, list[Path]]:
    import pandas as pd

    inputs: list[Path] = []
    records: list[dict[str, Any]] = []
    baseline_id = config.baseline_environment["id"]
    baseline_cm = float(config.baseline_environment["width_m"]) * 100.0
    exposure_envs = sorted(
        (config.environment(env_id) for env_id in config.stage_environments("initial_exposure")),
        key=lambda item: float(item["width_m"]),
    )
    for network_id in config.network_ids:
        baseline_group = evaluation_group_dir(config, "baseline", network_id, baseline_id)
        if not manifest_is_complete(
            baseline_group,
            config.accepted_run_hashes("baseline"),
            _required_evaluation_outputs(config, "baseline"),
        ):
            raise RuntimeError(f"Baseline is incomplete or changed: {baseline_group}")
        inputs.append(baseline_group / "network_summary.csv.gz")
        for module in (1, 2, 3):
            records.extend([
                {
                    "network_id": network_id, "environment": baseline_id,
                    "environment_length_cm": baseline_cm, "module": module,
                    "metric": metric, "value": reference,
                    "is_defined_baseline_reference": True,
                }
                for metric, reference in (("rescale_factor", 1.0), ("r_phase", 0.0))
            ])

        for env in exposure_envs:
            group = evaluation_group_dir(
                config, "initial_exposure", network_id, str(env["id"])
            )
            if not manifest_is_complete(
                group, config.hash,
                _required_evaluation_outputs(config, "initial_exposure"),
            ):
                raise RuntimeError(f"Initial exposure is incomplete or changed: {group}")
            summary_path = group / "network_summary.csv.gz"
            inputs.append(summary_path)
            summary = pd.read_csv(summary_path)
            for module in (1, 2, 3):
                for metric, _ in METRICS:
                    selected = summary[
                        (summary["population"] == "grid")
                        & (summary["module"].astype(float) == float(module))
                        & (summary["metric"] == metric)
                    ]
                    if len(selected) != 1:
                        raise RuntimeError(
                            f"Expected one network-level {metric} row for module {module}: "
                            f"{summary_path}; found {len(selected)}"
                        )
                    records.append({
                        "network_id": network_id,
                        "environment": str(env["id"]),
                        "environment_length_cm": float(env["width_m"]) * 100.0,
                        "module": module,
                        "metric": metric,
                        "value": float(selected.iloc[0]["mean"]),
                        "is_defined_baseline_reference": False,
                    })
    return pd.DataFrame(records), inputs


def create_post_exposure_summary(config: PipelineConfig) -> dict[str, Path]:
    try:
        import matplotlib as mpl
        mpl.use("Agg", force=True)
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas and matplotlib are required for the exposure figure") from exc

    result_dir = config.output_root / "post_exposure_summary"
    result_dir.mkdir(parents=True, exist_ok=True)
    values, inputs = _load_network_values(config)
    values_path = result_dir / "network_level_rescaling_phase.csv.gz"
    _atomic_dataframe(values, values_path, compression="gzip")

    draws = int(config.data["statistics"]["bootstrap_draws"])
    summary_rows: list[dict[str, Any]] = []
    for (length, environment, module, metric), group in values.groupby(
        ["environment_length_cm", "environment", "module", "metric"], sort=True
    ):
        observed = group["value"].to_numpy(dtype=float)
        seed = derive_seed(
            config.master_seed, "post_exposure_figure_bootstrap",
            environment, int(module), metric,
        )
        low, high = _bootstrap_interval(observed, seed, draws)
        summary_rows.append({
            "environment_length_cm": float(length),
            "environment": environment,
            "module": int(module),
            "metric": metric,
            "n_networks": int(np.isfinite(observed).sum()),
            "mean": float(np.nanmean(observed)),
            "ci95_low": low,
            "ci95_high": high,
            "bootstrap_seed": seed,
            "is_defined_baseline_reference": bool(
                group["is_defined_baseline_reference"].all()
            ),
        })
    group_summary = pd.DataFrame(summary_rows)
    summary_path = result_dir / "group_summary_rescaling_phase.csv"
    _atomic_dataframe(group_summary, summary_path)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    figure, axes = plt.subplots(2, 3, figsize=(7.2, 4.3), sharex=True)
    letters = iter("abcdef")
    x_values = sorted(values["environment_length_cm"].unique())
    for row, (metric, ylabel) in enumerate(METRICS):
        for column, module in enumerate((1, 2, 3)):
            axis = axes[row, column]
            module_values = values[
                (values["metric"] == metric) & (values["module"] == module)
            ]
            color = MODULE_COLORS[module]
            for _, network in module_values.groupby("network_id"):
                network = network.sort_values("environment_length_cm")
                axis.plot(
                    network["environment_length_cm"], network["value"],
                    color="#A7A7A7", alpha=0.5, linewidth=0.55, zorder=1,
                )
                axis.scatter(
                    network["environment_length_cm"], network["value"],
                    color="#A7A7A7", alpha=0.6, s=6, linewidths=0, zorder=2,
                )
            selected = group_summary[
                (group_summary["metric"] == metric)
                & (group_summary["module"] == module)
            ].sort_values("environment_length_cm")
            axis.errorbar(
                selected["environment_length_cm"], selected["mean"],
                yerr=np.vstack([
                    selected["mean"] - selected["ci95_low"],
                    selected["ci95_high"] - selected["mean"],
                ]),
                color=color, marker="o", markersize=3.8, linewidth=1.5,
                elinewidth=1.0, capsize=2.0, zorder=4,
            )
            baseline = selected[selected["is_defined_baseline_reference"]]
            axis.scatter(
                baseline["environment_length_cm"], baseline["mean"],
                facecolors="white", edgecolors=color, marker="D", s=24,
                linewidths=1.0, zorder=5,
            )
            axis.axvline(100.0, color="#BDBDBD", linestyle=":", linewidth=0.8)
            if metric == "r_phase":
                axis.axhline(0.0, color="#6F6F6F", linestyle="--", linewidth=0.8)
            else:
                axis.plot(x_values, np.asarray(x_values) / 100.0,
                          color="#6F6F6F", linestyle="--", linewidth=0.8)
            axis.set_xticks(x_values)
            axis.set_xlim(min(x_values) - 3, max(x_values) + 3)
            axis.set_title(f"GC module {module}")
            if column == 0:
                axis.set_ylabel(ylabel)
            if row == 1:
                axis.set_xlabel("Environment length (cm)")
            axis.text(
                -0.17, 1.08, next(letters), transform=axis.transAxes,
                fontsize=9.0, fontweight="bold", va="top",
            )
    figure.suptitle(
        "Immediate grid response to enclosure deformation "
        f"(n = {len(config.network_ids)} independent networks)",
        fontsize=9.0, y=1.01,
    )
    figure.text(
        0.5, -0.01,
        "Thin grey lines: individual networks; colored points: mean ± 95% bootstrap CI; "
        "open diamond at 100 cm: defined baseline reference.",
        ha="center", va="top", fontsize=6.2,
    )
    figure.tight_layout(pad=0.8, h_pad=1.0, w_pad=1.0)
    stem = result_dir / "initial_exposure_rescaling_phase_networks"
    figure_paths = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg", ".tiff")]
    for path in figure_paths:
        _atomic_figure(figure, path)
    plt.close(figure)

    caption_path = result_dir / "initial_exposure_rescaling_phase_caption.txt"
    caption_path.write_text(
        "Immediate rescaling and normalized phase differences across "
        f"{len(config.network_ids)} independent networks. "
        "Columns show grid modules 1–3. Thin grey trajectories are individual "
        "network estimates; colored symbols and bars are network means and 95% bootstrap "
        "confidence intervals. The 100-cm open diamond is the defined self-comparison "
        "reference (rescaling factor 1; normalized phase difference 0), not an additional "
        "independent measurement. Dashed lines show complete geometric rescaling (top) "
        "and zero phase difference (bottom).\n",
        encoding="utf-8",
    )
    outputs = [values_path, summary_path, caption_path, *figure_paths]
    complete_manifest(
        result_dir, config.hash, "post_exposure_summary",
        {"networks": config.network_ids,
         "environments": [config.baseline_environment["id"],
                          *config.stage_environments("initial_exposure")]},
        {}, outputs, inputs,
        notes={"statistical_unit": "network", "n_networks": len(config.network_ids),
               "bootstrap_draws": draws,
               "baseline_reference_is_defined_not_measured": True},
    )
    return {
        "network_values": values_path,
        "group_summary": summary_path,
        "figure_png": figure_paths[0],
        "figure_pdf": figure_paths[1],
        "figure_svg": figure_paths[2],
        "figure_tiff": figure_paths[3],
        "caption": caption_path,
    }

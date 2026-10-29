from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .metrics import build_rate_maps
from .simulation import SimulationResult
from .storage import atomic_npz


EVALUATION_QC_PNG_FILENAMES = (
    "qc_rate_maps_place.png",
    "qc_rate_maps_grid_m1.png",
    "qc_rate_maps_grid_m2.png",
    "qc_rate_maps_grid_m3.png",
)
EVALUATION_QC_FILENAMES = tuple(
    filename
    for png_name in EVALUATION_QC_PNG_FILENAMES
    for filename in (
        png_name,
        str(Path(png_name).with_suffix(".pdf")),
        str(Path(png_name).with_suffix(".svg")),
    )
)


def _pyplot():
    import matplotlib as mpl

    mpl.use("Agg", force=True)
    mpl.rcParams.update({
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    import matplotlib.pyplot as plt

    return plt


def _atomic_savefig(figure: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp{path.suffix}"
    )
    try:
        figure.savefig(
            temporary,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            format=path.suffix.lstrip("."),
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _mean_rate_maps(paths: list[Path], population: str) -> np.ndarray:
    stacks = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            stacks.append(data[population].astype(np.float32, copy=True))
    values = np.stack(stacks, axis=0)
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    mean = np.full(sums.shape, np.nan, dtype=np.float32)
    np.divide(sums, counts, out=mean, where=counts > 0)
    return mean


def _normalised_map(rate_map: np.ndarray) -> tuple[np.ndarray, float]:
    peak = (
        float(np.nanmax(rate_map))
        if np.any(np.isfinite(rate_map))
        else float("nan")
    )
    if np.isfinite(peak) and peak > 0.0:
        return rate_map / peak, peak
    return np.where(np.isfinite(rate_map), 0.0, np.nan), peak


def _plot_rate_map_atlas(
    rate_maps: np.ndarray,
    output_path: Path,
    *,
    title: str,
    width_cm: float,
    height_cm: float,
) -> None:
    plt = _pyplot()
    n_cells = int(rate_maps.shape[0])
    columns = 10 if n_cells == 100 else int(np.ceil(np.sqrt(n_cells)))
    rows = int(np.ceil(n_cells / columns))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    with plt.rc_context({
        "font.family": "sans-serif",
        "font.size": 6.0,
        "axes.titlesize": 5.0,
    }):
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(0.82 * columns, 0.82 * rows + 0.45),
            squeeze=False,
        )
        for cell, axis in enumerate(axes.flat):
            if cell >= n_cells:
                axis.set_visible(False)
                continue
            normalised, peak = _normalised_map(rate_maps[cell])
            axis.imshow(
                normalised,
                origin="lower",
                extent=(0.0, width_cm, 0.0, height_cm),
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                aspect="equal",
            )
            peak_label = f"{peak:.2g}" if np.isfinite(peak) else "NA"
            axis.set_title(f"{cell:02d} | {peak_label}", pad=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(0.35)
                spine.set_color("#666666")
        figure.suptitle(
            title + "\nEach cell normalized to its own peak; title shows cell | peak rate",
            y=1.0,
            fontsize=7.0,
        )
        figure.subplots_adjust(
            left=0.02, right=0.995, bottom=0.02, top=0.94, wspace=0.08, hspace=0.22
        )
        _atomic_savefig(figure, output_path)
        _atomic_savefig(figure, output_path.with_suffix(".pdf"))
        _atomic_savefig(figure, output_path.with_suffix(".svg"))
        plt.close(figure)


def write_initialization_grid_qc(
    config: PipelineConfig,
    result: SimulationResult,
    env: dict[str, Any],
    output_dir: Path,
    *,
    network_id: int,
    session: int,
    grid_flat_indices: np.ndarray,
) -> list[Path]:
    """Persist three-module GC traces and their per-session rate-map QC."""

    indices = np.asarray(grid_flat_indices, dtype=np.int64).reshape(-1)
    activity = np.concatenate(result.grid_activity, axis=0)
    modules = np.repeat(np.arange(1, 4, dtype=np.int16), len(indices))
    cell_indices = np.tile(indices, 3)
    analysis = config.data["analysis"]
    rate_maps, occupancy = build_rate_maps(
        result.positions,
        activity,
        float(env["width_m"]),
        float(env["height_m"]),
        float(analysis["rate_map_bin_m"]),
        float(analysis["rate_map_sigma_bins"]),
    )

    activity_path = output_dir / "qc_grid_activity.npz"
    atomic_npz(
        activity_path,
        compressed=True,
        positions=result.positions.astype(np.float32, copy=False),
        grid_activity=activity.astype(np.float32, copy=False),
        module=modules,
        grid_flat_index=cell_indices.astype(np.int32),
        rate_maps=rate_maps,
        occupancy=occupancy,
        record_dt_s=np.array(result.record_dt_s),
        width_cm=np.array(float(env["width_m"]) * 100.0),
        height_cm=np.array(float(env["height_m"]) * 100.0),
    )

    figure_path = output_dir / "qc_grid_fields.png"
    plt = _pyplot()
    count = len(rate_maps)
    columns = min(3, count)
    rows = int(np.ceil(count / columns))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    with plt.rc_context({"font.family": "sans-serif", "font.size": 7.0}):
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(2.25 * columns, 2.25 * rows + 0.45),
            squeeze=False,
            constrained_layout=True,
        )
        image = None
        for index, axis in enumerate(axes.flat):
            if index >= count:
                axis.set_visible(False)
                continue
            normalised, peak = _normalised_map(rate_maps[index])
            image = axis.imshow(
                normalised,
                origin="lower",
                extent=(
                    0.0,
                    float(env["width_m"]) * 100.0,
                    0.0,
                    float(env["height_m"]) * 100.0,
                ),
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
                aspect="equal",
            )
            row, column = np.unravel_index(
                int(cell_indices[index]),
                (int(config.data["model"]["n_grid"]),) * 2,
            )
            axis.set_title(
                f"Module {int(modules[index])}, sheet cell ({row}, {column})\n"
                f"peak={peak:.3g}"
            )
            axis.set_xlabel("x (cm)")
            if index % columns == 0:
                axis.set_ylabel("y (cm)")
        if image is not None:
            figure.colorbar(
                image,
                ax=[axis for axis in axes.flat if axis.get_visible()],
                label="Normalized rate",
                fraction=0.035,
                pad=0.02,
            )
        figure.suptitle(
            f"Initialization QC · network {network_id:03d} · session {session:02d}",
            fontsize=9.0,
        )
        _atomic_savefig(figure, figure_path)
        figure_pdf_path = figure_path.with_suffix(".pdf")
        figure_svg_path = figure_path.with_suffix(".svg")
        _atomic_savefig(figure, figure_pdf_path)
        _atomic_savefig(figure, figure_svg_path)
        plt.close(figure)
    return [activity_path, figure_path, figure_pdf_path, figure_svg_path]


def write_evaluation_rate_map_atlases(
    config: PipelineConfig,
    rate_map_paths: list[Path],
    output_dir: Path,
    *,
    stage: str,
    network_id: int,
    env: dict[str, Any],
    trajectory_ids: list[int] | None = None,
    variant: str = "",
    session: int | None = None,
) -> list[Path]:
    """Write PC and three GC-module atlases for a completed evaluation/session."""

    populations = (
        ("place", "PC"),
        ("grid_m1", "GC module 1"),
        ("grid_m2", "GC module 2"),
        ("grid_m3", "GC module 3"),
    )
    outputs: list[Path] = []
    selected_ids = config.trajectory_ids if trajectory_ids is None else trajectory_ids
    trajectory_label = ", ".join(str(value) for value in selected_ids)
    context = stage
    if variant:
        context += f" · {variant}"
    if session is not None:
        context += f" · session {session:02d}"
    for (population, label), filename in zip(
        populations, EVALUATION_QC_PNG_FILENAMES
    ):
        output_path = output_dir / filename
        _plot_rate_map_atlas(
            _mean_rate_maps(rate_map_paths, population),
            output_path,
            title=(
                f"Network {network_id:03d} · {context} · {env['id']} · {label} · "
                f"mean of trajectories {trajectory_label}"
            ),
            width_cm=float(env["width_m"]) * 100.0,
            height_cm=float(env["height_m"]) * 100.0,
        )
        outputs.extend([
            output_path,
            output_path.with_suffix(".pdf"),
            output_path.with_suffix(".svg"),
        ])
    return outputs

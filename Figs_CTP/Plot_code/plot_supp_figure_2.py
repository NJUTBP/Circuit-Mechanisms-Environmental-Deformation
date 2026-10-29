"""Experimental/model comparison: rescaling, spatial information and phase."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.container import ErrorbarContainer
import numpy as np
import pandas as pd
from scipy.stats import sem,t
from archive_common import DATA as DATA_ROOT,export
DATA=DATA_ROOT/"supp_figure_2"
EXP="#27628D"
MODEL="#D47625"
NEUTRAL="#A5ACB2"
EXPERIMENT_COLOR="#74439B"
MODEL_COLOR="#087D7B"
NORMALIZED_RESCALING_LABEL=r"$N_{\mathrm{R}}$"
DIRECTIONS=("compression","expansion")
def configure() -> None:
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.sf": "Times New Roman",
        "mathtext.tt": "Times New Roman",
        "mathtext.cal": "Times New Roman",
        "font.size": 8.5,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.4,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })

def five_ticks(low: float, high: float, reference: float | None = None) -> np.ndarray:
    """Five readable ticks enclosing all network values and plotted intervals."""
    if not np.isfinite([low, high]).all() or low > high:
        raise ValueError(f"Invalid y range: {low}, {high}")
    span = max(high - low, 1e-4)
    # Choose the tick grid from the data range; add breathing space to the
    # axis limits later so a value near a round tick cannot double the span.
    bottom, top = (low, high) if high > low else (low - 0.02 * span, high + 0.02 * span)
    needed = (top - bottom) / 4
    exponent = int(np.floor(np.log10(needed)))
    steps = [1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]
    candidates = []
    for power in range(exponent - 1, exponent + 3):
        for mantissa in steps:
            step = mantissa * 10 ** power
            if 4 * step < top - bottom - 1e-10:
                continue
            first = int(np.floor(bottom / step))
            for index in range(first - 4, first + 2):
                ticks = (index + np.arange(5, dtype=float)) * step
                if ticks[0] > bottom + 1e-10 or ticks[-1] < top - 1e-10:
                    continue
                missing_reference = (
                    reference is not None and bottom <= reference <= top
                    and not np.isclose(ticks, reference, atol=1e-9).any()
                )
                excess = (ticks[-1] - ticks[0]) - (top - bottom)
                center = abs((ticks[0] + ticks[-1]) / 2 - (bottom + top) / 2)
                candidates.append((int(missing_reference), excess, center, step, ticks))
    if not candidates:
        raise RuntimeError("Could not place five y ticks")
    # Equivalent decimal grids can differ by a few binary floating-point bits.
    # Resolve those ties by centering the data rather than choosing an edge.
    ticks = min(candidates, key=lambda item: (
        item[0], round(item[1], 9), round(item[2], 9), item[3]))[-1]
    return np.round(ticks, 10)

def apply_five_ticks(ax: plt.Axes, low: float, high: float,
                     reference: float | None = None) -> None:
    ticks = five_ticks(low, high, reference)
    margin = 0.02 * max(high - low, 1e-4)
    ax.set_ylim(min(float(ticks[0]), low - margin),
                max(float(ticks[-1]), high + margin))
    ax.set_yticks(ticks)
    ax.tick_params(direction="out", width=0.6, length=2.5, pad=2)

def rescaling_class_label(comparison: str, threshold: int) -> str:
    """Label normalized-rescaling classes with an upright R subscript."""
    return rf"$N_{{\mathrm{{R}}}} {comparison} {threshold}\%$"

def jitter(n: int, half_width: float = 0.09) -> np.ndarray:
    # Deterministic horizontal display offsets retain all observations.
    return half_width * np.sin(np.arange(n, dtype=float) * 2.399963229728653)

def base_axis(ax: plt.Axes, title: str, letter: str) -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    height_pt = ax.get_position().height * ax.figure.get_size_inches()[1] * 72
    label_y = 1.06 + title.count("\n") * 11.3 / height_pt
    ax.text(-0.16, label_y, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left", clip_on=False)
    ax.grid(axis="y", color="#E5E8EB", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", width=0.6, length=2.5, pad=2)

def draw_phase_axes(axes: np.ndarray, summary: dict, threshold: int = 30,
                    letters: str = "AB", *, compact: bool = False) -> None:
    exp = pd.read_csv(DATA / "phase_experimental_records.csv")
    net = pd.read_csv(DATA / f"phase_model_networks_{threshold}pct.csv")
    if len(exp) != 193 or len(net) != 16:
        raise ValueError("Incomplete prepared phase-comparison tables")
    low = net.loc[~net[f"rescaling_above_{threshold}pct"]].sort_values("network_id")
    high = net.loc[net[f"rescaling_above_{threshold}pct"]].sort_values("network_id")
    if (len(low) != 8 or len(high) != 8
            or not np.array_equal(low.network_id.to_numpy(), high.network_id.to_numpy())):
        raise ValueError("Phase classes must pair the same eight independent networks")
    envelope = list(exp.phase_change_half_grid_period.to_numpy(float))
    ax = axes[0]
    base_axis(ax, "Barry + Stensola records" if compact else
              "Barry + Stensola records\nKeinath reanalysis", letters[0])
    for x, group in enumerate(("≤50%", ">50%")):
        vals = exp.loc[exp.published_group.eq(group), "phase_change_half_grid_period"].to_numpy(float)
        info = summary["phase"]["experiment_50"]["low" if x == 0 else "high"]
        if len(vals) != info["n_cell_condition_axis_rows"] or not np.isclose(vals.mean(), info["mean"]):
            raise ValueError("Experimental phase source/summary mismatch")
        ax.scatter(x + jitter(len(vals), 0.12), vals, s=9,
                   color=EXP, alpha=0.22, edgecolors="none", zorder=2)
        sem = info["sem"]
        ax.errorbar([x], [vals.mean()], yerr=[[sem], [sem]],
                    fmt="D", ms=5.8, color=EXP, mec="white", mew=0.7,
                    capsize=2.5, elinewidth=1.0, zorder=5)
        envelope.extend((vals.mean() - sem, vals.mean() + sem))
    ax.set_xticks([0, 1], [rescaling_class_label(r"\leq", 50), rescaling_class_label(">", 50)])

    ax = axes[1]
    base_axis(ax, "Model" if compact else
              f"Model\n{NORMALIZED_RESCALING_LABEL} classification ({threshold}% threshold)", letters[1])
    for lv, hv in zip(low.phase_mean, high.phase_mean):
        ax.plot([0, 1], [lv, hv], color="#B9A99A", lw=0.55, alpha=0.65, zorder=1)
    for x, vals, label in ((0, low.phase_mean, "low"), (1, high.phase_mean, "high")):
        info = summary["phase"][str(threshold)]["model"][label]
        ci_low, ci_high = info["network_95ci"]
        mean = float(vals.mean())
        if not np.isclose(mean, info["mean_of_network_means"]):
            raise ValueError("Model phase source/summary mismatch")
        ax.scatter(x + jitter(len(vals), 0.06), vals, s=18, marker="s",
                   color=MODEL, alpha=0.65, edgecolors="none", zorder=3)
        ax.errorbar([x], [mean], yerr=[[mean - ci_low], [ci_high - mean]],
                    fmt="D", ms=5.8, color=MODEL, mec="white", mew=0.7,
                    capsize=2.5, elinewidth=1.0, zorder=5)
        envelope.extend((*vals.to_numpy(float), ci_low, ci_high))
    ax.set_xticks([0, 1], [rescaling_class_label(r"\leq", threshold), rescaling_class_label(">", threshold)])
    if not np.isfinite(envelope).all():
        raise ValueError("Nonfinite phase data or interval")
    for ax in axes:
        ax.axhline(0, color=NEUTRAL, ls=":", lw=0.7, zorder=0)
        ax.set_xlim(-0.38, 1.38)
        ax.set_ylabel("Deformed-familiar shift/(scale/2)")
        apply_five_ticks(ax, min(envelope), max(envelope), reference=0)
        ax.tick_params(axis="y", labelleft=True)

def read_figure_rows() -> dict[str, dict[str, str]]:
    with (DATA / "spatial_information_figure_source_data.csv").open(
            newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["system"]: row for row in rows}
    if set(result) != {"mouse_experiment", "circuit_model"} or len(rows) != 2:
        raise ValueError("Expected exactly experiment and model figure rows")
    return result

def read_networks() -> list[dict[str, str]]:
    with (DATA / "model_equal_weight_network_values.csv").open(
            newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 8 or len({r["network"] for r in rows}) != 8:
        raise ValueError("Expected eight independent model networks")
    return rows

def load_pooled() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    experimental = pd.read_csv(DATA / "rescaling_experimental_cells.csv")
    tables = [pd.read_csv(DATA / f"rescaling_model_networks_{direction}.csv")
              for direction in DIRECTIONS]
    model = pd.concat(tables, ignore_index=True)
    assert set(experimental.direction) == set(DIRECTIONS)
    assert experimental.groupby("direction").size().to_dict() == {
        "compression": 23, "expansion": 13}
    assert np.isfinite(experimental.normalized_rescaling).all()
    assert np.isfinite(model.normalized_rescaling).all()
    assert np.allclose(
        experimental.normalized_rescaling,
        (experimental.nominal_reconstructed_FR - 1) / (experimental.deformation_ratio - 1),
    )
    paired = model.pivot(index="network_id", columns="direction", values="normalized_rescaling")
    assert paired.shape == (8, 2) and not paired.isna().any().any()
    assert set(paired.columns) == set(DIRECTIONS)
    paired = paired.loc[:, list(DIRECTIONS)].copy()
    paired["pooled_equal_condition_NR"] = paired.mean(axis=1)
    network_values = paired.pooled_equal_condition_NR.to_numpy(float)
    center = float(network_values.mean())
    model_sem = float(sem(network_values))
    ci = t.interval(0.95, df=7, loc=center, scale=model_sem)
    experimental["pooled_summary_row_weight"] = experimental.direction.map({
        direction: 0.5 / int(experimental.direction.eq(direction).sum())
        for direction in DIRECTIONS})
    experimental_mean = float(np.dot(experimental.normalized_rescaling,
                                     experimental.pooled_summary_row_weight))
    summary = {
        "condition_weights": {direction: 0.5 for direction in DIRECTIONS},
        "experiment": {
            "mean": experimental_mean,
            "condition_means": experimental.groupby("direction").normalized_rescaling.mean().to_dict(),
            "n_source_rows_by_condition": experimental.groupby("direction").size().to_dict(),
            "interval": None,
            "interval_reason": "Source rows lack cell/animal IDs; no independent-animal interval inferred.",
        },
        "model": {
            "mean": center, "sem": model_sem, "sd": float(network_values.std(ddof=1)),
            "ci95": [float(ci[0]), float(ci[1])],
            "ci_method": "Two-sided Student t confidence interval across eight network paired means; df=7",
            "n_independent_networks": 8, "n_condition_network_points": 16,
            "condition_means": model.groupby("direction").normalized_rescaling.mean().to_dict(),
            "selected_cell_criterion": "N_R > 30%; original prepared selected populations unchanged",
        },
        "cross_study_test": None,
    }
    assert np.isclose(experimental.pooled_summary_row_weight.sum(), 1)
    assert np.isclose(experimental_mean, 0.6786850477200423)
    assert np.isclose(center, 0.5907622222680189)
    return experimental, paired.reset_index(), summary

def draw_pooled(ax: plt.Axes, experimental: pd.DataFrame, paired: pd.DataFrame,
                summary: dict) -> None:
    base_axis(ax, "Normalized rescaling", "A")
    ax.axhline(1, color=NEUTRAL, ls="--", lw=0.7, zorder=1)
    ax.axhline(0, color=NEUTRAL, ls=":", lw=0.7, zorder=1)
    ex = experimental.normalized_rescaling.to_numpy(float)
    vals = paired.loc[:, list(DIRECTIONS)].to_numpy(float).ravel()
    # One pooled point cloud per system: horizontal offsets encode no condition.
    ax.scatter(jitter(len(ex), 0.09), ex, s=18, marker="o",
               color=EXP, alpha=0.52, linewidths=0, zorder=2)
    ax.scatter(1 + jitter(len(vals), 0.09), vals, s=20, marker="o",
               color=MODEL, alpha=0.62, linewidths=0, zorder=2)
    ax.plot(0, summary["experiment"]["mean"], marker="D", ms=6,
            mfc=EXP, mec="white", mew=0.8, linestyle="none", zorder=5)
    center = summary["model"]["mean"]
    low, high = summary["model"]["ci95"]
    ax.errorbar([1], [center], yerr=[[center - low], [high - center]],
                fmt="D", ms=6, color=MODEL, mec="white", mew=0.8,
                capsize=3, elinewidth=1.0, zorder=5)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.55, 1.55)
    ax.set_yticks([-0.5, 0, 0.5, 1, 1.5])
    ax.set_ylabel(NORMALIZED_RESCALING_LABEL)
    ax.set_xticks([0, 1], ["Experiment", f"Model ({rescaling_class_label('>', 30)})"])

def draw_spatial(ax: plt.Axes) -> dict:
    base_axis(ax, "Spatial information", "B")
    rows = read_figure_rows()
    networks = read_networks()
    vals = np.array([float(row["paired_delta_bits_per_spike"]) for row in networks])
    assert np.isclose(vals.mean(), float(rows["circuit_model"]["delta_bits_per_spike"]))
    ax.scatter(1 + jitter(len(vals), 0.09), vals, s=20,
               color=MODEL_COLOR, alpha=0.40, linewidths=0, zorder=2)
    for x, system, color, marker in (
        (0, "mouse_experiment", EXPERIMENT_COLOR, "D"),
        (1, "circuit_model", MODEL_COLOR, "o"),
    ):
        row = rows[system]
        center, low, high = [float(row[key]) for key in
                             ("delta_bits_per_spike", "ci95_low", "ci95_high")]
        assert low < center < high
        assert np.isclose(float(row["final_bits_per_spike"]) -
                          float(row["first_exposure_bits_per_spike"]), center)
        ax.errorbar([x], [center], yerr=[[center - low], [high - center]],
                    fmt=marker, ms=6.2, color=color, mfc="white", mec=color, mew=1.2,
                    capsize=3, elinewidth=1.2, zorder=5)
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(0, 0.42)
    ax.set_yticks(np.arange(0, 0.41, 0.1))
    ax.set_xticks([0, 1], ["Mouse experiment", "Model"])
    ax.set_ylabel("Final − first-exposure\nspatial information (bits/spike)", labelpad=5)
    return rows

def plot(output_dir=None):
    experimental,paired,summary=load_pooled()
    original_summary=json.loads((DATA/"phase_summary.json").read_text(encoding="utf-8"))
    configure()
    fig=plt.figure(figsize=(7.2,5.8))
    grid=fig.add_gridspec(2,2,left=.115,right=.97,bottom=.09,top=.92,wspace=.48,hspace=.50)
    axes={"A":fig.add_subplot(grid[0,0]),"B":fig.add_subplot(grid[0,1]),
          "C":fig.add_subplot(grid[1,0]),"D":fig.add_subplot(grid[1,1])}
    draw_pooled(axes["A"],experimental,paired,summary)
    draw_spatial(axes["B"])
    draw_phase_axes(np.array([axes["C"],axes["D"]]),original_summary,threshold=30,letters="CD",compact=True)
    for container in axes["C"].containers:
        if isinstance(container,ErrorbarContainer):
            marker=container.lines[0]
            marker.set_markerfacecolor(EXP)
            marker.set_markeredgecolor("white")
            marker.set_markeredgewidth(.7)
            marker.set_markersize(5.8)
            for artist in (*container.lines[1],*container.lines[2]): artist.set_visible(False)
    return export(fig,2,output_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    print(plot(parser.parse_args().output_dir))

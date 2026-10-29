"""Plot pathway learning-rate sensitivity."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from controls_common import configure, tight_axis, fit_panel, save_figure

PACKAGE = Path(__file__).resolve().parents[1]

from controls_common import apply_five_ticks

SUBSCRIPTS = {"W_bp":"PB", "W_pg":"GP", "W_gp":"PG"}


def learning_rate_label(pathway):
    return rf"$\eta_{{\mathrm{{{SUBSCRIPTS[pathway]}}}}}=0$"


def learning_rate_symbol(pathway, bold=False):
    subscript = SUBSCRIPTS[pathway]
    if bold:
        return rf"$\mathbf{{\eta}}_{{\mathbf{{{subscript}}}}}$"
    return rf"$\eta_{{\mathrm{{{subscript}}}}}$"
PATHWAYS = ("W_bp", "W_gp", "W_pg")


PATH_LABELS = {pathway: learning_rate_symbol(pathway, bold=True) for pathway in PATHWAYS}


MULTIPLIERS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


MULTIPLIER_COLORS = {0.0: "#2166AC", 0.25: "#67A9CF", 0.5: "#1B9E77",
                     1.0: "#242424", 2.0: "#E69F00", 4.0: "#B2182B"}


MULTIPLIER_MARKERS = {0.0: "s", 0.25: "v", 0.5: "^", 1.0: "o", 2.0: "D", 4.0: "P"}


METRICS = ("rescale_factor", "spatial_information_bits_per_spike")


METRIC_LABELS = {"rescale_factor": r"$F_{\mathrm{R}}$",
                 "spatial_information_bits_per_spike": "Spatial information"}


SESSIONS = tuple(range(10))


def y_limits(curves: pd.DataFrame, metric: str, environments: tuple[str, ...]) -> tuple[float, float]:
    values = curves.loc[curves.metric.eq(metric) & curves.environment.isin(environments),
                        "network_mean"].to_numpy(float)
    if metric == "rescale_factor":
        values = np.r_[values, 1.0]
    lower, upper = float(values.min()), float(values.max())
    return lower, upper


def decorate(ax: plt.Axes, *, metric: str, limits: tuple[float, float], letter: str,
             xlabel: bool = True) -> None:
    if metric == "rescale_factor":
        ax.axhline(1.0, color="#888888", ls=(0, (2, 2)), lw=.6, zorder=0)
    ax.set_xlim(-.3, 9.3)
    ax.set_xticks(range(0, 10, 2))
    if xlabel:
        ax.set_xlabel("Session")
    ax.set_ylabel(METRIC_LABELS[metric])
    apply_five_ticks(ax, *limits, reference=1.0 if metric == "rescale_factor" else None)
    ax.text(-.16, 1.06, letter, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", clip_on=False)


def plot_series(ax: plt.Axes, individual: pd.DataFrame, mean: pd.DataFrame,
                color: str, marker: str, label: str) -> None:
    for _, one in individual.groupby("network_id", sort=True):
        one = one.sort_values("session")
        if one.session.tolist() != list(SESSIONS):
            raise RuntimeError("An individual-network line lacks sessions 0–9")
        ax.plot(one.session, one.network_mean, color=color, lw=.52, alpha=.19, zorder=1)
    mean = mean.sort_values("session")
    if mean.session.tolist() != list(SESSIONS):
        raise RuntimeError("A group-mean line lacks sessions 0–9")
    ax.plot(mean.session, mean["mean"], color=color, marker=marker,
            markerfacecolor="white", markeredgewidth=.8, markersize=4.1,
            lw=1.3, label=label, zorder=3)


def plot(curves: pd.DataFrame, group: pd.DataFrame, output: Path) -> None:
    env = "compressed_80"
    mpl.rcParams["legend.frameon"] = False
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.4))
    fig.subplots_adjust(left=.105, right=.985, bottom=.09, top=.88,
                        hspace=.45, wspace=.31)
    for row, pathway in enumerate(PATHWAYS):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            for multiplier in MULTIPLIERS:
                mask = (curves.pathway.eq(pathway) & curves.multiplier.eq(multiplier) &
                        curves.environment.eq(env) & curves.metric.eq(metric))
                gm = (group.pathway.eq(pathway) & group.multiplier.eq(multiplier) &
                      group.environment.eq(env) & group.metric.eq(metric))
                plot_series(ax, curves.loc[mask], group.loc[gm], MULTIPLIER_COLORS[multiplier],
                            MULTIPLIER_MARKERS[multiplier], f"{multiplier:g}×")
            decorate(ax, metric=metric,
                     limits=y_limits(curves.loc[curves.pathway.eq(pathway)], metric, (env,)),
                     letter="ABCDEF"[row * 2 + col], xlabel=row == 2)
            if row == 2 and col == 0 and env == "compressed_80":
                ax.set_ylim(bottom=0.8)
                ax.set_yticks([0.8, 0.9, 1.0, 1.1])
            if col == 0:
                ax.set_title(PATH_LABELS[pathway], loc="left", color="black",
                             fontweight="bold", pad=5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .995), ncol=6,
               columnspacing=1.15, handlelength=1.5)
    for axis in fig.axes:
        fit_panel(axis)
    return save_figure(fig, output, "supp_figure_15")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE / "Plot_data/supp_figure_15")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "Reproduced/Supp_figs")
    args = parser.parse_args()
    configure()
    network = pd.read_csv(args.data_dir / "network_curves.csv", float_precision="round_trip")
    group = pd.read_csv(args.data_dir / "group_curves.csv", float_precision="round_trip")
    print(plot(network, group, args.output_dir))


if __name__ == "__main__":
    main()

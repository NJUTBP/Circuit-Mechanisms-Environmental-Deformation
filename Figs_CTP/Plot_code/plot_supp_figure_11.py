"""Plot place-cell-count sensitivity during intact familiarization."""
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
COUNT_MARKERS = {100:"o",200:"s",300:"^",400:"D"}
COUNTS = (100, 200, 300, 400)


ENVS = ("compressed_80", "elongated_120")


LENGTHS = (70, 80, 90, 100, 110, 120, 130)


ENV_LENGTH = {"compressed_70": 70, "compressed_80": 80,
              "compressed_90": 90, "fam_100": 100,
              "elongated_110": 110, "elongated_120": 120,
              "elongated_130": 130}


COUNT_COLORS = {100: "#3B4CC0", 200: "#2697A9", 300: "#D99138", 400: "#B34C67"}


PANELS = (
    ("rescale_factor", r"$F_{\mathrm{R}}$", 1.0),
    ("delta_R_phase_x", r"$R_{\mathrm{phase}}$", 0.0),
    ("delta_place_field_length_x_cm", r"$\Delta L_{\mathrm{PF}}$ (cm)", 0.0),
    ("delta_place_phase_distance_x_cm", r"$\Delta D_{\mathrm{PF}}$ (cm)", 0.0),
)


def draw_curves(ax: plt.Axes, network: pd.DataFrame, group: pd.DataFrame, *,
                metric: str, module: int | None = None, env: str | None = None,
                initial: bool) -> tuple[float, float]:
    x_expected = LENGTHS if initial else tuple(range(10))
    lower, upper = [], []
    for count in COUNTS:
        n = network.loc[network["count"].eq(count) & network.metric.eq(metric)].copy()
        g = group.loc[group["count"].eq(count) & group.metric.eq(metric)].copy()
        if initial:
            n = n.loc[n.module.eq(module)]
            g = g.loc[g.module.eq(module)]
            n["x_value"] = n.environment.map(ENV_LENGTH)
            g["x_value"] = g.environment.map(ENV_LENGTH)
        else:
            n = n.loc[n.environment.eq(env)]
            g = g.loc[g.environment.eq(env)]
            n["x_value"], g["x_value"] = n.session, g.session
        g = g.set_index("x_value").reindex(x_expected)
        if (set(n.network_id) != {0, 1, 2, 3} or len(n) != 4 * len(x_expected)
                or len(g) != len(x_expected) or g["mean"].isna().any()
                or not g.n_networks.eq(4).all()):
            raise ValueError(f"Incomplete {metric} source for count={count}, module={module}, env={env}")
        x = np.asarray(x_expected, dtype=float)
        if initial:
            x = x / 100
        color = COUNT_COLORS[count]
        for _, trace in n.groupby("network_id", sort=True):
            y = trace.set_index("x_value").reindex(x_expected).network_mean.to_numpy(float)
            if not np.isfinite(y).all():
                raise ValueError(f"Incomplete independent network for count={count}, metric={metric}")
            ax.plot(x, y, color=color, lw=0.45, alpha=0.16, zorder=1)
            lower.append(float(y.min()))
            upper.append(float(y.max()))
        mean = g["mean"].to_numpy(float)
        ci_low = g.ci95_low.to_numpy(float)
        ci_high = g.ci95_high.to_numpy(float)
        if not np.isfinite(np.r_[mean, ci_low, ci_high]).all():
            raise ValueError(f"Nonfinite group interval for count={count}, metric={metric}")
        ax.fill_between(x, ci_low, ci_high, color=color, alpha=0.12,
                        linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, marker=COUNT_MARKERS[count], ms=4.1, lw=1.25, zorder=3)
        lower.append(float(ci_low.min()))
        upper.append(float(ci_high.max()))
    return min(lower), max(upper)


def format_axis(ax: plt.Axes, bounds: tuple[float, float], *,
                ylabel: str, reference: float, initial: bool,
                xlabel: bool, letter: str) -> None:
    ax.axhline(reference, color="#888888", lw=0.6, ls=(0, (2, 2)), zorder=0)
    if initial:
        ax.axvline(1.0, color="#A0A0A0", lw=0.55, ls=(0, (2, 2)), zorder=0)
        ax.set_xlim(0.67, 1.33)
        ax.set_xticks(np.asarray(LENGTHS) / 100)
    else:
        ax.set_xlim(-0.3, 9.3)
        ax.set_xticks(range(0, 10, 2))
    if xlabel:
        ax.set_xlabel("Deformation ratio" if initial else "Session")
    ax.set_ylabel(ylabel)
    apply_five_ticks(ax, min(bounds[0], reference), max(bounds[1], reference), reference)
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=10, fontweight="bold", clip_on=False)


def count_legend(fig: plt.Figure) -> None:
    handles = [Line2D([], [], color=COUNT_COLORS[count], marker=COUNT_MARKERS[count], ms=4.1,
                      lw=1.25, label=f"{count} place cells") for count in COUNTS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=4, frameon=False, handlelength=1.5, columnspacing=1.0)


def plot(network: pd.DataFrame, group: pd.DataFrame, output: Path) -> list[Path]:
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 8.1))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.91,
                        hspace=0.43, wspace=0.31)
    for row, (metric, ylabel, reference) in enumerate(PANELS):
        for col, env in enumerate(ENVS):
            ax = axes[row, col]
            bounds = draw_curves(ax, network, group, metric=metric,
                                 env=env, initial=False)
            format_axis(ax, bounds, ylabel=ylabel, reference=reference,
                        initial=False, xlabel=row == 3,
                        letter="ABCDEFGH"[row * 2 + col])
            if row == 0:
                ax.set_title("80 cm" if col == 0 else "120 cm",
                             color="black", fontweight="bold", pad=5)
    count_legend(fig)
    for axis in fig.axes:
        tight_axis(axis)
    return save_figure(fig, output, "supp_figure_11")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE / "Plot_data/supp_figure_11")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "Reproduced/Supp_figs")
    args = parser.parse_args()
    configure()
    network = pd.read_csv(args.data_dir / "network_curves.csv", float_precision="round_trip")
    group = pd.read_csv(args.data_dir / "group_curves.csv", float_precision="round_trip")
    print(plot(network, group, args.output_dir))


if __name__ == "__main__":
    main()

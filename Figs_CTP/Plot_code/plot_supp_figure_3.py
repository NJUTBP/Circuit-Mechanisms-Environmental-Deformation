"""Plot grid-spacing sensitivity at initial exposure."""
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

ALPHA_MARKERS = {0.08:"s",0.09:"v",0.10:"o",0.11:"^",0.12:"D"}
ALPHAS = (0.08, 0.09, 0.10, 0.11, 0.12)


LENGTHS = (70, 80, 90, 100, 110, 120, 130)


CONDITION_COLORS = ["#3B4CC0", "#6889D8", "#A9C2E8", "#E7A47E", "#C65556", "#8F2E4F"]


def plot(network: pd.DataFrame, summary: pd.DataFrame, output: Path) -> list[Path]:
    """Plot initial exposure only: rescaling and x-axis grid-phase change."""
    metrics = ("rescale_factor", "delta_R_phase_x")
    allowed = {f"alpha1_{value:.2f}".replace(".", "p") for value in ALPHAS}
    n = network.loc[
        network.condition.isin(allowed) & network.population.eq("grid")
        & network.metric.isin(metrics) & network.session.eq(0)
        & network.length_cm.isin(LENGTHS)
    ].copy()
    s = summary.loc[
        summary.condition.isin(allowed) & summary.population.eq("grid")
        & summary.metric.isin(metrics) & summary.session.eq(0)
        & summary.length_cm.isin(LENGTHS)
    ].copy()
    if (len(n) != 5 * 4 * 3 * 2 * 7 or len(s) != 5 * 3 * 2 * 7
            or n.duplicated(["condition", "network_id", "module", "metric", "length_cm"]).any()
            or s.duplicated(["condition", "module", "metric", "length_cm"]).any()
            or set(n.network_id) != {0, 1, 2, 3}
            or set(n.module) != {1, 2, 3}
            or set(n.length_cm) != set(LENGTHS)
            or not s.n_networks.eq(4).all()
            or not np.isfinite(n.network_mean).all()
            or not np.isfinite(s[["mean", "ci95_low", "ci95_high"]].to_numpy(float)).all()):
        raise ValueError("Incomplete or nonfinite initial-exposure grid-scale source")

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.5))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.12, top=0.84,
                        wspace=0.28, hspace=0.30)
    x = np.asarray(LENGTHS, dtype=float) / 100
    colors = {value: CONDITION_COLORS[i] for i, value in enumerate(ALPHAS)}
    for row, metric in enumerate(metrics):
        for col, module in enumerate((1, 2, 3)):
            ax = axes[row, col]
            for value in ALPHAS:
                condition = f"alpha1_{value:.2f}".replace(".", "p")
                color = colors[value]
                one_n = n.loc[n.metric.eq(metric) & n.module.eq(module)
                              & n.condition.eq(condition)]
                one_s = s.loc[s.metric.eq(metric) & s.module.eq(module)
                              & s.condition.eq(condition)].set_index("length_cm").reindex(LENGTHS)
                if len(one_n) != 4 * 7 or len(one_s) != 7 or one_s["mean"].isna().any():
                    raise ValueError(f"Incomplete curve for {metric}, module {module}, alpha {value}")
                for _, one in one_n.groupby("network_id", sort=True):
                    y = one.set_index("length_cm").reindex(LENGTHS).network_mean.to_numpy(float)
                    if not np.isfinite(y).all():
                        raise ValueError(f"Incomplete network curve for {metric}, module {module}")
                    ax.plot(x, y, color=color, lw=0.48, alpha=0.20, zorder=1)
                mean = one_s["mean"].to_numpy(float)
                low = one_s.ci95_low.to_numpy(float)
                high = one_s.ci95_high.to_numpy(float)
                ax.fill_between(x, low, high, color=color, alpha=0.10,
                                linewidth=0, zorder=2)
                ax.plot(x, mean, color=color, lw=1.35, marker=ALPHA_MARKERS[value], ms=4.1,
                        zorder=3)
            ax.axhline(1.0 if row == 0 else 0.0, color="#888888",
                       lw=0.6, ls=(0, (2, 2)), zorder=0)
            ax.axvline(1.0, color="#A0A0A0", lw=0.55,
                       ls=(0, (2, 2)), zorder=0)
            ax.set_xlim(0.67, 1.33)
            ax.set_xticks(x)
            ax.tick_params(direction="out", width=0.6, length=2.5, pad=2)
            ax.text(-0.20, 1.06, "ABCDEF"[row * 3 + col], transform=ax.transAxes,
                    fontsize=10, fontweight="bold", ha="left", va="bottom", clip_on=False)
            if row == 0:
                ax.set_title(f"Module {module}", color="black", fontweight="bold", pad=6)
                ax.set_ylim(0.65, 1.50)
                ax.set_yticks([0.7, 0.9, 1.1, 1.3, 1.5])
            else:
                ax.set_xlabel("Deformation ratio")
                # Preserve every network trace and the full 95% interval.
                panel_n = n.loc[n.metric.eq(metric) & n.module.eq(module), "network_mean"]
                panel_s = s.loc[s.metric.eq(metric) & s.module.eq(module)]
                envelope = np.r_[panel_n, panel_s.ci95_low, panel_s.ci95_high, 0.0]
                lo, hi = float(envelope.min()), float(envelope.max())
                margin = 0.03 * (hi - lo)
                ticks = [0.0, 0.25, 0.5, 0.75, 1.0] if hi > 1.0 else [-0.2, 0.0, 0.3, 0.6, 0.9]
                ax.set_ylim(min(lo - margin, ticks[0]), max(hi + margin, ticks[-1]))
                ax.set_yticks(ticks)
            if col == 0:
                ax.set_ylabel(r"$F_{\mathrm{R}}$" if row == 0 else
                              r"$R_{\mathrm{phase}}$")
    handles = [Line2D([], [], color=colors[value], marker=ALPHA_MARKERS[value], lw=1.35,
                      ms=4.1, label=rf"$\alpha_1={value:.2f}$") for value in ALPHAS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=5, frameon=False, columnspacing=1.1, handlelength=1.5)
    for axis in fig.axes:
        tight_axis(axis)
    return save_figure(fig, output, "supp_figure_3")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE / "Plot_data/supp_figure_3")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "Reproduced/Supp_figs")
    args = parser.parse_args()
    configure()
    network = pd.read_csv(args.data_dir / "network_curves.csv", float_precision="round_trip")
    group = pd.read_csv(args.data_dir / "group_curves.csv", float_precision="round_trip")
    print(plot(network, group, args.output_dir))


if __name__ == "__main__":
    main()

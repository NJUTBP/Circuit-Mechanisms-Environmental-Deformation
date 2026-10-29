"""Plot familiarization curves with two-sided 95% Student-t confidence bands."""
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

SUBSCRIPTS = {"W_bp":"PB", "W_pg":"GP", "W_gp":"PG"}


def learning_rate_label(pathway):
    return rf"$\eta_{{\mathrm{{{SUBSCRIPTS[pathway]}}}}}=0$"


def learning_rate_symbol(pathway, bold=False):
    subscript = SUBSCRIPTS[pathway]
    if bold:
        return rf"$\mathbf{{\eta}}_{{\mathbf{{{subscript}}}}}$"
    return rf"$\eta_{{\mathrm{{{subscript}}}}}$"
CONDITIONS = ("Intact", "Eta_bp=0", "Eta_pg=0", "Eta_gp=0")


LABELS = {
    "Intact": "Control",
    "Eta_bp=0": learning_rate_label("W_bp"),
    "Eta_pg=0": learning_rate_label("W_pg"),
    "Eta_gp=0": learning_rate_label("W_gp"),
}


COLORS = {
    "Intact": "#2B2B2B",
    "Eta_bp=0": "#0072B2",
    "Eta_pg=0": "#D55E00",
    "Eta_gp=0": "#009E73",
}


MARKERS = {"Intact": "o", "Eta_bp=0": "s", "Eta_pg=0": "^", "Eta_gp=0": "D"}


PANELS = (
    ("rescale_factor", r"$F_{\mathrm{R}}$", "A"),
    ("spatial_information_bits_per_spike", "Spatial information", "B"),
)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.sf": "Times New Roman",
        "mathtext.tt": "Times New Roman",
        "mathtext.cal": "Times New Roman",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "legend.frameon": False,
    })


def checked_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    networks = pd.read_csv(input_dir / "network_curves.csv", float_precision="round_trip")
    group = pd.read_csv(input_dir / "group_curves.csv", float_precision="round_trip")
    if len(networks) != 320 or len(group) != 80:
        raise ValueError("Expected 320 network points and 80 group points")
    for condition in CONDITIONS:
        for metric, _, _ in PANELS:
            item = networks.loc[
                networks["condition_label"].eq(condition) & networks["metric"].eq(metric)
            ]
            if set(item["network_id"]) != {0, 1, 2, 3}:
                raise ValueError(f"Incomplete networks: {condition}, {metric}")
            if not item.groupby("network_id")["session"].apply(
                lambda value: sorted(value.tolist()) == list(range(10))
            ).all():
                raise ValueError(f"Incomplete sessions: {condition}, {metric}")
            g = group.loc[group["condition_label"].eq(condition) & group["metric"].eq(metric)]
            direct = item.groupby("session")["network_mean"].mean().sort_index()
            saved = g.set_index("session")["mean"].sort_index()
            # Network CSV values retain nine decimal places; allow rounding only.
            if not np.allclose(direct, saved, rtol=0, atol=5.1e-10):
                raise ValueError(f"Saved network mean mismatch: {condition}, {metric}")
    return networks, group

def plot(networks, group, output_dir):
    configure_style()
    mpl.rcParams.update({"font.family":["Times New Roman","Arial"],"pdf.fonttype":42,
                         "svg.fonttype":"none","font.size":14})
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    fig.subplots_adjust(left=.08,right=.99,bottom=.19,top=.92,wspace=.24)
    for axis,(metric,ylabel,letter) in zip(axes,PANELS):
        for condition in CONDITIONS:
            individual=networks.loc[networks.metric.eq(metric)&networks.condition_label.eq(condition)]
            for _,one in individual.groupby("network_id",sort=True):
                one=one.sort_values("session")
                axis.plot(one.session,one.network_mean,color=COLORS[condition],alpha=.29,lw=.8,zorder=1)
            mean=group.loc[group.metric.eq(metric)&group.condition_label.eq(condition)].sort_values("session")
            center = mean["mean"].to_numpy(float)
            halfwidth = mean["t95_halfwidth"].to_numpy(float)
            if not np.isfinite(halfwidth).all() or (halfwidth < 0).any():
                raise ValueError(f"Invalid confidence interval: {condition}, {metric}")
            axis.fill_between(mean.session.to_numpy(float), center-halfwidth, center+halfwidth,
                              color=COLORS[condition], alpha=.11, linewidth=0, zorder=2)
            axis.plot(mean.session,mean["mean"],color=COLORS[condition],label=LABELS[condition],
                      marker=MARKERS[condition],markersize=5.770892890464477,
                      markerfacecolor="white",markeredgewidth=1.5,lw=1.8,zorder=3)
        axis.set_xlabel("Session")
        axis.set_ylabel(ylabel)
        axis.set_xlim(-.3,9.3)
        axis.set_xticks(range(0,10,2))
        axis.tick_params(direction="out",length=3)
        axis.text(-.15,1.05,letter,transform=axis.transAxes,fontsize=14,fontweight="bold",va="top")
        if metric=="rescale_factor":
            axis.set_ylim(.8,1.1)
            axis.set_yticks(np.arange(.8,1.101,.1))
        else:
            axis.set_ylim(1.2,1.9)
            axis.set_yticks(np.arange(1.2,1.901,.2))
    axes[0].axhline(1.,color="#999999",ls="--",lw=.8,zorder=0)
    handles,labels=axes[0].get_legend_handles_labels()
    axes[0].legend(handles,labels,ncol=2,loc="lower left",bbox_to_anchor=(.015,.015),fontsize=11,
                   columnspacing=.7,handlelength=1.4,borderaxespad=.2,labelspacing=.15,borderpad=.1)
    return save_figure(fig,output_dir,"main_figure_6")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE / "Plot_data/main_figure_6")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "Reproduced/Main_figs")
    args = parser.parse_args()
    configure()
    network, group = checked_data(args.data_dir)
    print(plot(network, group, args.output_dir))


if __name__ == "__main__":
    main()

"""Plot place-cell-count and pathway learning-rate-control trajectories."""
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
COUNTS = (100, 200, 300, 400)


METRICS = ("rescale_factor", "spatial_information_bits_per_spike")


PATHWAYS = ("intact", "W_bp", "W_pg", "W_gp")


PATH_COLORS = {"intact": "#222222", "W_bp": "#3B4CC0",
               "W_pg": "#D99138", "W_gp": "#218F72"}


PATH_MARKERS = {"intact": "o", "W_bp": "s", "W_pg": "^", "W_gp": "D"}


PATH_LABELS = {"intact": "Control",
               **{pathway: learning_rate_label(pathway) for pathway in PATHWAYS
                  if pathway != "intact"}}


METRIC_LABELS = {"rescale_factor": r"$F_{\mathrm{R}}$",
                 "spatial_information_bits_per_spike": "Spatial information"}


def load_data(data_dir):
    """Read network curves and group summaries from the plotting tables."""
    network = pd.read_csv(data_dir / "network_curves.csv", float_precision="round_trip")
    group = pd.read_csv(data_dir / "group_curves.csv", float_precision="round_trip")
    keys = ["count", "pathway", "network_id", "environment", "session", "metric"]
    required = keys + ["network_mean"]
    if not set(required).issubset(network.columns) or network.duplicated(keys).any():
        raise ValueError("Missing or duplicated network records")
    group_keys = ["count", "pathway", "metric", "session"]
    if not set(group_keys + ["mean"]).issubset(group.columns) or group.duplicated(group_keys).any():
        raise ValueError("Missing or duplicated group records")
    if not np.isfinite(network["network_mean"]).all() or not np.isfinite(group["mean"]).all():
        raise ValueError("Non-finite curve values")
    return network, group


def plot(network,group,output_dir):
    fig,axes=plt.subplots(4,2,figsize=(7.2,8.0))
    fig.subplots_adjust(left=.105,right=.985,bottom=.075,top=.91,hspace=.42,wspace=.31)
    for row,count in enumerate(COUNTS):
        for col,metric in enumerate(METRICS):
            ax=axes[row,col]
            for pathway in PATHWAYS:
                part=network.loc[network["count"].eq(count)&network.pathway.eq(pathway)&network.metric.eq(metric)]
                for _,trace in part.groupby("network_id"):
                    trace=trace.sort_values("session")
                    assert trace.session.tolist()==list(range(10))
                    ax.plot(trace.session,trace.network_mean,color=PATH_COLORS[pathway],alpha=.19,lw=.52,ls="-",zorder=1)
                mean=group.loc[group["count"].eq(count)&group.pathway.eq(pathway)&group.metric.eq(metric)].sort_values("session")
                assert mean.session.tolist()==list(range(10))
                ax.plot(mean.session,mean["mean"],color=PATH_COLORS[pathway],marker=PATH_MARKERS[pathway],
                        ms=4.1,markerfacecolor="white",markeredgewidth=.8,lw=1.3,zorder=3)
            if metric=="rescale_factor":
                ax.axhline(1.,color="#888888",lw=.6,ls=(0,(2,2)),zorder=0)
            ax.set_xlim(-.3,9.3)
            ax.set_xticks(range(0,10,2))
            ax.set_ylabel(METRIC_LABELS[metric])
            if row==3:
                ax.set_xlabel("Session")
            fit_panel(ax)
            ax.text(-.16,1.06,"ABCDEFGH"[row*2+col],transform=ax.transAxes,ha="left",va="bottom",
                    fontsize=10,fontweight="bold",clip_on=False)
            if col==0:
                ax.set_title(f"{count} place cells",loc="left",color="black",fontweight="bold",pad=5)
    handles=[Line2D([],[],color=PATH_COLORS[p],marker=PATH_MARKERS[p],markerfacecolor="white",
                    markeredgewidth=.8,ms=4.1,lw=1.3,label=PATH_LABELS[p]) for p in PATHWAYS]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.5,.995),ncol=4,frameon=False,
               columnspacing=1.1,handlelength=1.5)
    return save_figure(fig,output_dir,"supp_figure_16")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE / "Plot_data/supp_figure_16")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "Reproduced/Supp_figs")
    args = parser.parse_args()
    configure()
    network, group = load_data(args.data_dir)
    print(plot(network, group, args.output_dir))


if __name__ == "__main__":
    main()

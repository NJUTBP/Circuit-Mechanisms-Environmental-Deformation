"""Plot main Figure 2 from packaged rate-map arrays and curve tables."""
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
from familiarization_style import export_figure

DATA=Path(__file__).resolve().parents[1]/"Plot_data/main_figure_2"
ENVIRONMENTS=(70,80,90,100,110,120,130)
DEFORMED=(70,80,90,110,120,130)
COLORS={1:"#1783BE",2:"#DE7468",3:"#249A74"}
MODULE_MARKERS={1:"o",2:"^",3:"v"}
FIELD_COLOR,PHASE_COLOR="#1973A3","#D38123"
RATEMAP_CMAP="jet"
MARKER_SIZE=4.1

def configure() -> None:
    mpl.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold', 'mathtext.sf': 'Times New Roman', 'mathtext.tt': 'Times New Roman', 'mathtext.cal': 'Times New Roman', 'font.size': 7.3, 'axes.labelsize': 8.2, 'axes.titlesize': 7.8, 'xtick.labelsize': 7.0, 'ytick.labelsize': 7.0, 'legend.fontsize': 7.3, 'axes.linewidth': 0.65, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})

def curve(group: pd.DataFrame, population: str, metric: str, module: int, *, expected_networks: int=8) -> pd.DataFrame:
    selected = group.loc[group.population.eq(population) & group.metric.eq(metric) & group.module.eq(module) & group.length_cm.isin(DEFORMED)].set_index('length_cm').reindex(DEFORMED).reset_index()
    if len(selected) != 6 or selected[['group_mean', 'ci95_low', 'ci95_high']].isna().any().any():
        raise ValueError(f'Incomplete six-condition curve: {population}/{metric}/{module}')
    if set(selected.n_networks) != {expected_networks}:
        raise ValueError(f'Option 1 requires {expected_networks} complete independent networks')
    return selected

def draw_original_maps(fig: plt.Figure, original: dict, name: str, labels: list[str], letter: str, rect: tuple[float, float, float, float], *, label_fontsize: float=6.4) -> float:
    """Reuse the original 3 x 7 physical-size layout and imshow settings."""
    x0, _y0, x1, y1 = rect
    fig_w, fig_h = fig.get_size_inches()
    hgap, vgap = (0.004, 0.012)
    block_w, block_h = (x1 - x0, y1 - rect[1])
    row_h_by_height = (block_h - 2 * vgap) / 3
    row_h_by_width = (block_w - 6 * hgap) * fig_w / fig_h / (sum(ENVIRONMENTS) / 100)
    row_h = min(row_h_by_height, row_h_by_width)
    widths = [row_h * fig_h / fig_w * (environment / 100) for environment in ENVIRONMENTS]
    total_width = sum(widths) + 6 * hgap
    for row in range(3):
        left = x0
        bottom = y1 - (row + 1) * row_h - row * vgap
        for col, width in enumerate(widths):
            ax = fig.add_axes([left, bottom, width, row_h])
            ax.imshow(original[name][row, col], cmap=RATEMAP_CMAP, interpolation='bicubic', aspect='auto')
            ax.axis('off')
            if row == 0:
                fig.text(left + width / 2, y1 + 0.003, f'{ENVIRONMENTS[col]} cm', ha='center', va='bottom', fontsize=label_fontsize)
            if col == 0:
                fig.text(x0 - 0.009, bottom + row_h / 2, labels[row], rotation=90, ha='center', va='center', fontsize=label_fontsize)
            left += width + hgap
    fig.text(x0 - 0.028, y1 + 0.028, letter, fontsize=10.0, fontweight='bold', va='top')
    return x0 + total_width

def draw_network_curves(ax: plt.Axes, network: pd.DataFrame, population: str, metric: str, module: int, color: str, *, expected_networks: int=8) -> None:
    selected = network.loc[network.population.eq(population) & network.metric.eq(metric) & network.module.eq(module) & network.length_cm.isin(DEFORMED)]
    network_ids = sorted(selected.network_id.unique())
    if len(network_ids) != expected_networks:
        raise ValueError(f'Expected {expected_networks} independent networks: {population}/{metric}/{module}')
    for network_id in network_ids:
        values = selected.loc[selected.network_id.eq(network_id)]
        if len(values) != 6 or values.length_cm.nunique() != 6:
            raise ValueError(f'Incomplete network curve: {population}/{metric}/{module}/{network_id}')
        values = values.set_index('length_cm').reindex(DEFORMED).network_mean.to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite network curve: {population}/{metric}/{module}/{network_id}')
        ax.plot(range(6), values, color=color, linewidth=0.55, alpha=0.32, zorder=1.5)

def style_right_axis(ax: plt.Axes) -> None:
    ax.set_xlim(-0.35, 5.35)
    ax.set_xticks(range(6), [f'{length / 100:.1f}' for length in DEFORMED])
    ax.set_xlabel('Deformation ratio')
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.38, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', width=0.6, length=2.5)
def panel_values(network, group, population, metric, modules):
    net = network.loc[network.population.eq(population) & network.metric.eq(metric) & network.module.isin(modules) & network.length_cm.isin(DEFORMED)]
    summary = group.loc[group.population.eq(population) & group.metric.eq(metric) & group.module.isin(modules) & group.length_cm.isin(DEFORMED)]
    return np.r_[net.network_mean.to_numpy(float), summary.ci95_low.to_numpy(float), summary.ci95_high.to_numpy(float)]

def zero_aligned_limits(values):
    """Five labelled ticks; both place axes put zero at the same height."""
    low, high = (float(min(0, values.min())), float(max(0, values.max())))
    span = high - low
    required = max((high + 0.06 * span) / 4, -(low - 0.06 * span))
    power = 10 ** np.floor(np.log10(required))
    step = next((x * power for x in (1, 2, 2.5, 5, 10) if x * power >= required))
    return ((-step, 4 * step), np.arange(5) * step)

def summary_curve(ax,data,color,label,marker):
    x=np.arange(len(data))
    ax.fill_between(x,data.ci95_low.to_numpy(float),data.ci95_high.to_numpy(float),color=color,alpha=.16,linewidth=0,zorder=1.2)
    ax.plot(x,data.group_mean.to_numpy(float),color=color,marker=marker,markersize=MARKER_SIZE,linewidth=1.25,label=label,zorder=3)

def plot(output_dir=None):
    output=Path(output_dir) if output_dir else DATA.parents[1]/"Reproduced/Main_figs"
    network=pd.read_csv(DATA/"network_values.csv",float_precision="round_trip")
    group=pd.read_csv(DATA/"group_summary.csv",float_precision="round_trip")
    with np.load(DATA/"rate_maps.npz") as source:
        original={name:{(r,c):source[f"{name}_{r}_{c}"].copy() for r in range(3) for c in range(7)} for name in ("gc_maps","pc_maps")}
        original["p_index"]=source["place_cell_ids"].tolist()
    configure()
    mpl.rcParams.update({"font.family":"serif","font.serif":["Times New Roman"],"mathtext.fallback":None,"svg.fonttype":"none","pdf.fonttype":42})
    fig=plt.figure(figsize=(7.2,4.05),facecolor="white")
    right=draw_original_maps(fig,original,"gc_maps",["Module 1","Module 2","Module 3"],"A",(.05,.52,.62,.92),label_fontsize=8.)
    draw_original_maps(fig,original,"pc_maps",[f"cell #{x}" for x in original["p_index"]],"C",(.05,.055,.62,.455),label_fontsize=8.)
    left,width,letter=right+.058,.975-(right+.058),right+.024
    grid_ax=fig.add_axes([left,.565,width,.365])
    for module in (1,2,3):
        draw_network_curves(grid_ax,network,"grid","rescale_factor",module,COLORS[module])
        summary_curve(grid_ax,curve(group,"grid","rescale_factor",module),COLORS[module],f"Module {module}",MODULE_MARKERS[module])
    grid_ax.axhline(1,color="black",linewidth=.8,linestyle="--",zorder=1)
    grid_vals=np.r_[panel_values(network,group,"grid","rescale_factor",[1,2,3]),1]
    upper=np.ceil((grid_vals.max()+.04*np.ptp(grid_vals))/.05)*.05
    grid_ax.set_ylim(.8,upper)
    grid_ax.set_yticks([.8,1.,1.2])
    grid_ax.set_ylabel(r"$F_{\mathrm{R}}$")
    style_right_axis(grid_ax)
    grid_ax.grid(False,axis="y")
    grid_ax.legend(loc="upper left",frameon=False,handlelength=1.5,borderaxespad=.2,labelspacing=.45)
    fig.text(letter,.965,"B",fontsize=10,fontweight="bold",va="top")
    place_ax=fig.add_axes([left,.075,width,.365])
    specs=[("delta_field_length_x_cm",FIELD_COLOR,r"$\Delta L_{\mathrm{PF}}$","o"),("delta_place_phase_x_cm",PHASE_COLOR,r"$\Delta D_{\mathrm{PF}}$","^")]
    for metric,color,label,marker in specs:
        draw_network_curves(place_ax,network,"place",metric,0,color)
        summary_curve(place_ax,curve(group,"place",metric,0),color,label,marker)
    place_ax.axhline(0,color="black",linewidth=.8,linestyle="--",zorder=1)
    style_right_axis(place_ax)
    place_values=np.concatenate([panel_values(network,group,"place",metric,[0]) for metric,_,_,_ in specs])
    bounds,ticks=zero_aligned_limits(place_values)
    place_ax.set_ylim(*bounds)
    place_ax.set_yticks(np.r_[bounds[0],ticks])
    place_ax.set_ylabel(r"$\Delta L_{\mathrm{PF}},\,\Delta D_{\mathrm{PF}}$ (cm)",color="black")
    place_ax.spines["left"].set_color("black")
    place_ax.tick_params(axis="y",colors="black")
    place_ax.grid(False,axis="y")
    handles=[Line2D([],[],color=color,marker=marker,markersize=MARKER_SIZE,label=label) for _,color,label,marker in specs]
    place_ax.legend(handles=handles,loc="upper left",frameon=False,handlelength=1.5,borderaxespad=.2,labelspacing=.45)
    fig.text(letter,.475,"D",fontsize=10,fontweight="bold",va="top")
    export_figure(fig,output/"main_figure_2")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    plot(parser.parse_args().output_dir)

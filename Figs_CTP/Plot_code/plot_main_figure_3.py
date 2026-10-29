"""Plot main Figure 3 from the packaged phase, field and rate tables."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from familiarization_style import export_figure

DATA=Path(__file__).resolve().parents[1]/"Plot_data/main_figure_3"
COLOR_X = "#B8BCFD"
COLOR_Y = "#DADCFB"
COLOR_FAMILIAR = "#D3D3D3"
COLOR_DEFORMATION = "#B8BCFD"
COLOR_NETWORK = "#4A4A4A"
COLOR_ZERO = "#9A9A9A"
MODULE_COLORS = {1: "#1783BE", 2: "#DE7468", 3: "#249A74"}
MODULE_LIGHT = {1: "#8FC6DE", 2: "#F0A69E", 3: "#93CFB9"}
MODULE_MARKERS = {1: "o", 2: "^", 3: "v"}
MODULE_MARKER_SIZE = 4.1

def configure_style() -> None:
    mpl.rcParams.update({'font.family': 'Times New Roman', 'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold', 'mathtext.sf': 'Times New Roman', 'mathtext.tt': 'Times New Roman', 'mathtext.cal': 'Times New Roman', 'font.size': 9.2, 'axes.labelsize': 9.2, 'axes.titlesize': 10.4, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.0, 'axes.linewidth': 0.65, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.6, 'ytick.major.size': 2.6, 'lines.linewidth': 1.1, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})

def clean_axis(ax: plt.Axes) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', pad=2)

def panel_label(ax: plt.Axes, label: str, x: float=-0.2, y: float=1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha='left', va='bottom', fontsize=11.2, fontweight='bold', clip_on=False)

def star(q: float) -> str:
    if q < 0.001:
        return '***'
    if q < 0.01:
        return '**'
    if q < 0.05:
        return '*'
    return 'ns'

def add_bracket(ax: plt.Axes, x1: float, x2: float, y: float, height: float, label: str, fontsize: float=8.0) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color='black', lw=0.65, clip_on=False)
    ax.text((x1 + x2) / 2, y + height * 1.15, label, ha='center', va='bottom', fontsize=fontsize)

def set_ylim_with_brackets(ax: plt.Axes, minimum: float, bracket_top: float, pad_fraction: float=0.08) -> None:
    span = max(bracket_top - minimum, 1e-09)
    ax.set_ylim(minimum, bracket_top + pad_fraction * span)

def bar_with_networks(ax: plt.Axes, values: pd.DataFrame, order: list[str], colors: list[str], width: float=0.62, connect: bool=True, positions: np.ndarray | None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.arange(len(order), dtype=float) if positions is None else np.asarray(positions, dtype=float)
    if len(positions) != len(order):
        raise ValueError('Each bar group requires one x position')
    pivot = values.pivot(index='network_id', columns='group', values='value').reindex(columns=order)
    means = pivot.mean(axis=0).to_numpy(float)
    sems = pivot.sem(axis=0, ddof=1).to_numpy(float)
    ax.bar(positions, means, width=width, color=colors, edgecolor='black', linewidth=0.55, yerr=sems, error_kw={'elinewidth': 0.7, 'capsize': 2.0, 'capthick': 0.7}, zorder=2)
    if connect:
        for _, row in pivot.iterrows():
            ax.plot(positions, row.to_numpy(float), color=COLOR_NETWORK, alpha=0.3, lw=0.55, zorder=3)
    for pos, name in zip(positions, order):
        y = pivot[name].to_numpy(float)
        jitter = np.linspace(-0.055, 0.055, len(y))
        ax.scatter(np.full(len(y), pos) + jitter, y, s=9, facecolor='white', edgecolor=COLOR_NETWORK, linewidth=0.45, zorder=4)
    ax.set_xticks(positions, order)
    return (positions, means, sems)

def plot_abc(ax: plt.Axes, values: pd.DataFrame, summary: pd.DataFrame, module: int, letter: str, show_legend: bool=False) -> None:
    sub_summary = summary.loc[summary['module'] == module]
    sub_values = values.loc[values['module'] == module]
    for axis_name, label, color in [('x', 'Deformed', MODULE_COLORS[module]), ('y', 'Undeformed', MODULE_LIGHT[module])]:
        s = sub_summary.loc[sub_summary['axis'] == axis_name].sort_values('deformation_ratio')
        x = s['deformation_ratio'].to_numpy(float)
        mean = s['mean'].to_numpy(float)
        sem = s['sem'].to_numpy(float)
        wide = sub_values.loc[sub_values['axis'] == axis_name].pivot(index='network_id', columns='deformation_ratio', values='value').reindex(columns=x)
        for _, row in wide.iterrows():
            ax.plot(x, row.to_numpy(float), color=color, alpha=0.1, lw=0.45, zorder=1)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.22, linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, marker=MODULE_MARKERS[module], ms=MODULE_MARKER_SIZE, mfc=color if axis_name == 'x' else 'white', mec=color, label=label, zorder=3)
    ax.axhline(0, color=COLOR_ZERO, lw=0.55, ls=(0, (2, 2)), zorder=0)
    ax.axvline(1.0, color=COLOR_ZERO, lw=0.55, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(0.62, 1.38)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    ax.set_xlabel('Deformation ratio')
    ax.set_ylabel('$R_{\\mathrm{phase}}$')
    ax.set_title(f'Module {module}', color=MODULE_COLORS[module], fontweight='bold', pad=4)
    panel_label(ax, letter)
    clean_axis(ax)
    if show_legend:
        ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.0), handlelength=1.1, fontsize=7.4)

def plot_d(ax: plt.Axes, values: pd.DataFrame, stats: pd.DataFrame, significance_column: str='q_bh_within_panel_family') -> None:
    sub = values.loc[values['panel'] == 'D']
    modules = [1, 2, 3]
    centers = np.arange(3, dtype=float)
    offsets = [-0.18, 0.18]
    width = 0.32
    group_info = [('Deformed axis (x)', offsets[0]), ('Undeformed axis (y)', offsets[1])]
    all_high = []
    all_low = []
    for label, offset in group_info:
        group = sub.loc[sub['group'] == label]
        means = []
        sems = []
        for module in modules:
            x = group.loc[group['module'] == module, 'value'].to_numpy(float)
            means.append(np.mean(x))
            sems.append(np.std(x, ddof=1) / np.sqrt(len(x)))
            all_high.extend(x.tolist())
            all_low.extend(x.tolist())
        pos = centers + offset
        colors = [MODULE_COLORS[m] if label == 'Deformed axis (x)' else MODULE_LIGHT[m] for m in modules]
        ax.bar(pos, means, width=width, color=colors, edgecolor='black', linewidth=0.55, yerr=sems, error_kw={'elinewidth': 0.7, 'capsize': 2, 'capthick': 0.7}, label=label, zorder=2)
    for module, center in zip(modules, centers):
        wide = sub.loc[sub['module'] == module].pivot(index='network_id', columns='group', values='value')
        xa, xb = (center + offsets[0], center + offsets[1])
        for _, row in wide.iterrows():
            ya = row['Deformed axis (x)']
            yb = row['Undeformed axis (y)']
            ax.plot([xa, xb], [ya, yb], color=COLOR_NETWORK, alpha=0.3, lw=0.55, zorder=3)
            ax.scatter([xa, xb], [ya, yb], s=8, marker='o', facecolor='white', edgecolor=COLOR_NETWORK, lw=0.4, zorder=4)
    data_min = min(all_low)
    data_max = max(all_high)
    data_span = max(data_max - data_min, 0.02)
    base = data_max + 0.045 * data_span
    step = 0.075 * data_span
    for module, center in zip(modules, centers):
        q = float(stats.loc[(stats['panel'] == 'D') & (stats['module'] == module), significance_column].iloc[0])
        add_bracket(ax, center + offsets[0], center + offsets[1], base, step * 0.35, star(q))
    ax.axhline(0, color=COLOR_ZERO, lw=0.55, ls=(0, (2, 2)), zorder=0)
    ax.set_xticks(centers, ['Module 1', 'Module 2', 'Module 3'])
    ax.set_ylabel('$R_{\\mathrm{phase}}$')
    set_ylim_with_brackets(ax, min(data_min - 0.12 * data_span, 0), base + step * 0.8)
    panel_label(ax, 'D', x=-0.1)
    clean_axis(ax)

def plot_e(ax: plt.Axes, values: pd.DataFrame, stats: pd.DataFrame, significance_column: str='q_bh_within_panel_family') -> None:
    sub = values.loc[values['panel'] == 'E']
    order = ['Deformed axis (x)', 'Undeformed axis (y)']
    positions, means, sems = bar_with_networks(ax, sub, order, [COLOR_X, COLOR_Y], width=0.38, positions=np.array([0.15, 0.85]))
    ax.set_xlim(-0.4, 1.4)
    q = float(stats.loc[stats['panel'] == 'E', significance_column].iloc[0])
    top = max(sub['value'].max(), np.max(means + sems))
    bottom = min(sub['value'].min(), 0)
    span = max(top - bottom, 0.02)
    y = top + 0.08 * span
    if star(q) != 'ns':
        add_bracket(ax, positions[0], positions[1], y, 0.04 * span, star(q))
    ax.axhline(0, color=COLOR_ZERO, lw=0.55, ls=(0, (2, 2)), zorder=0)
    ax.set_xticklabels(['Deformed\nAxis', 'Undeformed\nAxis'])
    ax.set_ylabel('$R_{\\mathrm{phase}}$')
    set_ylim_with_brackets(ax, bottom - 0.08 * span, y + 0.08 * span)
    panel_label(ax, 'E')
    clean_axis(ax)

def plot_f(ax: plt.Axes, values: pd.DataFrame, stats: pd.DataFrame, field_label: str, significance_column: str='q_bh_within_panel_family') -> None:
    sub = values.loc[values['panel'] == 'F']
    order = ['Familiar x', 'Deformed x', 'Deformed y']
    positions, means, sems = bar_with_networks(ax, sub, order, [COLOR_FAMILIAR, COLOR_X, COLOR_Y], connect=True)
    top = max(sub['value'].max(), np.max(means + sems))
    bottom = min(sub['value'].min(), 0)
    span = max(top - bottom, 1.0)
    q_xy = float(stats.loc[stats['contrast'] == 'deformed_x_minus_deformed_y', significance_column].iloc[0])
    q_xf = float(stats.loc[stats['contrast'] == 'deformed_x_minus_familiar_x', significance_column].iloc[0])
    y1 = top + 0.05 * span
    y2 = top + 0.15 * span
    add_bracket(ax, positions[1], positions[2], y1, 0.04 * span, star(q_xy))
    add_bracket(ax, positions[0], positions[1], y2, 0.04 * span, star(q_xf))
    ax.set_xticklabels(['Familiar', 'Deformed\naxis', 'Undeformed\naxis'])
    ax.tick_params(axis='x', labelsize=8.0)
    ax.set_ylabel(field_label)
    set_ylim_with_brackets(ax, bottom - 0.05 * span, y2 + 0.05 * span)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    panel_label(ax, 'F')
    clean_axis(ax)

def plot_g(ax: plt.Axes, values: pd.DataFrame, stats: pd.DataFrame, measure: str, title: str, show_panel_label: bool, significance_column: str='q_bh_within_panel_family') -> None:
    sub = values.loc[(values['panel'] == 'G') & (values['measure'] == measure)]
    order = ['Familiar', 'Deformation']
    positions, means, sems = bar_with_networks(ax, sub, order, [COLOR_FAMILIAR, COLOR_DEFORMATION], width=0.38, connect=True, positions=np.array([0.15, 0.85]))
    ax.set_xlim(-0.4, 1.4)
    q = float(stats.loc[stats['contrast'] == f'{measure}_deformation_minus_familiar', significance_column].iloc[0])
    top = max(sub['value'].max(), np.max(means + sems))
    bottom = min(sub['value'].min(), 0)
    span = max(top - bottom, 0.1)
    y = top + 0.08 * span
    add_bracket(ax, positions[0], positions[1], y, 0.04 * span, star(q))
    ax.set_xticklabels(['Familiar', 'Deformation'], rotation=0, ha='center')
    if show_panel_label:
        ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(title, pad=4)
    set_ylim_with_brackets(ax, min(0, bottom - 0.04 * span), y + 0.08 * span)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    if show_panel_label:
        panel_label(ax, 'G')
    clean_axis(ax)

def plot(output_dir=None):
    output=Path(output_dir) if output_dir else DATA.parents[1]/"Reproduced/Main_figs"
    configure_style()
    abc_values=pd.read_csv(DATA/"panels_ABC_network_values.csv")
    abc_summary=pd.read_csv(DATA/"panels_ABC_group_summary.csv")
    panel_values=pd.read_csv(DATA/"panels_DEFG_network_values.csv")
    statistics=pd.read_csv(DATA/"network_level_statistics.csv")
    fig=plt.figure(figsize=(7.2,8.0))
    grid=fig.add_gridspec(3,6,height_ratios=[1.,1.05,1.05],left=.085,right=.975,bottom=.07,top=.95,wspace=1.12,hspace=.38)
    axes=[fig.add_subplot(grid[0,0:2])]
    axes.extend([fig.add_subplot(grid[0,2:4],sharex=axes[0]),fig.add_subplot(grid[0,4:6],sharex=axes[0])])
    axes.extend([fig.add_subplot(grid[1,0:4]),fig.add_subplot(grid[1,4:6]),fig.add_subplot(grid[2,0:2]),fig.add_subplot(grid[2,2:4]),fig.add_subplot(grid[2,4:6])])
    for i in range(3): plot_abc(axes[i],abc_values,abc_summary,i+1,"ABC"[i],show_legend=True)
    plot_d(axes[3],panel_values,statistics,"p_raw")
    plot_e(axes[4],panel_values,statistics,"p_raw")
    plot_f(axes[5],panel_values,statistics,r"$L_{\mathrm{G}}$ (cm)","p_raw")
    plot_g(axes[6],panel_values,statistics,"peak_rate_hz","Peak rate",True,"p_raw")
    plot_g(axes[7],panel_values,statistics,"mean_rate_hz","Mean rate",False,"p_raw")
    for ax in axes[:3]:
        ax.set_ylim(-.12,.78)
        ax.set_yticks([0.,.2,.4,.6])
    for ax in axes[3:5]:
        ax.set_ylim(-.045,.40)
        ax.set_yticks([0.,.1,.2,.3,.4])
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(0)
            label.set_ha("center")
    export_figure(fig,output/"main_figure_3",tight=False)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    plot(parser.parse_args().output_dir)

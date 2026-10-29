"""Shared layout and curve drawing for familiarization figures."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENVS = ("compressed_80", "elongated_120")
SESSIONS = np.arange(10)
MODULE_COLORS = {1: "#1783BE", 2: "#DE7468", 3: "#249A74"}
MODULE_MARKERS = {1: "o", 2: "^", 3: "v"}
MODULE_MARKER_SIZE = 4.1
PLACE_COLORS = {"length": "#1973A3", "phase": "#D38123"}

def export_figure(fig, stem, tight=True):
    stem.parent.mkdir(parents=True,exist_ok=True)
    options={"bbox_inches":"tight"} if tight else {}
    for ext in ("png","pdf","svg","tiff"):
        fig.savefig(stem.with_suffix("."+ext),dpi=600,facecolor="white",**options)
    plt.close(fig)

def configure() -> None:
    mpl.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold', 'mathtext.sf': 'Times New Roman', 'mathtext.tt': 'Times New Roman', 'mathtext.cal': 'Times New Roman', 'mathtext.fallback': None, 'font.size': 9.2, 'axes.labelsize': 9.5, 'axes.titlesize': 10.0, 'xtick.labelsize': 8.3, 'ytick.labelsize': 8.3, 'legend.fontsize': 7.2, 'axes.linewidth': 0.65, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})

def select(frame: pd.DataFrame, environment: str, population: str, metric: str, module: int) -> pd.DataFrame:
    return frame.loc[frame.environment.eq(environment) & frame.population.eq(population) & frame.metric.eq(metric) & frame.module.eq(module)].sort_values('session')

def limits(values: list[float], reference: float) -> tuple[float, float]:
    finite = np.asarray(values + [reference], float)
    finite = finite[np.isfinite(finite)]
    lo, hi = (float(finite.min()), float(finite.max()))
    margin = max(0.12 * (hi - lo), 0.025 if reference == 1 else 0.24)
    return (lo - margin, hi + margin)

def manuscript_axes(height: float=8.64) -> tuple[plt.Figure, np.ndarray]:
    """Four A-D groups, each with two stacked measures, as in the manuscript."""
    fig = plt.figure(figsize=(7.2, height))
    outer = fig.add_gridspec(2, 1, left=0.13, right=0.96, top=0.94, bottom=0.06, hspace=0.28)
    axes = np.empty((4, 2), dtype=object)
    for group in range(2):
        inner = outer[group].subgridspec(2, 2, hspace=0.08, wspace=0.22)
        for col in range(2):
            axes[2 * group, col] = fig.add_subplot(inner[0, col])
            axes[2 * group + 1, col] = fig.add_subplot(inner[1, col], sharex=axes[2 * group, col])
    for group, name in enumerate(('Grid cells', 'Place cells')):
        fig.text(0.018, (0.73, 0.27)[group], name, rotation=90, rotation_mode='anchor', ha='center', va='center', fontsize=9.5, fontweight='bold')
    return (fig, axes)

def style_axis(ax: plt.Axes, reference: float) -> None:
    ax.axhline(reference, color='#6B7175', linewidth=0.8, linestyle='--', zorder=0)
    ax.set_xlim(-0.45, 9.45)
    ax.set_xticks(SESSIONS)
    ax.tick_params(direction='out', length=2.7, width=0.6)

def draw_main_panel(ax: plt.Axes, network: pd.DataFrame, group: pd.DataFrame, env: str, population: str, metric: str, reference: float, place_key: str | None) -> list[float]:
    values = []
    modules = (1, 2, 3) if population == 'grid' else (0,)
    for module in modules:
        points = select(network, env, population, metric, module)
        curve = select(group, env, population, metric, module)
        if len(curve) != 10 or set(curve.n_networks) != {8}:
            raise ValueError(f'Incomplete main curve: {env}/{population}/{metric}/{module}')
        if len(points) != 80 or points.network_id.nunique() != 8:
            raise ValueError(f'Incomplete main network values: {env}/{population}/{metric}/{module}')
        color = MODULE_COLORS[module] if population == 'grid' else PLACE_COLORS[place_key]
        for _, individual in points.groupby('network_id', sort=True):
            if len(individual) != 10:
                raise ValueError('Network has incomplete sessions')
            ax.plot(individual.session, individual.network_mean, color=color, linewidth=0.52, alpha=0.16, zorder=1)
            values.extend(individual.network_mean.to_numpy(float).tolist())
        x = curve.session.to_numpy(int)
        y = curve.group_mean.to_numpy(float)
        low = curve.ci95_low.to_numpy(float)
        high = curve.ci95_high.to_numpy(float)
        if not (np.isfinite(y).all() and np.isfinite(low).all() and np.isfinite(high).all()):
            raise ValueError(f'Nonfinite group curve: {env}/{metric}/{module}')
        ax.fill_between(x, low, high, color=color, alpha=0.17, linewidth=0, zorder=2)
        ax.plot(x, y, linestyle='-', marker=MODULE_MARKERS[module] if population == 'grid' else 'o', color=color, linewidth=1.65, markersize=MODULE_MARKER_SIZE, label=f'Module {module}' if population == 'grid' else 'Network mean', zorder=3)
        values.extend(low.tolist() + high.tolist())
    style_axis(ax, reference)
    return values

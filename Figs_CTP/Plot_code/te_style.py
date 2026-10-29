"""Times New Roman typography matched to the surrounding Figs package."""
from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
import numpy as np

CONFIG = {'font_family':'Times New Roman','font_size_pt':8.5,'axes_label_size_pt':9.,'tick_size_pt':8.}

MAIN = '#18576B'
GRAY = '#ABB0B7'
STATE_COLORS = {'approach': '#674091', 'departure': '#11858B'}
ENV_COLORS = {'compressed_70': '#1F4E79', 'compressed_80': '#367FA6',
              'compressed_90': '#70A6B8', 'elongated_110': '#CB9164',
              'elongated_120': '#B65E46', 'elongated_130': '#823A44'}
COMPONENT_COLORS = {'GP': '#C46B2D', 'PG': '#315C8A'}
COMPONENT_MARKERS = {'GP': 's', 'PG': 'D'}
CONTROL_COLORS = {'real': '#395160', 'cross': '#9DA5A9'}
STATE_AXIS_COLORS = {('approach', 'x'): '#674091', ('departure', 'x'): '#11858B',
                     ('approach', 'y'): '#A182B5', ('departure', 'y'): '#78B5B0'}
LABELS = {'C_delta_I_D': r'$C_{\Delta I_D}$', 'C_delta_T': r'$C_{\Delta T}$ (bits/sample)',
          'delta_I_D': r'$\Delta I_D$', 'delta_T': r'$\Delta T$ (bits/sample)'}


def configure():
    font = findfont(CONFIG['font_family'], fallback_to_default=False)
    plt.rcParams.update({
        'font.family': CONFIG['font_family'],
        'mathtext.fontset': 'custom', 'mathtext.rm': 'Times New Roman',
        'mathtext.it': 'Times New Roman:italic', 'mathtext.bf': 'Times New Roman:bold',
        'font.size': CONFIG['font_size_pt'], 'axes.labelsize': CONFIG['axes_label_size_pt'],
        'axes.titlesize': 9.4, 'xtick.labelsize': CONFIG['tick_size_pt'],
        'ytick.labelsize': CONFIG['tick_size_pt'], 'legend.fontsize': 8.,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .65,
        'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    })
    return font


def scale(metric):
    return 1. if 'I_D' in metric else 1000.


def limits(*values, top=.25, bottom=.10):
    v = np.concatenate([np.asarray(x, float).ravel() for x in values] + [np.array([0.])])
    v = v[np.isfinite(v)]
    low, high = float(v.min()), float(v.max())
    span = max(high - low, 1e-4)
    return low - bottom * span, high + top * span, high, span


def zero(ax):
    ax.axhline(0., color='#80858C', lw=.7, ls='--', zorder=0)
    ax.tick_params(pad=2.)


def panel_label(ax, letter, x=-.19, y=1.10):
    ax.text(x, y, letter.upper(), transform=ax.transAxes, fontsize=10.,
            fontweight='bold', ha='left', va='bottom')


def ylabel(ax, metric, label=None):
    # A >=10.5 pt parent keeps the nested D in C_(Delta I_D) above 5 pt.
    ax.set_ylabel(label or LABELS[metric], fontsize=11. if metric.startswith('C_') else 9.)
    if scale(metric) == 1000.:
        ax.text(0., 1.012, '×10⁻³', transform=ax.transAxes, ha='left', va='bottom', fontsize=8.)


def interval_series(ax, x, stats, metric, color, label=None, marker='o', linestyle='-', shading=False):
    factor = scale(metric)
    x = np.asarray(x, float)
    y, lo, hi = [stats[k].to_numpy(float) * factor for k in ('mean', 'ci_low', 'ci_high')]
    ax.plot(x, y, color=color, label=label, marker=marker, ms=4., lw=1.35, ls=linestyle, zorder=4)
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if shading:
        ax.fill_between(x, lo, hi, color=color, alpha=.13, lw=0, zorder=2)
    elif valid.any():
        ax.errorbar(x[valid], y[valid], yerr=np.vstack([y[valid] - lo[valid], hi[valid] - y[valid]]),
                    fmt='none', ecolor=color, elinewidth=1., capsize=2., zorder=3)


def p_text(p, adjustment='none'):
    symbol = 'p' if adjustment == 'none' else 'q6'
    return f'{symbol}=NA' if not np.isfinite(p) else f'{symbol}={p:.4f}' + (' *' if p < .05 else '')

"""Shared numerical-data drawing helpers."""
from pathlib import Path
import io
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.container import ErrorbarContainer
from PIL import Image
import te_style as st

ROOT=Path(__file__).resolve().parents[1]
ENVS=('compressed_70','compressed_80','compressed_90','elongated_110','elongated_120','elongated_130')
WIDTHS=(70,80,90,110,120,130)
PARAMETERS=('pca_components','lag_frames','source_history','target_history')
DEFAULTS=dict(pca_components=2,lag_frames=5,source_history=1,target_history=1)
STATES=(('approach','x'),('departure','x'),('approach','y'),('departure','y'))

def load_table(figure, name):
    return pd.read_csv(ROOT/'Plot_data'/figure/(name+'.csv'),float_precision='round_trip')

def export_figure(fig, figure, output_dir=None):
    role='Main_figs' if figure.startswith('main_') else 'Supp_figs'
    output=Path(output_dir) if output_dir else ROOT/'Reproduced'/role
    output.mkdir(parents=True,exist_ok=True)
    stem=output/figure
    fig.savefig(stem.with_suffix('.png'),dpi=600)
    fig.savefig(stem.with_suffix('.pdf'),dpi=600)
    fig.savefig(stem.with_suffix('.svg'),dpi=600)
    buffer=io.BytesIO()
    fig.savefig(buffer,format='png',dpi=600)
    buffer.seek(0)
    with Image.open(buffer) as image:
        image.save(stem.with_suffix('.tiff'),compression='tiff_lzw',dpi=(600,600))
    st.plt.close(fig)
    return stem

def cli(draw):
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    print(draw(args.output_dir))

def movement_handles():
    return [Line2D([], [], color=st.STATE_COLORS[s], marker='o' if s == 'approach' else '^', lw=0, label=s.capitalize()) for s in ('approach', 'departure')]

def direction_handles():
    return [Line2D([], [], color=st.COMPONENT_COLORS[d], marker=st.COMPONENT_MARKERS[d], ls='-' if d == 'GP' else '--', lw=1.2, label='G→P' if d == 'GP' else 'P→G') for d in ('GP', 'PG')]

def environment_handles():
    return [Line2D([], [], color=st.ENV_COLORS[e], lw=1.2, marker='o', ms=3.0, label=e.rsplit('_', 1)[1] + ' cm') for e in ENVS]

def shared_legend(fig, handles, y, ncol, fontsize=8.0):
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.54, y), ncol=ncol, columnspacing=1.5 if ncol == 2 else 0.9, handlelength=1.5, fontsize=fontsize)

def robustness_panels(axes, t, metric, letters):
    frame = t['ofat_statistics']
    n_requested = t['ofat_values'].network_id.nunique()
    panels, plot_stats = ([], [])
    for ax, parameter, letter in zip(axes, PARAMETERS, letters):
        part = frame.loc[frame.parameter.eq(parameter) & frame.metric.eq(metric)].copy()
        reference = frame.loc[frame.candidate_id.eq('reference') & frame.metric.eq(metric)].copy()
        reference['parameter'], reference['parameter_value'] = (parameter, DEFAULTS[parameter])
        part = pd.concat([part, reference], ignore_index=True)
        part['x_display'] = pd.to_numeric(part.parameter_value) * (10 if parameter == 'lag_frames' else 1)
        part['panel'] = letter
        plot_stats.append(part)
        for env in ENVS:
            summary = part.loc[part.environment.eq(env)].sort_values('x_display')
            x = summary.x_display.to_numpy(float)
            y, lo, hi = [summary[c].to_numpy(float) * st.scale(metric) for c in ('mean', 'ci_low', 'ci_high')]
            color = st.ENV_COLORS[env]
            ax.plot(x, y, lw=1.2, color=color)
            ax.fill_between(x, lo, hi, color=color, alpha=0.08, lw=0)
            full = np.isfinite(y) & summary.n.eq(n_requested).to_numpy()
            reduced = np.isfinite(y) & summary.n.lt(n_requested).to_numpy()
            ax.scatter(x[full], y[full], color=color, s=15, zorder=3)
            ax.scatter(x[reduced], y[reduced], facecolors='white', edgecolors=color, s=18, lw=0.9, zorder=4)
            assert (np.isfinite(y) == summary.n.gt(0)).all()
        ax.axvline(DEFAULTS[parameter] * (10 if parameter == 'lag_frames' else 1), ls=':', lw=0.8, color='#63696F')
        st.zero(ax)
        ax.set_title({'pca_components': 'PCA dimensions', 'lag_frames': 'Prediction delay', 'source_history': 'Source history', 'target_history': 'Target history'}[parameter], pad=8.0)
        st.ylabel(ax, metric)
        ax.set_xlabel({'pca_components': 'Components per population', 'lag_frames': 'Prediction delay (ms)', 'source_history': 'Source-history frames', 'target_history': 'Target-history frames'}[parameter])
        ax.set_xticks(list(range(1, 11)) if parameter == 'pca_components' else [10, 50, 100, 150, 200] if parameter == 'lag_frames' else [1, 2, 3, 5])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        st.panel_label(ax, letter)
        panels.append(dict(panel=letter, source_panel_family='F3', parameter=parameter, metric=metric, n_min=int(part.n.min()), n_max=int(part.n.max()), missing_points=int(part.n.eq(0).sum())))
    return (panels, pd.concat(plot_stats, ignore_index=True))

def component_panels(axes, t, letters):
    data, stats = (t['f2_values'], t['component_statistics'])
    low, high, _, _ = st.limits(data[['delta_TE_GP', 'delta_TE_PG']].to_numpy() * 1000, stats[['ci_low', 'ci_high']].to_numpy() * 1000, top=0.1)
    panels = []
    for ax, (state, axis), letter in zip(axes, STATES, letters):
        values = data.loc[data.state.eq(state) & data.wall_axis.eq(axis)]
        for direction, ls in [('GP', '-'), ('PG', '--')]:
            field = 'delta_TE_' + direction
            for _, group in values.groupby('network_id'):
                y = group.set_index('environment')[field].reindex(ENVS).to_numpy() * 1000
                ax.plot(WIDTHS, y, color=st.GRAY, lw=0.5, alpha=0.34, ls=ls)
            s = stats.loc[stats.metric.eq(field) & stats.state.eq(state) & stats.wall_axis.eq(axis)].set_index('environment').reindex(ENVS)
            st.interval_series(ax, WIDTHS, s, field, st.COMPONENT_COLORS[direction], marker=st.COMPONENT_MARKERS[direction], linestyle=ls)
        st.zero(ax)
        ax.set_ylim(low, high)
        ax.set_xticks(WIDTHS)
        ax.set_xlim(65, 135)
        ax.set_xlabel('Environment width (cm)')
        st.ylabel(ax, 'TE', 'ΔTE (bits/sample)')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(f"{state.capitalize()} | {('Deformed' if axis == 'x' else 'Non-deformed')} axis", pad=8.0)
        st.panel_label(ax, letter)
        panels.append(dict(panel=letter, source_panel_family='F4', state=state, wall_axis=axis, role='Directional TE decomposition'))
    return panels

def cross_panels(axes, t, letters):
    values, stats = (t['cross_global_values'], t['cross_global_statistics'])
    panels = []
    for ax, direction, letter in zip(axes, ('grid_to_place', 'place_to_grid'), letters):
        group = values.loc[values.endpoint.eq('absolute') & values.direction.eq(direction)]
        s = stats.loc[stats.endpoint.eq('absolute') & stats.direction.eq(direction)].iloc[0]
        for r in group.itertuples():
            ax.plot([0, 1], np.asarray([r.matched_real, r.cross]) * 1000, color=st.GRAY, lw=0.8, marker='o', ms=3.0)
        for x, label in [(0, 'real'), (1, 'cross')]:
            mapped = pd.DataFrame({k: [s[label + '_' + k]] for k in ('mean', 'ci_low', 'ci_high')})
            st.interval_series(ax, [x], mapped, 'TE', st.CONTROL_COLORS[label], marker='o' if x == 0 else 'v')
        low, high, peak, span = st.limits(group[['matched_real', 'cross']].to_numpy() * 1000, [s.real_ci_low * 1000, s.real_ci_high * 1000, s.cross_ci_low * 1000, s.cross_ci_high * 1000], top=0.35)
        ax.text(0.5, peak + 0.15 * span, st.p_text(s.p_two_sided), ha='center', fontsize=8.0)
        st.zero(ax)
        ax.set_ylim(low, high)
        ax.set_xlim(-0.35, 1.35)
        ax.set_xticks([0, 1], ['Matched real', 'Cross-trajectory'])
        st.ylabel(ax, 'TE', 'TE (bits/sample)')
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title('G→P' if direction == 'grid_to_place' else 'P→G', pad=8.0)
        st.panel_label(ax, letter, y=1.05)
        panels.append(dict(panel=letter, source_panel_family='F5', endpoint='absolute', direction=direction))
    return panels

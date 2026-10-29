"""Figure 6: familiarization time courses and paired Session-9 effects.

A/C: network means with pointwise two-sided 95% Student-t confidence bands.
B/D: blockade minus matched Control, with 95% paired-mean confidence intervals.
"""
from pathlib import Path
import argparse
import csv
import numpy as np
from scipy import stats
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PACKAGE = Path(__file__).resolve().parents[1]
CONDITIONS = ['Intact', 'Eta_bp=0', 'Eta_pg=0', 'Eta_gp=0']

NAMES = {'Intact': 'Control', 'Eta_bp=0': r'$\eta_{\mathrm{PB}}=0$',
         'Eta_pg=0': r'$\eta_{\mathrm{GP}}=0$', 'Eta_gp=0': r'$\eta_{\mathrm{PG}}=0$'}

PATHWAYS = [NAMES[c] for c in CONDITIONS[1:]]

COLORS = {'Intact': '#2B2B2B', 'Eta_bp=0': '#0072B2',
          'Eta_pg=0': '#D55E00', 'Eta_gp=0': '#009E73'}

MARKERS = {'Intact': 'o', 'Eta_bp=0': 's', 'Eta_pg=0': '^', 'Eta_gp=0': 'D'}

METRICS = ['rescale_factor', 'spatial_information_bits_per_spike']

WIDTH_MM, HEIGHT_MM = 232, 82


def load_and_verify(data_dir):
    with (data_dir / 'network_curves.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 320
    v = {}
    for row in rows:
        key = (row['condition_label'], row['metric'], int(row['network_id']), int(row['session']))
        assert key not in v
        v[key] = float(row['network_mean'])
    expected = {(c, m, n, s) for c in CONDITIONS for m in METRICS
                for n in range(4) for s in range(10)}
    assert set(v) == expected and np.isfinite(list(v.values())).all()
    with (data_dir / 'endpoint_effects.csv').open(encoding='utf-8-sig', newline='') as f:
        effects = list(csv.DictReader(f))
    assert len(effects) == 6
    paired = []
    for e in effects:
        c, m = e['condition'], e['metric']
        differences = np.array([v[c, m, n, 9] - v['Intact', m, n, 9] for n in range(4)])
        mean = differences.mean()
        half = stats.t.ppf(.975, 3) * differences.std(ddof=1) / np.sqrt(4)
        assert np.allclose([mean, mean-half, mean+half],
                           [float(e[k]) for k in ['mean_paired_difference', 'ci95_low', 'ci95_high']],
                           rtol=0, atol=1e-12)
        for n, difference in enumerate(differences):
            paired.append({'condition': c, 'metric': m, 'network_id': n,
                           'session': 9, 'blockade_minus_control': float(difference)})
    assert len(paired) == 24
    return v, effects, paired


def time_course_intervals(v):
    rows = []
    for m in METRICS:
        for c in CONDITIONS:
            values = np.array([[v[c, m, n, s] for s in range(10)] for n in range(4)])
            mean = values.mean(axis=0)
            half = stats.t.ppf(.975, 3) * values.std(axis=0, ddof=1) / np.sqrt(4)
            lo, hi = stats.t.interval(.95, df=3, loc=mean, scale=stats.sem(values, axis=0))
            assert np.allclose(lo, mean-half, rtol=0, atol=1e-12)
            assert np.allclose(hi, mean+half, rtol=0, atol=1e-12)
            for s in range(10):
                rows.append({'condition': c, 'metric': m, 'session': s,
                             'n_networks': 4, 'network_mean': float(mean[s]),
                             'ci95_low': float(lo[s]), 'ci95_high': float(hi[s])})
    assert len(rows) == 80
    return rows


def draw(v, effects, time_intervals, output_dir):
    mpl.rcParams.update({
        'font.family': ['Times New Roman', 'Arial'], 'mathtext.fontset': 'custom',
        'mathtext.rm': 'Times New Roman', 'mathtext.it': 'Times New Roman:italic',
        'mathtext.bf': 'Times New Roman:bold', 'font.size': 10,
        'axes.labelsize': 10, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'legend.fontsize': 9.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .8, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
        'legend.frameon': False, 'axes.unicode_minus': True,
    })
    fig = plt.figure(figsize=(WIDTH_MM/25.4, HEIGHT_MM/25.4), facecolor='white')
    # The endpoint panels are 27 mm wide; the time-course panels remain 54 mm wide.
    boxes = [(15, 18, 54, 51), (86, 18, 27, 51),
             (133, 18, 54, 51), (203, 18, 27, 51)]
    axes = [fig.add_axes([x/WIDTH_MM, y/HEIGHT_MM, w/WIDTH_MM, h/HEIGHT_MM])
            for x, y, w, h in boxes]
    time_axes = [axes[0], axes[2]]
    bands = []
    for ax, m in zip(time_axes, METRICS):
        for c in CONDITIONS:
            values = np.array([[v[c, m, n, s] for s in range(10)] for n in range(4)])
            interval = [r for r in time_intervals if r['metric'] == m and r['condition'] == c]
            band = ax.fill_between(range(10), [r['ci95_low'] for r in interval],
                                   [r['ci95_high'] for r in interval], color=COLORS[c],
                                   alpha=.14, linewidth=0, zorder=.5)
            bands.append((ax, band))
            for row in values:
                ax.plot(range(10), row, color=COLORS[c], alpha=.28, lw=.7, zorder=1)
            ax.plot(range(10), values.mean(axis=0), color=COLORS[c], marker=MARKERS[c],
                    ms=4, mfc='white', mew=1.1, lw=1.5, zorder=3)
        ax.set(xlabel='Session', xlim=(-.3, 9.3), xticks=[0, 2, 4, 6, 8])
        if m == 'rescale_factor':
            ax.axhline(1, color='#999999', ls='--', lw=.7, zorder=0)
            ax.set(ylabel=r'$F_{\mathrm{R}}$', ylim=(.8, 1.1), yticks=[.8, .9, 1., 1.1])
        else:
            ax.set(ylabel='Spatial information (bits/spike)', ylim=(1.2, 1.9),
                   yticks=[1.2, 1.4, 1.6, 1.8])
        for r in [r for r in time_intervals if r['metric'] == m]:
            assert ax.get_ylim()[0] < r['ci95_low'] < r['ci95_high'] < ax.get_ylim()[1]
    handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], mfc='white',
                      mew=1.1, lw=1.5, ms=4, label=NAMES[c]) for c in CONDITIONS]
    legend = axes[0].legend(handles=handles, loc='lower left',
                            bbox_to_anchor=(.02, .0), borderaxespad=0,
                            labelspacing=-.1, handlelength=.9, handletextpad=.3,
                            borderpad=0)
    for ax, m in zip([axes[1], axes[3]], METRICS):
        ax.axhline(0, color='#999999', ls='--', lw=.8, zorder=0)
        for j, c in enumerate(CONDITIONS[1:]):
            differences = np.array([v[c, m, n, 9] - v['Intact', m, n, 9] for n in range(4)])
            ax.scatter(j + np.array([-.25, -.12, .12, .25]), differences,
                       s=11, color='#8B8B8B', alpha=.9, zorder=2)
            e = next(e for e in effects if e['condition'] == c and e['metric'] == m)
            mean, lo, hi = [float(e[k]) for k in ['mean_paired_difference', 'ci95_low', 'ci95_high']]
            ax.errorbar(j, mean, yerr=[[mean-lo], [hi-mean]], color=COLORS[c],
                        marker=MARKERS[c], ms=5, mfc='white', mew=1.2,
                        capsize=3, elinewidth=1.3, lw=0, zorder=4)
        ax.set(xlim=(-.4, 2.4), xticks=[0, 1, 2], xticklabels=PATHWAYS)
        ax.tick_params(axis='x', labelsize=9.5, length=0, pad=6)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_title('Session 9\nBlockade − Control', fontsize=9.5, pad=8, linespacing=1.2)
        if m == 'rescale_factor':
            ax.set(ylim=(-.20, .06), yticks=[-.20, -.15, -.10, -.05, 0, .05],
                   ylabel=r'Change in $F_{\mathrm{R}}$')
        else:
            ax.set(ylim=(-.60, .15), yticks=[-.60, -.45, -.30, -.15, 0, .15],
                   ylabel='Change in information (bits/spike)')
    for letter, (x, _, _, _) in zip('ABCD', boxes):
        fig.text((x-7)/WIDTH_MM, 77/HEIGHT_MM, letter,
                 fontsize=12, fontweight='bold', va='center', ha='left')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Check axes labels/ticks/titles/legend against the complete export canvas.
    canvas = fig.bbox
    for ax in axes:
        bbox = ax.get_tightbbox(renderer)
        assert bbox.x0 >= 0 and bbox.y0 >= 0 and bbox.x1 <= canvas.x1 and bbox.y1 <= canvas.y1
    # No adjacent panels may collide, including axis labels and uncertainty titles.
    panel_bounds = [ax.get_tightbbox(renderer) for ax in axes]
    assert all(a.x1 < b.x0 for a, b in zip(panel_bounds, panel_bounds[1:]))
    # Legend must occupy lower-left data-free space; audit all original curves.
    leg_bbox = legend.get_window_extent(renderer)
    collisions = []
    for line in axes[0].lines:
        vertices = line.get_transform().transform(line.get_xydata())
        for p, q in zip(vertices, vertices[1:]):
            samples = p + np.linspace(0, 1, 101)[:, None] * (q-p)
            if np.any((samples[:, 0] >= leg_bbox.x0) & (samples[:, 0] <= leg_bbox.x1) &
                      (samples[:, 1] >= leg_bbox.y0) & (samples[:, 1] <= leg_bbox.y1)):
                collisions.append(True)
    assert not collisions, 'Legend overlaps a time-course observation.'
    band_collisions = [path.transformed(band.get_transform()).intersects_bbox(leg_bbox, filled=True)
                       for ax, band in bands if ax is axes[0] for path in band.get_paths()]
    assert not any(band_collisions), 'Legend overlaps a time-course confidence band.'
    for ax in [axes[1], axes[3]]:
        tick_boxes = [tick.get_window_extent(renderer) for tick in ax.get_xticklabels()]
        assert all(a.x1 < b.x0 for a, b in zip(tick_boxes, tick_boxes[1:])), 'Pathway labels overlap.'
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "main_figure_6"
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".pdf"), dpi=600)
    fig.savefig(stem.with_suffix(".svg"), dpi=600)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    return {'adjacent_panel_bbox_collisions': 0, 'legend_curve_collisions': 0,
            'legend_confidence_band_collisions': 0, 'time_course_bands_clipped': 0,
            'all_axes_labels_within_canvas': True}

def plot(data_dir=None, output_dir=None):
    data_dir = Path(data_dir) if data_dir is not None else PACKAGE / "Plot_data/main_figure_6"
    output_dir = Path(output_dir) if output_dir is not None else PACKAGE / "Reproduced/Main_figs"
    values, effects, paired = load_and_verify(data_dir)
    intervals = time_course_intervals(values)
    draw(values, effects, intervals, output_dir)
    return output_dir / "main_figure_6"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PACKAGE / "Plot_data/main_figure_6")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "Reproduced/Main_figs")
    args = parser.parse_args()
    print(plot(args.data_dir, args.output_dir))


if __name__ == "__main__":
    main()

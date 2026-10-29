"""Draw representative trajectories and classified movement segments."""
from types import SimpleNamespace
import json
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from te_common import ROOT,load_table,export_figure,cli
import numpy as np
PALETTE={'approach':'#6A3D9A','departure':'#159C8E','full':'#9A9A9A'}
FIG_WIDTH_MM=183
FIG_HEIGHT_MM=106.13
def configure_style():
    font_path = font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8.5, 'axes.labelsize': 9.5, 'axes.titlesize': 10.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'path.simplify': False, 'savefig.facecolor': 'white'})
    return font_path

def draw_panel(ax, trajectory, segments, config, start_frame, stop_frame, title, letter):
    width, height = (trajectory.width_m, trajectory.height_m)
    boundary = float(config['boundary_zone_m'])
    centre_fraction = float(config['centre_fraction'])
    mx, my = (width * (1 - centre_fraction) / 2, height * (1 - centre_fraction) / 2)
    ax.add_patch(Rectangle((0, 0), width, height, facecolor='lightcoral', alpha=0.3, edgecolor='none', zorder=0))
    ax.add_patch(Rectangle((boundary, boundary), width - 2 * boundary, height - 2 * boundary, facecolor='white', edgecolor='none', zorder=0.1))
    ax.add_patch(Rectangle((mx, my), width * centre_fraction, height * centre_fraction, facecolor='lightblue', alpha=0.35, edgecolor='none', zorder=0.2))
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor='red', lw=1.15, zorder=1))
    ax.add_patch(Rectangle((boundary, boundary), width - 2 * boundary, height - 2 * boundary, fill=False, edgecolor='gray', ls='--', lw=0.75, zorder=1))
    ax.add_patch(Rectangle((mx, my), width * centre_fraction, height * centre_fraction, fill=False, edgecolor='blue', lw=1.05, zorder=1))
    shown = trajectory.positions[start_frame:stop_frame]
    ax.plot(shown[:, 0], shown[:, 1], color=PALETTE['full'], alpha=0.4, lw=0.55, zorder=2)
    for segment in segments:
        points = trajectory.positions[segment.start:segment.stop]
        colour = PALETTE[segment.direction]
        ax.plot(points[:, 0], points[:, 1], color=colour, alpha=0.78, lw=1.35, zorder=3)
        ax.plot(*points[0], marker='o', ms=4.3, color=colour, mec='white', mew=0.3, zorder=4)
        ax.plot(*points[-1], marker='^', ms=4.8, color=colour, mec='white', mew=0.3, zorder=4)
    ax.set_xlim(-0.05, width + 0.05)
    ax.set_ylim(-0.05, height + 0.05)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks(np.arange(0, width + 0.001, 0.2))
    ax.set_yticks(np.arange(0, height + 0.001, 0.2))
    ax.set_xlabel('X (m)', labelpad=3)
    ax.set_title(title, fontweight='bold', pad=6)
    ax.grid(alpha=0.22, linewidth=0.45, zorder=0.5)
    ax.tick_params(length=3, width=0.65)
    ax.text(-0.115, 1.047, letter, transform=ax.transAxes, fontsize=12, fontweight='bold', ha='left', va='bottom')

def make_figure(examples, config):
    font_path = configure_style()
    fig = plt.figure(figsize=(FIG_WIDTH_MM / 25.4, FIG_HEIGHT_MM / 25.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9], left=0.085, right=0.975, bottom=0.225, top=0.88, wspace=0.24)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    for ax, example, title, letter in zip(axes, examples, ['Square', 'Compressed'], ['A', 'B']):
        trajectory, segments, first, last = example
        draw_panel(ax, trajectory, segments, config, first, last, title, letter)
    axes[0].set_ylabel('Y (m)', labelpad=4)
    legend_handles = [Rectangle((0, 0), 1, 1, facecolor='lightcoral', edgecolor='red', alpha=0.3, label='Boundary region'), Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='blue', alpha=0.35, label='Center region'), Rectangle((0, 0), 1, 1, fill=False, edgecolor='gray', ls='--', label='Intermediate region'), Line2D([], [], color=PALETTE['full'], lw=0.8, label='Full trajectory'), Line2D([], [], color=PALETTE['approach'], marker='o', ls='none', ms=4.3, label='Approach start'), Line2D([], [], color=PALETTE['approach'], marker='^', ls='none', ms=4.8, label='Approach end'), Line2D([], [], color=PALETTE['departure'], marker='o', ls='none', ms=4.3, label='Departure start'), Line2D([], [], color=PALETTE['departure'], marker='^', ls='none', ms=4.8, label='Departure end')]
    legend = fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.53, 0.035), ncol=4, frameon=False, handlelength=1.7, handletextpad=0.6, columnspacing=1.55, labelspacing=0.8)
    return fig

def draw(output_dir=None):
    positions=load_table('supp_figure_1','positions')
    segments=load_table('supp_figure_1','segments')
    config=json.loads((ROOT/'Plot_data/supp_figure_1/geometry.json').read_text(encoding='utf-8'))
    examples=[]
    for letter in ('A','B'):
        part=positions.loc[positions.panel.eq(letter)].sort_values('frame')
        assert part.frame.to_list()==list(range(len(part)))
        geometry=config['panels'][letter]
        trajectory=SimpleNamespace(width_m=geometry['width_m'],height_m=geometry['height_m'],positions=part[['x_m','y_m']].to_numpy())
        selected=[SimpleNamespace(start=int(row.start_frame),stop=int(row.stop_frame_exclusive),direction=row.direction) for row in segments.loc[segments.panel.eq(letter)].itertuples()]
        examples.append((trajectory,selected,0,len(part)))
    fig=make_figure(examples,config)
    return export_figure(fig,'supp_figure_1',output_dir)

if __name__=='__main__':cli(draw)

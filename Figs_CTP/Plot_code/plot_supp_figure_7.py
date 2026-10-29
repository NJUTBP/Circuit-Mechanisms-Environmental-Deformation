"""Whole-session and boundary-conditioned grid-cell rate maps."""
import argparse
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from archive_common import DATA,export
FIGURE_WIDTH_IN=7.2
FIGURE_HEIGHT_IN=FIGURE_WIDTH_IN*629/947
FONT_SIZE_PT=10
EXTENT_M=[0,.8,0,1]
def configure():
    # The user explicitly requires Times New Roman and the original jet map.
    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "font.size": FONT_SIZE_PT, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "savefig.facecolor": "white"})

def plot(output_dir=None):
    with np.load(DATA/"supp_figure_7"/"rate_maps.npz") as data:
        sessions=data["sessions"].tolist()
        maps=data["rate_maps"]
        zero_bins=data["zero_occupancy_masks"]
    SESSIONS=sessions
    configure()
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, FIGURE_HEIGHT_IN))
    top_margin, bottom_margin, row_gap = 0.25, 0.155, 0.20
    map_height = (FIGURE_HEIGHT_IN - top_margin - bottom_margin - 2 * row_gap) / 3
    map_width = 0.8 * map_height
    left, column_gap = 0.42, 0.06
    mask_cmap = ListedColormap([(0, 0, 0, 0), (0.9,0.9,0.9,1.0)])
    axes_records = []
    for row in range(3):
        bottom = FIGURE_HEIGHT_IN - top_margin - (row + 1) * map_height - row * row_gap
        for col, session in enumerate(sessions):
            index = SESSIONS.index(session)
            ax = fig.add_axes([(left + col * (map_width + column_gap)) / FIGURE_WIDTH_IN,
                               bottom / FIGURE_HEIGHT_IN,
                               map_width / FIGURE_WIDTH_IN, map_height / FIGURE_HEIGHT_IN])
            ax.imshow(maps[index, row], cmap="jet", interpolation="bicubic", extent=EXTENT_M, zorder=0)
            ax.imshow(zero_bins[index, row], cmap=mask_cmap, interpolation="nearest", extent=EXTENT_M, zorder=0.001)
            ax.axis("off")
            ax.text(0, 0, f"{maps[index, row].max():.2f} Hz", transform=ax.transAxes,
                    ha="left", va="top", fontsize=FONT_SIZE_PT, color="black")
            if row == 0:
                ax.text(0.5, 1.04, f"Session {session}", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=FONT_SIZE_PT, fontweight="bold")
            if col == 0:
                ax.text(-0.20, 0.5, ("Module 1", "Left", "Right")[row], transform=ax.transAxes,
                        rotation=90, rotation_mode="anchor", ha="center", va="center",
                        fontsize=FONT_SIZE_PT, fontweight="bold")
            axes_records.append({"row": row, "column": col, "actual_session": session,
                                 "peak_Hz": float(maps[index, row].max()),
                                 "map_shape": list(maps[index, row].shape),
                                 "position": list(ax.get_position().bounds)})
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for ax in fig.axes:
        for text in ax.texts:
            box = text.get_window_extent(renderer)
            if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
                outside.append(text.get_text())
    assert not outside, f"Text extends beyond final canvas: {outside}"
    return export(fig,7,output_dir)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    print(plot(parser.parse_args().output_dir))

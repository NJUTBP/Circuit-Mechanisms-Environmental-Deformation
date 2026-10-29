"""Representative-cell rate-map layouts."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from archive_common import DATA,export
def draw(number, cfg, maps, peaks, output_dir=None):
    width, height = cfg["size"]
    # Six columns, four cells. Real map aspect remains 0.8:1 or 1.2:1.
    left, right, gap = cfg["left"], .04, cfg["gap"]
    map_width = (width-left-right-5*gap)/6
    map_height = map_width/cfg["width"]
    top, bottom = .26, .17
    row_gap = (height-top-bottom-4*map_height)/3
    assert row_gap > .16  # enough for the peak-rate text below each map
    fig = plt.figure(figsize=(width,height), facecolor="white")
    font_size = 9.5
    for i, cell in enumerate(cfg["cells"]):
        y = height-top-(i+1)*map_height-i*row_gap
        for j, session in enumerate(SESSIONS):
            x = left+j*(map_width+gap)
            ax = fig.add_axes([x/width,y/height,map_width/width,map_height/height])
            # Preserve the rate-map color scale and display interpolation.
            # No interpolation of the underlying measurements, no new samples.
            ax.imshow(maps[i,j], cmap="jet", interpolation="bicubic", aspect="equal")
            ax.set_axis_off()
            ax.text(0, -.012, f"{peaks[i,j]:.2f} Hz", transform=ax.transAxes,
                    ha="left", va="top", fontsize=font_size)
            if i == 0:
                ax.text(.5,1.045,f"Session {session}",transform=ax.transAxes,
                        ha="center",va="bottom",fontsize=font_size,fontweight="bold")
            if j == 0:
                ax.text(-.22,.5,f"cell #{cell}",transform=ax.transAxes,
                        rotation=90,rotation_mode="anchor",ha="center",va="center",
                        fontsize=font_size,fontweight="bold")
    return export(fig,number,output_dir)

def plot_place(number,output_dir=None):
    configs={9:dict(width=.8,size=(7.2,6.73),left=.40,gap=.10),
             10:dict(width=1.2,size=(7.2,3.92),left=.35,gap=.32)}
    global SESSIONS
    with np.load(DATA/f"supp_figure_{number}"/"rate_maps.npz") as d:
        maps=d["maps"]; peaks=d["peaks"]; SESSIONS=d["sessions"].tolist()
        cfg={**configs[number],"cells":d["cells"].tolist()}
    plt.rcParams.update({"font.family":"Times New Roman","font.size":9.5,
                         "pdf.fonttype":42,"svg.fonttype":"none"})
    return draw(number,cfg,maps,peaks,output_dir)

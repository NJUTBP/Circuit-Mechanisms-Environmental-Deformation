"""Plot main Figure 5 from the packaged network and group tables."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from familiarization_style import *

DATA = Path(__file__).resolve().parents[1]/"Plot_data/main_figure_5"
ROWS = (
    ("grid", "rescale_factor", r"$F_{\mathrm{R}}$", 1.0, None),
    ("grid", "delta_phase_shift_x_cm", r"$\Delta S_{\mathrm{phase}}$ (cm)", 0.0, None),
    ("place", "delta_field_length_x_cm", r"$\Delta L_{\mathrm{PF}}$ (cm)", 0.0, "length"),
    ("place", "delta_place_phase_x_cm", r"$\Delta D_{\mathrm{PF}}$ (cm)", 0.0, "phase"),
)

def plot(output_dir=None):
    output=Path(output_dir) if output_dir else DATA.parents[1]/"Reproduced/Main_figs"
    network=pd.read_csv(DATA/"network_values.csv",float_precision="round_trip")
    group=pd.read_csv(DATA/"group_summary.csv",float_precision="round_trip")
    configure()
    fig,axes=manuscript_axes()
    for row,(population,metric,ylabel,reference,place_key) in enumerate(ROWS):
        for col,env in enumerate(ENVS):
            ax=axes[row,col]
            values=draw_main_panel(ax,network,group,env,population,metric,reference,place_key)
            ax.set_ylim(*limits(values,reference))
            ax.set_ylabel(ylabel)
            ax.tick_params(labelbottom=row in (1,3))
            if row in (0,2):
                letter=("A","B")[col] if row==0 else ("C","D")[col]
                ax.text(-.12,1.08,letter,transform=ax.transAxes,fontweight="bold",fontsize=11.2,ha="left",va="bottom")
                ax.set_title(("Compressed environment","Elongated environment")[col],fontweight="bold",pad=7)
            if row in (1,3): ax.set_xlabel("Session")
    handles,labels=axes[0,0].get_legend_handles_labels()
    axes[0,0].legend(handles,labels,loc="upper left",frameon=True,framealpha=.9,borderpad=.25,handlelength=1.5)
    export_figure(fig,output/"main_figure_5")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    plot(parser.parse_args().output_dir)

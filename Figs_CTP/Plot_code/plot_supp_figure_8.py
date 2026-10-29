"""Plot supplementary Figure 8 from the packaged firing-rate tables."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter
from familiarization_style import *

DATA=Path(__file__).resolve().parents[1]/"Plot_data/supp_figure_8"
ROWS=[
    ("grid","peak_rate_ratio","Peak firing rate (ratio)",1.,None),
    ("grid","mean_rate_ratio","Mean firing rate (ratio)",1.,None),
    ("place","delta_peak_rate_hz","Peak firing rate (Hz)",0.,"length"),
    ("place","delta_mean_rate_hz","Mean firing rate (Hz)",0.,"phase"),
]

def plot(output_dir=None):
    output=Path(output_dir) if output_dir else DATA.parents[1]/"Reproduced/Supp_figs"
    network=pd.read_csv(DATA/"network_values.csv",float_precision="round_trip")
    group=pd.read_csv(DATA/"group_summary.csv",float_precision="round_trip")
    configure()
    mpl.rcParams.update({"font.family":"serif","font.serif":["Times New Roman"],"svg.fonttype":"none","pdf.fonttype":42})
    fig,axes=manuscript_axes()
    for row,(pop,metric,label,reference,color_key) in enumerate(ROWS):
        for col,env in enumerate(ENVS):
            ax=axes[row,col]
            values=draw_main_panel(ax,network,group,env,pop,metric,reference,color_key)
            bounds=np.array(values+[reference],float)
            extent=bounds.max()-bounds.min()
            margin=max(.12*extent,.002 if pop=="grid" else .0001)
            ax.set_ylim(bounds.min()-margin,bounds.max()+margin)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6,steps=[1,2,2.5,5,10]))
            formatter=ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)
            if pop=="place" and metric=="delta_mean_rate_hz":
                ax.set_ylim(-.02,.02)
                ax.set_yticks([-.02,-.01,0.,.01,.02])
                ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.set_ylabel(label)
            ax.tick_params(labelbottom=row in (1,3))
            if row in (0,2):
                letter=("A","B")[col] if row==0 else ("C","D")[col]
                ax.text(-.12,1.08,letter,transform=ax.transAxes,fontweight="bold",fontsize=11.2,ha="left",va="bottom")
                ax.set_title(("Compressed environment","Elongated environment")[col],fontweight="bold",pad=7)
            if row in (1,3): ax.set_xlabel("Session")
    handles,labels=axes[0,0].get_legend_handles_labels()
    axes[0,0].legend(handles,labels,loc="upper left",frameon=True,framealpha=.9,borderpad=.25,handlelength=1.5)
    export_figure(fig,output/"supp_figure_8")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path)
    plot(parser.parse_args().output_dir)

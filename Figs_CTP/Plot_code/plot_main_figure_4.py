"""Draw Figure 4 from the bundled contrast and paired-state values."""
from te_panels import *

def draw(output_dir=None):
    tables=load_tables('main_figure_4')
    st.configure()
    fig=st.plt.figure(figsize=(7.20472440945,3.14960629921))
    grid=fig.add_gridspec(1,3,left=.090,right=.985,bottom=.215,top=.85,wspace=.53,width_ratios=(1.02,1.,1.))
    axes=[fig.add_subplot(grid[0,i]) for i in range(3)]
    contrast_panel(axes[0],tables,('compressed_80','elongated_120'))
    main_state_panel(axes[1],tables,'compressed_80','B')
    main_state_panel(axes[2],tables,'elongated_120','C')
    axes[1].legend(handles=movement_handles(),loc='lower left',bbox_to_anchor=(.02,.015),ncol=1,frameon=False,fontsize=8.,handlelength=.9,handletextpad=.45,borderaxespad=0.,columnspacing=1.)
    calibrate_state_markers(fig)
    return export_figure(fig,'main_figure_4',output_dir)

if __name__=='__main__':cli(draw)

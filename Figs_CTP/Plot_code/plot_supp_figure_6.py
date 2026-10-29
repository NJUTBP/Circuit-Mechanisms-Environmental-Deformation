"""Draw Supplementary Figure 6 from the bundled contrast and state values."""
from te_panels import *

def draw(output_dir=None):
    tables=load_tables('supp_figure_6')
    st.configure()
    fig=st.plt.figure(figsize=(7.20472440945,7.32283464567))
    grid=fig.add_gridspec(3,2,left=.095,right=.985,bottom=.075,top=.922,hspace=.68,wspace=.37,height_ratios=(1.05,1.,1.))
    axes=[fig.add_subplot(grid[1+i//2,i%2]) for i in range(4)]
    contrast=fig.add_subplot(grid[0,:])
    environments=('compressed_70','compressed_90','elongated_110','elongated_130')
    for ax,env,letter in zip(axes,environments,'BCDE'):supp_state_panel(ax,tables,env,letter)
    contrast_panel(contrast,tables,environments,supplement=True)
    fig.legend(handles=movement_handles(),loc='upper center',bbox_to_anchor=(.54,axes[0].get_position().y1+.070),ncol=2,frameon=False,fontsize=8.,handlelength=1.1,columnspacing=1.4,borderaxespad=0.)
    calibrate_state_markers(fig)
    return export_figure(fig,'supp_figure_6',output_dir)

if __name__=='__main__':cli(draw)

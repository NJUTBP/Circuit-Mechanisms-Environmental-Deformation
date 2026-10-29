"""Draw Supplementary Figure 5 from its six numerical tables."""
from te_common import *

def draw(output_dir=None):
    t={name:load_table('supp_figure_5',name) for name in ('cross_global_values','cross_global_statistics','f2_values','component_statistics','ofat_values','ofat_statistics')}
    st.configure()
    fig=st.plt.figure(figsize=(183/25.4,260/25.4))
    grid=fig.add_gridspec(5,2,left=.115,right=.985,top=.965,bottom=.065,wspace=.36,hspace=.75,height_ratios=[.80,1.,1.,1.10,1.10])
    top=[fig.add_subplot(grid[0,j]) for j in range(2)]
    middle=[fig.add_subplot(grid[i,j]) for i in (1,2) for j in range(2)]
    bottom=[fig.add_subplot(grid[i,j]) for i in (3,4) for j in range(2)]
    cross_panels(top,t,'AB')
    component_panels(middle,t,'CDEF')
    robustness_panels(bottom,t,'C_delta_T','GHIJ')
    shared_legend(fig,direction_handles(),middle[0].get_position().y1+.055,2)
    shared_legend(fig,environment_handles(),bottom[0].get_position().y1+.055,6,fontsize=7.8)
    for ax in top:ax.set_ylabel(r'$E$ (bits/sample)',fontsize=9.)
    for ax in middle:ax.set_ylabel(r'$\Delta E$ (bits/sample)',fontsize=9.)
    return export_figure(fig,'supp_figure_5',output_dir)

if __name__=='__main__':cli(draw)

"""Draw familiarization contrasts from the bundled displayed series."""
from te_common import *

def draw(output_dir=None):
    values=load_table('supp_figure_12','network_values')
    statistics=load_table('supp_figure_12','statistics')
    st.configure()
    fig,axes=st.plt.subplots(1,2,figsize=(183/25.4,78/25.4))
    fig.subplots_adjust(left=.105,right=.985,top=.86,bottom=.18,wspace=.35)
    low,high,peak,span=st.limits(values.C_delta_T*1000,statistics[['ci_low','ci_high']].to_numpy()*1000,top=.30)
    for index,(ax,env) in enumerate(zip(axes,('compressed_80','elongated_120'))):
        group=values.loc[values.environment.eq(env)]
        for _,g in group.groupby('display_series'):
            assert not g.session.duplicated().any()
            ax.plot(range(10),g.set_index('session').C_delta_T.reindex(range(10)).to_numpy()*1000,color=st.GRAY,lw=.65,marker='o',ms=2.3,alpha=.60)
        summary=statistics.loc[statistics.environment.eq(env)].set_index('session').reindex(range(10))
        st.interval_series(ax,range(10),summary,'C_delta_T',st.ENV_COLORS[env],shading=True)
        for session,p in enumerate(summary.p_display):
            if np.isfinite(p) and p<.05:ax.text(session,peak+.12*span,'*',ha='center',fontsize=9.)
        st.zero(ax)
        ax.axvspan(-.3,.3,color=st.ENV_COLORS[env],alpha=.08)
        ax.set_ylim(low,high);ax.set_xlim(-.35,9.35)
        ax.set_xticks(range(10));ax.set_xlabel('Session')
        st.ylabel(ax,'C_delta_T');ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_title(env.rsplit('_',1)[1]+' cm',pad=8.)
        st.panel_label(ax,chr(65+index))
    return export_figure(fig,'supp_figure_12',output_dir)

if __name__=='__main__':cli(draw)

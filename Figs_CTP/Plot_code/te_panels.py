"""Contrast and paired-state panels."""
from te_common import *

def networks(tables):
    return tuple(sorted(tables['f2_values'].network_id.unique()))

def p_text(p):
    return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns'

def p_fontsize(p):
    return 10. if p<.05 else 9.

def load_tables(figure):
    values=load_table(figure,'network_values')
    statistics=load_table(figure,'statistics')
    return {'f1_values':values.loc[values.source_panel_family.eq('F1')],
            'f2_values':values.loc[values.source_panel_family.eq('F2')],
            'f1_statistics':statistics.loc[statistics.statistic_kind.eq('combined_state_axis_contrast')],
            'f2_state_descriptive_statistics':statistics.loc[statistics.statistic_kind.eq('state_descriptive')],
            'f2_paired_statistics':statistics.loc[statistics.statistic_kind.eq('paired_departure_minus_approach')]}

def contrast_panel(ax,tables,environments,letter='A',supplement=False):
    values=tables['f1_values']
    summary=tables['f1_statistics'].set_index('environment').reindex(environments)
    complete=np.r_[values.C_delta_T.to_numpy()*1000,summary.ci_low.to_numpy()*1000,summary.ci_high.to_numpy()*1000,0]
    span=float(np.ptp(complete))
    low=float(complete.min()-.07*span)
    high=float(complete.max()+.20*span)
    offsets=np.linspace(-.055,.055,len(networks(tables)))
    for location,env in enumerate(environments):
        row=summary.loc[env]
        observed=values.loc[values.environment.eq(env)].set_index('network_id').C_delta_T.reindex(networks(tables)).to_numpy()*1000
        mean,lower,upper=np.asarray([row['mean'],row.ci_low,row.ci_high])*1000
        ax.scatter(location+offsets,observed,s=15 if supplement else 14,facecolors=st.GRAY,edgecolors='white',linewidths=.4 if supplement else .35,alpha=.78,zorder=2)
        marker='D' if supplement or location else 'o'
        ax.errorbar(location,mean,yerr=[[mean-lower],[upper-mean]],fmt=marker,linestyle='none',ms=5.6 if supplement else 5.8,color=st.ENV_COLORS[env],mec='white',mew=.6,elinewidth=1.2,capsize=3.,zorder=4)
        p=row.p_display
        y=complete.max()+(.065 if supplement else .075)*span
        ax.text(location,y,p_text(p),ha='center',va='bottom',fontsize=p_fontsize(p),fontweight='bold' if p<.05 or not supplement else 'normal')
    st.zero(ax)
    ax.set_xlim(-.48,len(environments)-.52)
    ax.set_ylim(low,high)
    ax.set_xticks(range(len(environments)),[e.rsplit('_',1)[1]+' cm' for e in environments])
    st.ylabel(ax,'C_delta_T')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title('Overall contrast',pad=8.,fontsize=9.4,fontweight='bold')
    st.panel_label(ax,letter,x=-.072 if supplement else -.20,y=1.08)

def calibrate_state_markers(fig):
    for ax in fig.axes:
        if ax.get_title()=='Overall contrast':continue
        for line in ax.lines:
            if line.get_marker() in ('o','^') and line.get_color() in set(st.STATE_COLORS.values()):
                line.set_markersize(4.195793667335314)
    for legend in list(fig.legends)+[ax.get_legend() for ax in fig.axes if ax.get_legend()]:
        for handle in legend.legend_handles:handle.set_markersize(4.195793667335314)

def main_state_panel(ax, tables, env, letter):
    values = tables['f2_values']
    summaries = tables['f2_state_descriptive_statistics']
    low, high, peak, span = st.limits(values.delta_T.to_numpy() * 1000, summaries.ci_low.to_numpy() * 1000, summaries.ci_high.to_numpy() * 1000, top=0.28, bottom=0.07)
    group = values.loc[values.environment.eq(env)]
    for axis, base in [('x', 0.0), ('y', 3.0)]:
        paired = group.loc[group.wall_axis.eq(axis)].pivot(index='network_id', columns='state', values='delta_T').reindex(networks(tables))
        for network in networks(tables):
            observed_pair = paired.loc[network, ['approach', 'departure']].to_numpy(float) * 1000
            ax.plot([base, base + 1], observed_pair, color=st.GRAY, alpha=0.66, lw=0.7, marker='o', ms=2.7, zorder=1, gid=f'network_pair_{env}_{axis}_{network}')
        for state, location in [('approach', base), ('departure', base + 1)]:
            summary = summaries.loc[summaries.environment.eq(env) & summaries.wall_axis.eq(axis) & summaries.state.eq(state)]
            st.interval_series(ax, [location], summary, 'delta_T', st.STATE_COLORS[state], marker='o' if state == 'approach' else '^')
        p = tables['f2_paired_statistics'].loc[lambda x: x.environment.eq(env) & x.wall_axis.eq(axis), 'p_display'].iloc[0]
        bracket = peak + 0.055 * span
        ax.plot([base, base, base + 1, base + 1], [bracket, bracket + 0.025 * span, bracket + 0.025 * span, bracket], color='#555B60', lw=0.65)
        ax.text(base + 0.5, bracket + 0.045 * span, p_text(p), ha='center', va='bottom', fontsize=p_fontsize(p), fontweight='bold' if p < 0.05 else 'normal')
    st.zero(ax)
    ax.set_xlim(-0.65, 4.65)
    ax.set_ylim(low, high)
    ax.set_xticks([0.5, 3.5], ['Deformed\naxis', 'Undeformed\naxis'])
    st.ylabel(ax, 'delta_T')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_title(env.rsplit('_',1)[1]+' cm', fontweight='bold', pad=8.0, fontsize=9.4)
    st.panel_label(ax, letter, x=-0.19, y=1.08)
def supp_state_panel(ax, tables, env: str, letter: str):
    values = tables['f2_values']
    summaries = tables['f2_state_descriptive_statistics']
    low, high, peak, span = st.limits(values.delta_T.to_numpy() * 1000, summaries.ci_low.to_numpy() * 1000, summaries.ci_high.to_numpy() * 1000, top=0.28, bottom=0.07)
    group = values.loc[values.environment.eq(env)]
    offsets = np.linspace(-0.085, 0.085, 8)
    for axis, base in [('x', 0.0), ('y', 3.0)]:
        paired = group.loc[group.wall_axis.eq(axis)].pivot(index='network_id', columns='state', values='delta_T').reindex(networks(tables))
        for network, offset in zip(networks(tables), offsets):
            observed_pair = paired.loc[network, ['approach', 'departure']].to_numpy(float) * 1000
            ax.plot([base + offset, base + 1 + offset], observed_pair, color=st.GRAY, alpha=0.58, lw=0.55, zorder=1, gid=f'network_pair_{env}_{axis}_{network}')
        for state, location in [('approach', base), ('departure', base + 1)]:
            part = group.loc[group.wall_axis.eq(axis) & group.state.eq(state)]
            observed = part.set_index('network_id').delta_T.reindex(networks(tables)).to_numpy() * 1000
            row = summaries.loc[summaries.environment.eq(env) & summaries.wall_axis.eq(axis) & summaries.state.eq(state)].iloc[0]
            mean, lower, upper = np.asarray([row['mean'], row.ci_low, row.ci_high]) * 1000
            ax.scatter(location + offsets, observed, s=12, facecolors=st.GRAY, edgecolors='white', linewidths=0.35, alpha=0.72, zorder=2)
            ax.errorbar(location, mean, yerr=[[mean - lower], [upper - mean]], fmt='o' if state == 'approach' else '^', linestyle='none', ms=4.7, color=st.STATE_COLORS[state], mec='white', mew=0.5, elinewidth=1.0, capsize=2.5, zorder=4)
        p = tables['f2_paired_statistics'].loc[lambda x: x.environment.eq(env) & x.wall_axis.eq(axis), 'p_display'].iloc[0]
        bracket = peak + 0.055 * span
        ax.plot([base, base, base + 1, base + 1], [bracket, bracket + 0.025 * span, bracket + 0.025 * span, bracket], color='#555B60', lw=0.65)
        ax.text(base + 0.5, bracket + 0.045 * span, p_text(p), ha='center', va='bottom', fontsize=p_fontsize(p), fontweight='bold' if p < 0.05 else 'normal')
    st.zero(ax)
    ax.set_xlim(-0.65, 4.65)
    ax.set_ylim(low, high)
    ax.set_xticks([0.5, 3.5], ['Deformed\naxis', 'Undeformed\naxis'])
    st.ylabel(ax, 'delta_T')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_title(env.rsplit('_', 1)[1] + ' cm', fontweight='bold', pad=8.0, fontsize=9.4)
    st.panel_label(ax, letter, x=-0.16, y=1.08)
    return {'panel': letter, 'metric': 'delta_T', 'environment': env, 'network_point_count': 32, 'network_connections': True, 'matched_network_connections': 16, 'connection_definition': 'same network Approach to Departure within each axis only', 'state_mean_connections': False, 'tests': 'fixed one-sided x Departure > Approach; y Departure < Approach; unadjusted'}

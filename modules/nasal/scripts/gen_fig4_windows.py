from plot_common import *
t,records=growth();windows=[6,t[-1]]
fig,axes=plt.subplots(1,2,figsize=(11,4.3),gridspec_kw={'width_ratios':[1.35,1]},layout='constrained')
ax=axes[0];wt=np.array([auc(t,records['WT'][0]['y'],end) for end in windows])
for group in ORDER[1:]:
    values=np.array([[auc(t,r['y'],end) for end in windows] for r in records[group]])
    delta=(values.mean(0)/wt-1)*100
    ax.plot(windows,delta,marker='o',ms=6,color=COLORS[group],label=group)
    for index,(x,y) in enumerate(zip(windows,delta)):
        offset=(0,8 if group!='Lp-NAC' else -16)
        if index==0 and group=='Lp-NAC':offset=(12,15)
        if index==0 and group=='Lp-Q':offset=(-12,-17)
        ax.annotate(f'{y:+.2f}%',(x,y),xytext=offset,textcoords='offset points',ha='center',fontsize=9,color=COLORS[group])
ax.axhline(0,color=COLORS['WT'],ls='--',lw=1)
ax.set(xlabel='Integration window (h)',ylabel='OD–time integral relative to WT (%)',xticks=windows,xticklabels=['0–6',f'0–{t[-1]:.1f}'],xlim=(3,26),ylim=(-28,16))
ax.legend(loc='lower right',frameon=False,fontsize=9);panel(ax,'A');polish(ax)
ax=axes[1]
for j,group in enumerate(ORDER):
    vals=np.array([r['y'][-1] for r in records[group]])
    ax.scatter(j+np.linspace(-.075,.075,len(vals)) if len(vals)>1 else [j],vals,s=50,facecolor='white',edgecolor=COLORS[group],lw=1.5,zorder=4)
    ax.text(j,vals.max()+.07,f'{vals.mean():.4f}',ha='center',fontsize=10,color=COLORS[group])
ax.set(xticks=range(4),xticklabels=ORDER,ylabel=r'Endpoint OD$_{600}$',ylim=(1.3,1.92),xlim=(-.6,3.6))
ax.text(.03,.96,f'{t[-1]:.4f} h',transform=ax.transAxes,va='top',fontsize=10)
ax.text(.03,.07,'Open dots: supplied curve for each strain',transform=ax.transAxes,fontsize=8.5)
ax.tick_params(axis='x',labelsize=9);panel(ax,'B');polish(ax)
save(fig,'fig04_growth_windows')

from plot_common import *
t,records=growth()
fig,axes=plt.subplots(1,3,figsize=(12.6,4.1),layout='constrained')
for j,angle in enumerate([10,20]):
    d,m=transport(angle);time=d['time_after_handoff_s'];base=d['total_cfu'];ax=axes[j]
    ax.plot(time/60,base/1000,color='#96A3AE',ls='--',lw=2,label='Zero growth',zorder=4)
    for group in ORDER:
        factor=np.mean([curve(time/3600,r['b0'],r['p'])/r['b0'] for r in records[group]],axis=0)
        ax.plot(time/60,base*factor/1000,color=COLORS[group],lw=1.8,ls=':' if group=='WT' else '-',label=group,zorder=5 if group=='WT' else 3)
    ax.set(xlabel='Time after deposition (min)',ylabel=r'Local retained amount ($10^3$ CFU)',xlim=(0,60),xticks=[0,20,40,60],ylim=(0,36 if angle==10 else 9.6))
    ax.text(.04,.93,f'{angle}° full cone',transform=ax.transAxes,color=ANGLES[angle],fontweight='bold')
axes[0].legend(frameon=False,fontsize=8,loc='lower left',ncol=2)
d,m=transport(10);time=d['time_after_handoff_s'];base=d['total_cfu'];baseauc=trapezoid(base,time/3600)
for j,group in enumerate(ORDER):
    gains=[]
    for r in records[group]:
        factor=curve(time/3600,r['b0'],r['p'])/r['b0']
        gains.append((trapezoid(base*factor,time/3600)/baseauc-1)*100)
    axes[2].bar(j,np.mean(gains),width=.55,color=COLORS[group],alpha=.85,zorder=3)
    axes[2].scatter([j]*len(gains),gains,s=24,facecolor='white',edgecolor=COLORS[group],zorder=4)
    axes[2].text(j,max(gains)+.18,f'{np.mean(gains):.2f}%',ha='center',fontsize=9)
axes[2].set(xticks=range(4),xticklabels=ORDER,ylabel='0–60 min CFU–time integral gain (%)',ylim=(0,40));axes[2].tick_params(axis='x',labelsize=8.5)
axes[2].text(.05,.93,'10° · relative to zero growth',transform=axes[2].transAxes,fontsize=8.5)
for ax,l in zip(axes,'ABC'):panel(ax,l);polish(ax)
save(fig,'fig06_growth_coupling')

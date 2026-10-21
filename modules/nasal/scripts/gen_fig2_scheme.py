from plot_common import *
fig,ax=plt.subplots(figsize=(5.8,3.9),layout='constrained')
t,records=growth();dense=np.linspace(0,t[-1],400)
for g in ORDER:
    b=np.mean([curve(dense,r['b0'],r['p'])/r['b0'] for r in records[g]],axis=0)
    ax.scatter(t,records[g][0]['relative_y'],s=12,facecolor='white',edgecolor=COLORS[g],linewidth=.6,zorder=3)
    ax.plot(dense,b,color=COLORS[g],label=g,lw=1.8)
ax.set(xlabel='Culture time (h)',ylabel=r'Relative growth OD$_t$/OD$_0$',xlim=(0,24),ylim=(0,50),xticks=[0,6,12,18,24])
ax.text(.03,.95,'Dots: supplied ratios · Lines: normalized fits',transform=ax.transAxes,va='top',fontsize=7.5)
ax.legend(loc='lower right',frameon=False,fontsize=8.5)
polish(ax)
save(fig,'fig02_transport_scheme')

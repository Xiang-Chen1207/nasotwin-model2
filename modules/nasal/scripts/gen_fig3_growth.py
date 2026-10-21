from plot_common import *
t,records=growth(); dense=np.linspace(0,t[-1],500)
fig,axes=plt.subplots(2,2,figsize=(10.8,7.1),sharex=True,sharey=True,layout='constrained')
for ax,group,letter in zip(axes.flat,ORDER,'ABCD'):
    for j,r in enumerate(records[group]):
        fitted=curve(t,r['b0'],r['p'])
        r2=1-np.sum((r['y']-fitted)**2)/np.sum((r['y']-r['y'].mean())**2)
        ax.scatter(t,r['y'],s=20,marker='o',facecolor='white',edgecolor=COLORS[group],linewidth=.8,alpha=.85,zorder=3,label=f'{r["name"]} · measured')
        ax.plot(dense,curve(dense,r['b0'],r['p']),color=COLORS[group],ls='-',lw=2,label=f'{r["name"]} · fitted ($R^2={r2:.4f}$)')
    ax.text(.03,.95,group,transform=ax.transAxes,va='top',fontweight='bold',fontsize=13,color=COLORS[group])
    ax.set(xlim=(-.4,24.4),ylim=(0,1.95),xticks=[0,6,12,18,24],yticks=[0,.5,1,1.5])
    ax.legend(loc='lower right',fontsize=8,frameon=False,ncol=1)
    panel(ax,letter);polish(ax)
for ax in axes[-1]:ax.set_xlabel('Culture time (h)')
for ax in axes[:,0]:ax.set_ylabel(r'OD$_{600}$')
save(fig,'fig03_growth_fits')

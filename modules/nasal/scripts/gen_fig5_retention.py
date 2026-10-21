from plot_common import *
fig,axes=plt.subplots(1,3,figsize=(12.6,4.0),layout='constrained')
for j,angle in enumerate([10,20]):
    d,m=transport(angle);n0=d['total_cfu'][0];col=ANGLES[angle]
    f=m['deposited_volume_fraction_of_emitted']*100
    axes[0].bar(j,f,width=.52,color=col,zorder=3)
    axes[0].text(j,f+.11,f'{f:.3f}%',ha='center',fontweight='bold',color=col)
    axes[1].plot(d['time_after_handoff_s']/60,d['total_cfu']/n0*100,color=col,ls=['-','--'][j],label=f'{angle}° cone')
    bottom=0
    for key,color,label in [('attached_cfu','#7962A9','Attached'),('mobile_cfu','#5BACC0','Mobile'),('cleared_cfu','#DCE3E8','Cleared')]:
        val=d[key][-1]/n0*100
        axes[2].bar(j,val,bottom=bottom,width=.52,color=color,label=label if j==0 else None,zorder=3)
        if val>8:axes[2].text(j,bottom+val/2,f'{val:.1f}%',ha='center',va='center',fontsize=10,color='white' if key=='attached_cfu' else '#25394A')
        bottom+=val
axes[0].set(xticks=[0,1],xticklabels=['10°','20°'],ylabel='Target deposition / emitted liquid (%)',ylim=(0,4.15))
axes[1].set(xlabel='Time after deposition (min)',ylabel='Retained / initial local dose (%)',xlim=(0,60),ylim=(0,105),xticks=[0,20,40,60]);axes[1].legend(frameon=False,fontsize=9)
axes[1].text(.05,.08,'Zero-growth baseline',transform=axes[1].transAxes,fontsize=9)
axes[2].set(xticks=[0,1],xticklabels=['10°','20°'],ylabel='Fraction of initial local dose (%)',ylim=(0,105))
axes[2].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.11),ncol=3,fontsize=8)
axes[2].text(.06,.92,'60 min',transform=axes[2].transAxes,fontsize=9)
for ax,l in zip(axes,'ABC'):panel(ax,l);polish(ax)
save(fig,'fig05_deposition_retention')

from pathlib import Path
import os
OUT = Path(__file__).resolve().parents[1]
ROOT = OUT
BUILD = OUT / '.build'
for key, folder in [('MPLCONFIGDIR','mpl'),('TMPDIR','cache'),('XDG_CACHE_HOME','cache')]:
    (BUILD / folder).mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(BUILD / folder)
(OUT / 'figures').mkdir(exist_ok=True)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.integrate import trapezoid
import openpyxl, json, csv
COLORS = {'WT':'#526575','Lp-NAC':'#007F86','Lp-SFN':'#CD762B','Lp-Q':'#7962A9'}
ANGLES = {10:'#126D99',20:'#D27B32'}
ORDER = ['WT','Lp-NAC','Lp-SFN','Lp-Q']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#B2BDC7',
 'axes.labelcolor':'#25394A','text.color':'#25394A','xtick.color':'#526575',
 'ytick.color':'#526575','axes.linewidth':.8,'lines.linewidth':2,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def save(fig,name):
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
    print('Saved',name,flush=True)
def panel(ax,label):
    ax.text(-.13,1.07,label,transform=ax.transAxes,fontweight='bold',fontsize=15)
def polish(ax):
    ax.grid(axis='y',color='#E4E9ED',lw=.65,zorder=0)
    ax.set_axisbelow(True)
def curve(t,b0,p):
    """Evaluate the smooth-startup Logistic curve at culture times in hours."""
    r,k,lag=p
    t=np.asarray(t,dtype=float)
    a=t if lag <= 0 else t+lag*np.expm1(-t/lag)
    return k/(1+(k/b0-1)*np.exp(-r*a))
def growth(workbook=None):
    """Fit one 47-point Sheet1 trajectory for each of the four strains."""
    workbook=workbook or os.environ.get('NASOTWIN_GROWTH_WORKBOOK',ROOT/'data/growth_curve.xlsx')
    wb=openpyxl.load_workbook(workbook,read_only=True,data_only=True)
    rows=list(wb['Sheet1'].iter_rows(min_row=1,max_row=48,min_col=1,max_col=11,values_only=True))
    headers=rows[0]; data=np.array(rows[1:],dtype=float); t=data[:,0]
    records={}
    for group in ORDER:
        column=headers.index(group)
        y=data[:,column]; b0=y[0]
        p=least_squares(lambda p:curve(t,b0,p)-y,[.5,max(y),1],bounds=([1e-9,b0+1e-9,1e-8],[np.inf]*3),xtol=1e-12,ftol=1e-12,gtol=1e-12).x
        records[group]=[{'name':group,'y':y,'relative_y':data[:,column+6],'b0':b0,'p':p}]
    wb.close()
    return t,records
def auc(t,y,end):
    """Integrate observed OD by trapezoids with an interpolated window endpoint."""
    mask=t<end
    return trapezoid(np.r_[y[mask],np.interp(end,t,y)],np.r_[t[mask],end])
def culture_metrics(t,records):
    """Compute fitted agreement and observed endpoint and OD-AUC metrics."""
    wt=records['WT'][0]['y']
    wt_full=auc(t,wt,t[-1]); wt_6h=auc(t,wt,6.)
    rows=[]
    for group in ORDER:
        record=records[group][0]
        y=record['y']; residual=curve(t,record['b0'],record['p'])-y
        full=auc(t,y,t[-1]); first6=auc(t,y,6.)
        rows.append(dict(
            strain=group,record=record['name'],observations=len(t),
            observation_start_h=float(t[0]),observation_end_h=float(t[-1]),
            r_squared=float(1-np.sum(residual**2)/np.sum((y-y.mean())**2)),
            rmse_od=float(np.sqrt(np.mean(residual**2))),
            endpoint_od=float(y[-1]),endpoint_relative_od=float(record['relative_y'][-1]),
            od_auc_full_h=float(full),od_auc_0_6h=float(first6),
            od_auc_full_vs_wt_pct=float(100*(full/wt_full-1)),
            od_auc_0_6h_vs_wt_pct=float(100*(first6/wt_6h-1))))
    return rows
def transport(angle):
    path=ROOT/f'data/reference_retention/cone{angle}'
    meta=json.loads((path/'model_inputs.json').read_text())
    rows=[r for r in csv.DictReader((path/'retention.csv').open()) if r['strain']=='Gln']
    data={k:np.array([float(r[k]) for r in rows]) for k in rows[0] if k!='strain'}
    return data,meta

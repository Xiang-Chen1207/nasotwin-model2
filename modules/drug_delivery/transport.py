"""Track drug amounts through parallel direct and systemic transport routes.

Units are h, nmol, and mL. External mode uses a concentration boundary; finite
mode uses source rates and initial amounts supplied by the caller.
"""
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


class EvidenceGap(ValueError):
    """Signal a required transport input."""

    pass


NAMES=['nose_S','nose_N','nose_SN','transit_S','transit_N','plasma_S','plasma_N',
       'brain_S','brain_N','exits_S','exits_N','inputs_S','inputs_N']

@dataclass(frozen=True)
class TransportParameters:
    V_nose:float;V_plasma:float;V_brain:float
    PS_ep:list;f_direct:list;f_systemic:list;tau_direct:list
    PS_pb:list;K_pb:list;CL_plasma:list;CL_brain:list
    k_nasal_loss:list;kf:float;kr:float;k_conjugate_loss:float
    def validate(self):
        vals=[self.V_nose,self.V_plasma,self.V_brain,self.kf,self.kr,self.k_conjugate_loss]
        for field in ['PS_ep','f_direct','f_systemic','tau_direct','PS_pb','K_pb','CL_plasma','CL_brain','k_nasal_loss']:
            x=getattr(self,field)
            if x is None or len(x)!=2:raise EvidenceGap(field+' needs separate SFN/NAC values')
            vals+=list(x)
        if any(x is None for x in vals):raise EvidenceGap('Unconstrained transport coefficients')
        if not np.isfinite(vals).all() or min(vals)<0:raise ValueError('Finite nonnegative coefficients required')
        if min(self.V_nose,self.V_plasma,self.V_brain,*self.tau_direct,*self.K_pb)<=0:raise ValueError('Positive volumes, delay and partition required')
        if any(a+b>1 for a,b in zip(self.f_direct,self.f_systemic)):raise ValueError('Absorption fractions exceed unity')

def steady_state_gain(parameters):
    """Return SFN/NAC source-to-target gains in h/mL (Wiki Equation 27).

    The linear steady state uses constant production and zero conjugation.
    Direct transit affects arrival time; route fractions enter the gain.
    """
    parameters.validate()
    if any((parameters.kf, parameters.kr, parameters.k_conjugate_loss)):
        raise EvidenceGap('Steady-state gain requires zero conjugation rates')
    epithelial = np.asarray(parameters.PS_ep, dtype=float)
    nasal_removal = epithelial + np.asarray(parameters.k_nasal_loss) * parameters.V_nose
    exchange = np.asarray(parameters.PS_pb, dtype=float)
    plasma_clearance = np.asarray(parameters.CL_plasma, dtype=float)
    target_clearance = np.asarray(parameters.CL_brain, dtype=float)
    denominator = ((exchange + plasma_clearance) * target_clearance
                   + exchange * plasma_clearance / np.asarray(parameters.K_pb))
    if np.any(nasal_removal <= 0) or np.any(denominator <= 0):
        raise EvidenceGap('Steady-state gain requires stable nasal and plasma-target pools')
    epithelial_fraction = epithelial / nasal_removal
    numerator = (exchange * np.asarray(parameters.f_systemic)
                 + (exchange + plasma_clearance) * np.asarray(parameters.f_direct))
    return epithelial_fraction * numerator / denominator


def simulate(boundary,p,*,mode,start,end,source=None,initial_nasal=None,method='DOP853',rtol=1e-9,atol=1e-11,max_step=.025,points=301):
    p.validate()
    if mode not in ['external_concentration','finite_supply']:raise ValueError('Choose one supply contract')
    if end<=start:raise ValueError('Invalid time span')
    y0=np.zeros(13)
    if mode=='external_concentration':
        if source is not None or initial_nasal is not None:raise ValueError('Fixed boundary cannot also debit a finite donor')
        if boundary is None:raise EvidenceGap('No concentration boundary')
        if p.kf or p.kr or p.k_conjugate_loss or any(p.k_nasal_loss):raise EvidenceGap('Measured free-parent boundary already contains donor chemistry; do not apply it again')
        boundary(start);boundary(end)
    else:
        if boundary is not None:raise ValueError('Finite mode cannot fix donor concentration')
        if source is None or initial_nasal is None:raise EvidenceGap('Finite supply requires known replenishment and initial amounts')
        y0[:3]=initial_nasal
        if not np.isfinite(y0).all() or y0.min()<0:raise ValueError('Invalid initial amounts')
    def rhs(t,y):
        d=np.zeros(13)
        if mode=='external_concentration':
            c=boundary(t);u=np.asarray(p.PS_ep)*c
            # Accumulate the amount crossing the concentration boundary.
            d[11:13]=u
        else:
            c=y[:2]/p.V_nose;u=np.asarray(p.PS_ep)*c
            supply=np.asarray(source(t),dtype=float)
            if supply.shape!=(2,) or not np.isfinite(supply).all() or supply.min()<0:raise ValueError('Known source must be nmol/h and nonnegative')
            r=p.kf*y[0]*y[1]/p.V_nose-p.kr*y[2]
            loss=np.asarray(p.k_nasal_loss)*y[:2];lc=p.k_conjugate_loss*y[2]
            d[:2]=supply-u-loss-r;d[2]=r-lc;d[9:11]+=loss+lc;d[11:13]=supply
        for j in range(2):
            tr=y[3+j]/p.tau_direct[j]
            cp=y[5+j]/p.V_plasma;cb=y[7+j]/p.V_brain
            flow=p.PS_pb[j]*(cp-cb/p.K_pb[j])
            ploss=p.CL_plasma[j]*cp;bloss=p.CL_brain[j]*cb
            d[3+j]+=p.f_direct[j]*u[j]-tr
            d[5+j]+=p.f_systemic[j]*u[j]-flow-ploss
            d[7+j]+=tr+flow-bloss
            d[9+j]+=(1-p.f_direct[j]-p.f_systemic[j])*u[j]+ploss+bloss
        return d
    # Integrate separately between input knots.
    knots=boundary.t if boundary is not None else np.array([start,end])
    knots=np.unique(np.r_[start,knots[(knots>start)&(knots<end)],end])
    all_t=[];all_y=[];state=y0.copy();grid=np.linspace(start,end,points)
    for a,b in zip(knots[:-1],knots[1:]):
        sol=solve_ivp(rhs,(a,b),state,method=method,rtol=rtol,atol=atol,max_step=max_step,dense_output=True)
        if not sol.success:raise RuntimeError(sol.message)
        take=grid[(grid>=a)&(grid<b if b<end else grid<=b)]
        all_t.extend(take);all_y.extend(sol.sol(take).T);state=sol.y[:,-1]
    y=np.asarray(all_y).T
    if y.min()<-1e-8:raise RuntimeError('Negative amount: no clipping used')
    ledger=[]
    for j in range(2):
        total=y[3+j]+y[5+j]+y[7+j]+y[9+j]-y[11+j]
        initial=0
        if mode=='finite_supply':total+=y[j]+y[2];initial=y0[j]+y0[2]
        ledger.append(total-initial)
    return dict(time_h=np.asarray(all_t),amounts=y,names=NAMES,ledger=np.asarray(ledger),
                boundary_mode=mode,brain_free_uM=y[7:9]/p.V_brain,
                interpretation='effective brain free-parent pool; physiology not inferred')

def observation(result,variable):
    if variable=='brain_free_parent':return result['brain_free_uM']
    if variable in ['CSF','brain_homogenate','total14C','intracellular']:
        raise EvidenceGap('No calibrated '+variable+' observation bridge; not interchangeable with free brain pool')
    raise ValueError(variable)

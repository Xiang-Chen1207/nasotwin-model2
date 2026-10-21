"""Compute a two-stage NC04 insertion-depth and spray-angle deposition grid."""
from __future__ import annotations

import os as _runtime_os
from pathlib import Path as _RuntimePath
_runtime_root = _RuntimePath(__file__).resolve().parents[2] / ".build" / "runtime"
for _key, _folder in {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
                      "MPLCONFIGDIR": "mpl", "XDG_CACHE_HOME": "cache",
                      "PYVISTA_USERDATA_PATH": "pyvista"}.items():
    _cache = _runtime_root / _folder
    _cache.mkdir(parents=True, exist_ok=True)
    _runtime_os.environ.setdefault(_key, str(_cache))

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import time

import numpy as np

import nc04_angles as solver
from nc04_optimization import run_candidate, write_csv


def case_for(depth, pitch, cfg):
    return dict(case=f'd{depth:+03d}_p{pitch:05.1f}'.replace('.', 'p'),
                depth_mm=int(depth), pitch_deg=float(pitch),
                yaw_deg=cfg['yaw_deg'],
                cone_full_angle_deg=cfg['cone_full_angle_deg'],
                rr_characteristic_diameter_um=cfg['rr_characteristic_diameter_um'],
                speed_m_s=cfg['speed_m_s'])


def group_results(rows, cfg):
    groups = {}
    for row in rows:
        groups.setdefault(row['case'], []).append(row)
    result = []
    for name, group in sorted(groups.items()):
        if {r['seed'] for r in group} != set(cfg['seeds']):
            continue
        first = group[0]
        item = dict(case=name, depth_mm=first['depth_mm'], pitch_deg=first['pitch_deg'],
                    seeds=len(group), parcels_per_seed=cfg['parcels'],
                    valid=all(r['valid'] for r in group))
        if item['valid']:
            hits = [r['standard_hits'] for r in group]
            values = np.array([r['standard_volume_pct'] for r in group])
            item.update(status='positive' if sum(hits) else 'observed_zero',
                        total_roi_hits=sum(hits), total_parcels=cfg['parcels']*len(group),
                        mean_pct=float(values.mean()), min_pct=float(values.min()),
                        max_pct=float(values.max()), seed_sd_pct=float(values.std(ddof=1)))
        else:
            item.update(status='infeasible',
                        reason='; '.join(sorted({r.get('reason', '') for r in group if not r['valid']})))
        result.append(item)
    return result


def run_phase(pool, cases, rows, cfg, out, phase):
    completed = {(r['case'], r['seed']) for r in rows}
    jobs = [(case, cfg['parcels'], seed, cfg, cfg['entrance_native_mm'],
             str(out/'trajectories'))
            for case in cases for seed in cfg['seeds']
            if (case['case'], seed) not in completed]
    print(f'PHASE {phase}: {len(cases)} settings; {len(jobs)} pending seeded jobs', flush=True)
    futures = [pool.submit(run_candidate, job) for job in jobs]
    started = time.monotonic()
    with (out/'seed_results.jsonl').open('a') as stream:
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            row['phase_first_computed'] = phase
            rows.append(row)
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            if i % 25 == 0 or i == len(jobs):
                groups = group_results(rows, cfg)
                counts = {state: sum(g['status']==state for g in groups)
                          for state in ['positive', 'observed_zero', 'infeasible']}
                print(f'PROGRESS {phase} {i}/{len(jobs)}; complete settings {counts}; '
                      f'elapsed {time.monotonic()-started:.0f}s', flush=True)
                write_csv(out/'deposition_grid.csv', groups)
    return rows


def boundaries(groups):
    result = []
    for depth in sorted({g['depth_mm'] for g in groups}):
        ordered = sorted((g for g in groups if g['depth_mm']==depth),
                         key=lambda g:g['pitch_deg'])
        positive = [i for i,g in enumerate(ordered) if g['status']=='positive']
        row = dict(depth_mm=depth,
                   feasible_settings=sum(g['valid'] for g in ordered),
                   positive_settings=len(positive),
                   observed_zero_settings=sum(g['status']=='observed_zero' for g in ordered))
        if positive:
            low, high = positive[0], positive[-1]
            row.update(first_positive_pitch_deg=ordered[low]['pitch_deg'],
                       last_positive_pitch_deg=ordered[high]['pitch_deg'])
            for side, index in [('lower_neighbor',low-1),('upper_neighbor',high+1)]:
                if 0 <= index < len(ordered):
                    row[side+'_pitch_deg']=ordered[index]['pitch_deg']
                    row[side+'_status']=ordered[index]['status']
                else:
                    row[side+'_status']='outside_tested_range'
            row['internal_zero_pitch_deg']=[g['pitch_deg'] for g in ordered[low:high+1]
                                            if g['status']=='observed_zero']
        result.append(row)
    return result


def plot(groups, cfg, refined_angles, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch, Rectangle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    lookup={(g['depth_mm'],g['pitch_deg']):g for g in groups}
    depths=np.arange(cfg['depth_min_mm'],cfg['depth_max_mm']+1,cfg['depth_step_mm'])
    best=max((g for g in groups if g['valid']),key=lambda g:g['mean_pct'])
    max_value=max(best['mean_pct'],.01)
    fig,axes=plt.subplots(1,2,figsize=(13.2,6.7),layout='constrained')
    cmap=plt.get_cmap('viridis').copy()
    cmap.set_bad('#D5DCE2')
    norm=Normalize(vmin=0,vmax=max_value)
    for ax,angles,letter in zip(axes,[cfg['coarse_pitch_deg'],refined_angles],'AB'):
        angles=np.array(angles,dtype=float)
        dp=float(angles[1]-angles[0])
        values=np.full((len(angles),len(depths)),np.nan)
        for j,pitch in enumerate(angles):
            for i,depth in enumerate(depths):
                row=lookup.get((int(depth),float(pitch)))
                if row and row['valid']:
                    values[j,i]=row['mean_pct']
        im=ax.pcolormesh(np.r_[depths-.5,depths[-1]+.5],
                         np.r_[angles-dp/2,angles[-1]+dp/2],values,
                         cmap=cmap,norm=norm,shading='flat',rasterized=False)
        for j,pitch in enumerate(angles):
            for i,depth in enumerate(depths):
                row=lookup.get((int(depth),float(pitch)))
                if row and row['status']=='observed_zero':
                    ax.add_patch(Rectangle((depth-.5,pitch-dp/2),1,dp,
                                           fc='#F3F6F8',ec='none'))
                elif row is None:
                    ax.add_patch(Rectangle((depth-.5,pitch-dp/2),1,dp,
                                           fc='white',ec='#AAB5BE',hatch='///',lw=0))
        if best['pitch_deg'] in angles:
            ax.scatter([best['depth_mm']],[best['pitch_deg']],marker='*',s=170,
                       c='#F5A742',edgecolors='white',linewidths=1,zorder=5)
        ax.axvline(0,color='#7A8B98',lw=.8,ls='--')
        ax.set(xlabel='Axial insertion depth (mm)',ylabel='Nozzle pitch angle (°)',
               xticks=np.arange(np.ceil(depths[0]/10)*10,depths[-1]+1,10),
               ylim=(angles[0]-dp/2,angles[-1]+dp/2))
        ax.text(-.11,1.025,letter,transform=ax.transAxes,fontweight='bold',fontsize=15)
        ax.text(.02,1.025,f'Angle step: {dp:g}°',transform=ax.transAxes,fontsize=10)
    cb=fig.colorbar(im,ax=axes,shrink=.80,pad=.025)
    cb.set_label('Mean target deposition / emitted liquid (%)')
    axes[0].legend(handles=[Patch(fc='#F3F6F8',ec='#AAB5BE',label='Observed zero (all 3 seeds)'),
                            Patch(fc='#D5DCE2',label='Geometrically infeasible')],
                   loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False,fontsize=8)
    axes[1].text(.5,-.18,f'3 seeds × {cfg["parcels"]:,} parcels · depth step 1 mm\n'
                 f'Highest sampled mean: {best["mean_pct"]:.3f}% '
                 f'at {best["depth_mm"]:g} mm / {best["pitch_deg"]:g}°',
                 transform=axes[1].transAxes,ha='center',va='top',fontsize=9)
    for ext in ['png','svg','pdf']:
        fig.savefig(out/f'deposition_boundary_heatmap.{ext}',dpi=300,
                    bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--coarse-only',action='store_true')
    parser.add_argument('--geometry',type=Path,help='Path to the separately supplied NC04 STL.')
    parser.add_argument('--output-dir',type=Path,
                        default=Path(__file__).resolve().parents[2]/'outputs/spray_scan')
    args=parser.parse_args()
    if args.geometry:
        os.environ['NASOTWIN_NC04_GEOMETRY']=str(args.geometry.resolve())
    cfg=json.loads(args.config.read_text())
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    (out/'scan_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    (out/'trajectories').mkdir(exist_ok=True)
    rows=[]
    source=out/'seed_results.jsonl'
    if source.exists():
        rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    depths=range(cfg['depth_min_mm'],cfg['depth_max_mm']+1,cfg['depth_step_mm'])
    coarse=[case_for(d,p,cfg) for d in depths for p in cfg['coarse_pitch_deg']]
    refined_angles=cfg['coarse_pitch_deg']
    with ProcessPoolExecutor(max_workers=args.workers,initializer=solver.initialize_worker) as pool:
        rows=run_phase(pool,coarse,rows,cfg,out,'coarse')
        coarse_names={case['case'] for case in coarse}
        coarse_groups=[g for g in group_results(rows,cfg) if g['case'] in coarse_names]
        write_csv(out/'coarse_grid.csv',coarse_groups)
        positive=[g for g in coarse_groups if g['status']=='positive']
        if positive and not args.coarse_only:
            lower=max(min(cfg['coarse_pitch_deg']),
                      min(g['pitch_deg'] for g in positive)-cfg['refine_margin_deg'])
            upper=min(max(cfg['coarse_pitch_deg']),
                      max(g['pitch_deg'] for g in positive)+cfg['refine_margin_deg'])
            refined_angles=np.arange(lower,upper+1e-8,cfg['refine_pitch_step_deg']).tolist()
            print(f'REFINE angle window {lower:g}–{upper:g}°, '
                  f'step {cfg["refine_pitch_step_deg"]:g}°; all depths retained',flush=True)
            cases=[case_for(d,p,cfg) for d in depths for p in refined_angles]
            rows=run_phase(pool,cases,rows,cfg,out,'refined')
    groups=group_results(rows,cfg)
    write_csv(out/'seed_results.csv',rows)
    write_csv(out/'deposition_grid.csv',groups)
    edge=boundaries(groups)
    write_csv(out/'boundaries_by_depth.csv',edge)
    good=[g for g in groups if g['status']=='positive']
    summary=dict(configuration=cfg,complete_settings=len(groups),seeded_jobs=len(rows),
                 counts={s:sum(g['status']==s for g in groups)
                         for s in ['positive','observed_zero','infeasible']},
                 refined_pitch_deg=refined_angles,
                 best=max(good,key=lambda g:g['mean_pct']) if good else None,
                 observed_positive_bounds=(
                     {'depth_mm':[min(g['depth_mm'] for g in good),max(g['depth_mm'] for g in good)],
                      'pitch_deg':[min(g['pitch_deg'] for g in good),max(g['pitch_deg'] for g in good)]}
                     if good else None),
                 scope='Scope: tested feasible settings; zero sampled hits are retained as finite-sample observations. '
                       'Sampling comprises the coarse grid and the listed local refinement window.')
    (out/'boundary_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(groups,cfg,refined_angles,out)
    print('COMPLETE '+json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

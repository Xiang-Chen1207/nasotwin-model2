"""Compute NC04 olfactory ROI masks and screen spray directions in still air.

The Lagrangian solver advances equal-volume droplets through the NC04 geometry.
Triangle intersections assign first-hit deposition to fixed ROI masks. The
prepare command exports annotated surfaces and views; the sweep command computes
spray settings with parallel CPU workers."""
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
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
import pyvista as pv
import vtk

from solve_anatomy import SurfaceCollision, drag_rate, frozen_step

BASE = Path(__file__).resolve().parent
CONFIG = BASE / 'assets/nc04/olfactory_roi.json'
OUT = BASE / 'outputs/nc04_angles'
REGIONS = ('narrow', 'standard', 'wide')
COLLISION = None
SURFACE = None
MASKS = None
CFG = None
ROI_AREAS_M2 = None


def geometry_path(config):
    """Resolve the NC04 STL from the environment or ROI configuration."""
    external = os.environ.get('NASOTWIN_NC04_GEOMETRY')
    return Path(external).resolve() if external else CONFIG.parent / config['geometry']


def roi_masks(surface, config):
    centres = surface.cell_centers().points
    result = {}
    for name, bounds in config['bounds_mm'].items():
        b = np.asarray(bounds).reshape(3, 2)
        result[name] = np.all((centres >= b[:, 0]) & (centres <= b[:, 1]), axis=1)
    return result


def axis_native(pitch, yaw):
    p, y = np.deg2rad([pitch, yaw])
    return np.array([np.cos(p)*np.cos(y), np.sin(p), np.cos(p)*np.sin(y)])


def to_simulation(points):
    # Proper cyclic coordinate permutation: simulation x=lateral, y=posterior,
    # z=superior. Existing gravitational acceleration then points downward.
    return np.asarray(points)[..., [2, 0, 1]] * 1e-3


def to_native(points):
    return np.asarray(points)[..., [1, 2, 0]] * 1e3


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text())
    surface = pv.read(geometry_path(config))
    masks = roi_masks(surface, config)
    surface.cell_data['original_triangle_id'] = np.arange(surface.n_cells)
    area = surface.compute_cell_sizes(length=False, area=True, volume=False).cell_data['Area']
    stats = {}
    for name in REGIONS:
        surface.cell_data['olfactory_candidate_' + name] = masks[name].astype(np.uint8)
        part = surface.extract_cells(masks[name]).extract_surface(algorithm='dataset_surface')
        part.save(OUT / ('olfactory_' + name + '.stl'))
        stats[name] = {'area_mm2': float(area[masks[name]].sum()), 'triangles': int(masks[name].sum())}
    surface.field_data['annotation'] = [config['annotation_type']]
    surface.save(OUT / 'nc04_labelled_surface.vtp')
    (OUT / 'roi_areas.json').write_text(json.dumps(stats, indent=2) + '\n')
    centres = surface.cell_centers().points
    show = ((centres[:, 0] > 23) & (centres[:, 0] < 105)
            & (centres[:, 1] > -25) & (centres[:, 1] < 60)
            & (np.abs(centres[:, 2]) < 27))
    context = surface.extract_cells(show).extract_surface(algorithm='dataset_surface')
    plot = pv.Plotter(shape=(1, 2), off_screen=True, window_size=(2200, 1050))
    for panel in range(2):
        plot.subplot(0, panel)
        plot.set_background('white')
        plot.add_mesh(context, color='#b9c6ca', opacity=0.22, smooth_shading=True)
        for name, mask, color in [
            ('Wide envelope', masks['wide'] & ~masks['standard'], '#d0a6d6'),
            ('Standard envelope', masks['standard'] & ~masks['narrow'], '#9f5ca8'),
            ('Narrow envelope', masks['narrow'], '#633878')]:
            part = surface.extract_cells(mask).extract_surface(algorithm='dataset_surface')
            plot.add_mesh(part, color=color, smooth_shading=True, label=name)
        tip = np.array(config['spray']['nozzle_tip_native_mm'], dtype=float)
        plot.add_mesh(pv.Sphere(radius=0.8, center=tip), color='#d89020')
        if panel == 0:
            for pitch in config['spray']['pitch_deg']:
                plot.add_arrows(tip[None], (axis_native(pitch, 0)*17)[None], mag=1, color='#237a9a')
            plot.camera_position = [(55, 20, 200), (55, 20, 0), (0, 1, 0)]
            plot.add_point_labels([[49, 49, 0], [54, 19, 13], tip],
                                  ['Candidate olfactory region', 'Nasal passages', 'Virtual spray tip'],
                                  font_size=18, text_color='#263746', point_size=3,
                                  shape_color='white', shape_opacity=0.85, always_visible=True)
        else:
            plot.camera_position = [(200, 28, 0), (50, 28, 0), (0, 1, 0)]
            plot.add_point_labels([[49., 47., -4.]],
                                  ['Upper septal / roof walls'],
                                  font_size=18, text_color='#263746', point_size=3,
                                  shape_color='white', shape_opacity=0.85, always_visible=True)
        plot.enable_parallel_projection()
        plot.reset_camera()
        plot.add_legend(bcolor='white', size=(0.32, 0.13), loc='lower left', face='circle')
        plot.add_axes(xlabel='Posterior X', ylabel='Superior Y', zlabel='Lateral Z',
                      viewport=(0.80, 0.05, 0.98, 0.22))
        plot.add_text('NC04 | candidate ROI defined by geometric coordinates',
                      position='lower_edge', font_size=11, color='#48545d')
    plot.screenshot(OUT / 'nc04_olfactory_regions.png')
    plot.close()
    print(json.dumps(stats, indent=2), flush=True)


def initialize_worker():
    global COLLISION, SURFACE, MASKS, CFG, ROI_AREAS_M2
    vtk.vtkSMPTools.Initialize(2)
    CFG = json.loads(CONFIG.read_text())
    native = pv.read(geometry_path(CFG))
    MASKS = roi_masks(native, CFG)
    areas = np.asarray(native.compute_cell_sizes(length=False, area=True, volume=False).cell_data['Area']) * 1e-6
    ROI_AREAS_M2 = np.array([areas[MASKS[name]].sum() for name in REGIONS])
    SURFACE = native.copy(deep=True)
    SURFACE.points = to_simulation(native.points)
    origin = (-0.035, -0.01, -0.05)
    grid = pv.ImageData(dimensions=(151, 251, 231), spacing=(0.0005,)*3, origin=origin)
    COLLISION = SurfaceCollision(SURFACE, grid, np.nan, (0, 0, 0, 0), 0,
                                 require_outlet=False)


def injection(pitch, yaw, count, seed, nozzle_tip=None, spray_overrides=None):
    # Copy the worker baseline into the settings for this spray configuration.
    s = dict(CFG['spray'])
    if spray_overrides:
        allowed = {'rr_characteristic_diameter_um', 'rr_shape', 'diameter_min_um',
                   'diameter_max_um', 'speed_m_s', 'cone_full_angle_deg',
                   'breakup_distance_mm', 'release_radius_mm'}
        if set(spray_overrides) - allowed:
            raise ValueError('Unsupported spray override: ' + str(set(spray_overrides) - allowed))
        s.update(spray_overrides)
    if not (0 < s['diameter_min_um'] < s['diameter_max_um']
            and s['rr_shape'] > 0 and s['rr_characteristic_diameter_um'] > 0
            and s['speed_m_s'] > 0 and 0 <= s['cone_full_angle_deg'] < 180
            and s['release_radius_mm'] >= 0 and s['breakup_distance_mm'] >= 0):
        raise ValueError('Invalid physical spray parameters.')
    rng = np.random.default_rng(seed)
    shape, scale = s['rr_shape'], s['rr_characteristic_diameter_um']
    lower = -np.expm1(-(s['diameter_min_um']/scale)**shape)
    upper = -np.expm1(-(s['diameter_max_um']/scale)**shape)
    quantile = (np.arange(count) + rng.random(count))/count
    diameter = scale * (-np.log1p(-(lower + (upper-lower)*quantile)))**(1/shape) * 1e-6
    rng.shuffle(diameter)
    # Equal liquid-volume parcels sample the volume/mass Rosin-Rammler law.
    axis = axis_native(pitch, yaw)[[2, 0, 1]]
    tangent = np.cross(axis, [0., 0., 1.]); tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(axis, tangent)
    tip = s['nozzle_tip_native_mm'] if nozzle_tip is None else nozzle_tip
    centre = to_simulation(tip) + s['breakup_distance_mm']*1e-3*axis
    radius = s['release_radius_mm']*1e-3*np.sqrt(rng.random(count))
    azimuth = rng.uniform(0, 2*np.pi, count)
    positions = centre + (radius*np.cos(azimuth))[:, None]*tangent + (radius*np.sin(azimuth))[:, None]*bitangent
    cosine = rng.uniform(np.cos(np.deg2rad(s['cone_full_angle_deg']/2)), 1, count)
    sine = np.sqrt(1-cosine**2)
    azimuth = rng.uniform(0, 2*np.pi, count)
    direction = cosine[:, None]*axis + (sine*np.cos(azimuth))[:, None]*tangent + (sine*np.sin(azimuth))[:, None]*bitangent
    return positions, s['speed_m_s']*direction, diameter, axis


def simulate(job, *, nozzle_tip=None, case_name=None, output_dir=None,
             sample_count=160, frame_times=None, spray_overrides=None):
    pitch, yaw, count, seed, end_time, dt_max = job
    started = time.monotonic()
    position, velocity, diameter, axis = injection(pitch, yaw, count, seed, nozzle_tip, spray_overrides)
    # Use a fine relative tolerance to resolve nearby nasal-wall intersections.
    inside = pv.PolyData(position).select_enclosed_points(
        SURFACE, tolerance=1e-8, check_surface=False).point_data['SelectedPoints']
    initial = position.copy()
    case = case_name or f'pitch_{pitch:02d}_yaw_{yaw:+03d}'
    row = {'case': case, 'pitch_deg': pitch, 'yaw_deg': yaw, 'parcels': count, 'seed': seed,
           'valid': bool(np.all(inside)), 'invalid_injections': int(np.count_nonzero(~inside.astype(bool)))}
    if not row['valid']:
        print('Excluded invalid injection:', row, flush=True)
        return row
    status = np.zeros(count, dtype=np.int8)
    hit_cells = np.full(count, -1, dtype=np.int64)
    hit_time = np.full(count, np.nan)
    sample = np.arange(min(sample_count, count))
    times = np.linspace(0, end_time, 151) if frame_times is None else np.asarray(frame_times, dtype=float)
    paths = np.zeros((len(times), len(sample), 3), dtype=np.float32)
    paths[0] = to_native(position[sample])
    time_s, steps = 0., 0
    for frame in range(1, len(times)):
        target = times[frame]
        while time_s < target - 1e-13:
            active = np.flatnonzero(status == 0)
            if not len(active):
                time_s = target
                break
            pos, vel, d = position[active], velocity[active], diameter[active]
            air = np.zeros_like(vel)
            beta = drag_rate(vel, air, d)
            reynolds = 1.225*d*np.linalg.norm(vel, axis=1)/1.81e-5
            # Resolve the nonlinear transient while Re >= 1. At low Re the
            # exponential step handles stiff, near-Stokes relaxation directly.
            nonlinear = reynolds >= 1
            relaxation_cap = float(np.min(0.25/beta[nonlinear])) if np.any(nonlinear) else dt_max
            displacement_cap = 0.0002/max(float(np.linalg.norm(vel, axis=1).max()), 1e-9)
            dt = min(dt_max, relaxation_cap, displacement_cap, target-time_s)
            mid, mid_vel = frozen_step(pos, vel, air, beta, dt/2)
            next_pos, next_vel = frozen_step(pos, vel, air, drag_rate(mid_vel, air, d), dt)
            fraction, point, cell = COLLISION.intersect(pos, next_pos)
            hit = np.isfinite(fraction)
            next_pos[hit] = point[hit]
            position[active], velocity[active] = next_pos, next_vel
            status[active[hit]] = 1
            hit_cells[active[hit]] = cell[hit]
            hit_time[active[hit]] = time_s + dt*fraction[hit]
            time_s += dt
            steps += 1
        paths[frame] = to_native(position[sample])
    hit = status == 1
    row.update({'surface_hit_count': int(hit.sum()), 'airborne_count': int((~hit).sum()),
                'surface_hit_volume_pct': 100*float(hit.mean()), 'airborne_volume_pct': 100*float((~hit).mean()),
                'steps': steps, 'elapsed_s': time.monotonic()-started})
    row.update({f'dv{q}_um': float(np.percentile(diameter*1e6, q)) for q in (10, 50, 90)})
    roi_hits = []
    for name in REGIONS:
        selected = hit & MASKS[name][np.maximum(hit_cells, 0)]
        roi_hits.append(selected)
        row[name+'_hits'] = int(selected.sum())
        row[name+'_volume_pct'] = 100*float(selected.mean())
    assert row['surface_hit_count']+row['airborne_count'] == count
    assert row['narrow_hits'] <= row['standard_hits'] <= row['wide_hits']
    resolved_spray = dict(CFG['spray'], **(spray_overrides or {}))
    resolved_spray.update(pitch_deg=pitch, yaw_deg=yaw,
                          nozzle_tip_native_mm=np.asarray(CFG['spray']['nozzle_tip_native_mm']
                              if nozzle_tip is None else nozzle_tip).tolist())
    np.savez_compressed((OUT if output_dir is None else Path(output_dir)) / (case+'.npz'), positions_native_mm=to_native(position),
                        injection_native_mm=to_native(initial), diameter_um=diameter*1e6,
                        status=status, hit_triangle_id=hit_cells, hit_time_s=hit_time,
                        parcel_volume_weight=np.full(count, 1/count),
                        sample_paths_native_mm=paths, sample_particle_id=sample, times_s=times,
                        spray_axis_native=axis[[1, 2, 0]],
                        roi_names=np.asarray(REGIONS), roi_hit_mask=np.asarray(roi_hits),
                        roi_area_m2=ROI_AREAS_M2,
                        spray_settings_json=json.dumps(resolved_spray))
    print(f'{case}: olfactory standard={row["standard_volume_pct"]:.4f}%, '
          f'hits={row["standard_hits"]}/{count}, elapsed={row["elapsed_s"]:.1f}s', flush=True)
    return row


def sweep(args):
    config = json.loads(CONFIG.read_text())
    jobs = [(pitch, yaw, args.parcels, args.seed, args.end_time, args.dt)
            for pitch in config['spray']['pitch_deg'] for yaw in config['spray']['yaw_deg']]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker) as pool:
        rows = list(pool.map(simulate, jobs))
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT/'angle_results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    valid = [r for r in rows if r['valid']]
    if not valid:
        raise RuntimeError('Every injection was outside the fluid volume; no angle ranking is defined.')
    best_hits = max(row['standard_hits'] for row in valid)
    tied = [row for row in valid if row['standard_hits'] == best_hits]
    best = max(tied, key=lambda row: row['wide_hits'])
    summary = {'model': 'NC04 breath-hold, quiescent-air Lagrangian initial screening',
               'configuration': config, 'parcels_per_angle': args.parcels, 'seed': args.seed,
               'flight_time_s': args.end_time, 'maximum_time_step_s': args.dt,
               'best_sampled_angle': best, 'valid_cases': len(valid),
               'top_standard_roi_cases': [row['case'] for row in tied],
               'unique_sampled_maximum': len(tied) == 1,
               'representative_tie_break': 'standard-ROI tie retained; representative ordered by larger wide-envelope dose',
               'sampling': 'common random draws across angles; equal-volume stratified Rosin-Rammler samples',
               'gravity_native_m_s2': [0, -9.81, 0], 'drag': 'Schiller-Naumann',
               'wall_policy': 'first exact triangle intersection terminates point droplet; all non-ROI surface hits include other nasal walls, external face and computational boundaries',
               'limitations': ['candidate ROI defined by the recorded geometric coordinates and boundary rules',
                              'still-air motion with gravity and Schiller-Naumann drag', 'virtual point nozzle with spatial path-clearance constraints',
                              'search scope: one NC04 geometry, one random sample and the listed candidate angles',
                              'flight model: each droplet is integrated to first wall contact; deposition supplies the local mucus model',
                              'dose metric: deposited liquid volume; CFU mapping requires bacterial concentration and viability inputs']}
    (OUT/'angle_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print('SWEEP COMPLETE', json.dumps(best), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'sweep'])
    parser.add_argument('--parcels', type=int, default=6000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--end-time', type=float, default=0.3)
    parser.add_argument('--dt', type=float, default=0.0001)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--geometry', type=Path, help='Path to the separately supplied NC04 STL.')
    args = parser.parse_args()
    if args.geometry:
        os.environ['NASOTWIN_NC04_GEOMETRY'] = str(args.geometry.resolve())
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'prepare': prepare()
    else: sweep(args)


if __name__ == '__main__':
    main()

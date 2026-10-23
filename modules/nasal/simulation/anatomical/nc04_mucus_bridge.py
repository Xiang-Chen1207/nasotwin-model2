"""Transfer NC04 ROI deposition into the mucus transport model.

The posterior coordinates of deposited parcels define an equivalent local strip.
Its area matches the ROI wall area, and parcel weights preserve the deposited
dose during the transfer to axial mucus cells."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
SIM = BASE.parent


def make_handoff(archive_path, config, roi_config, mucus_overrides=None):
    """Construct a mucus configuration and source with the original dose weights."""
    with np.load(archive_path, allow_pickle=False) as archive:
        region = list(archive['roi_names']).index('standard')
        selected = archive['roi_hit_mask'][region].astype(bool)
        positions = archive['positions_native_mm'][selected]
        weights = archive['parcel_volume_weight'][selected]
        area = float(archive['roi_area_m2'][region])
        handoff_time = float(archive['times_s'][-1])
    bounds = np.asarray(roi_config['bounds_mm']['standard']).reshape(3, 2)
    # Extend the axial strip to include actual hit coordinates from every
    # centroid-labelled ROI triangle.
    start = min(bounds[0, 0], float(positions[:, 0].min())) if len(positions) else bounds[0, 0]
    end = max(bounds[0, 1], float(positions[:, 0].max())) if len(positions) else bounds[0, 1]
    length = (end-start)*1e-3
    cfg = copy.deepcopy(config)
    cfg['mucus'].update(mucus_overrides or {})
    cfg['mucus'].update(length_m=length, width_m=area/length)
    source = {'deposition_x_m': (positions[:, 0]-start)*1e-3,
              'deposition_weight': weights}
    fraction = float(weights.sum())
    meta = {'source_archive': str(archive_path), 'region': 'standard',
            'deposited_volume_fraction_of_emitted': fraction,
            'original_triangle_roi_area_m2': area, 'posterior_origin_mm': float(start),
            'equivalent_length_m': length, 'equivalent_width_m': area/length,
            'spray_handoff_time_s': handoff_time,
            'initial_cfu_expected': [fraction*s['emitted_cfu']*s['post_spray_culturability']
                                     for s in cfg['strains']],
            'assumption': 'Model conditions: uniform bacterial concentration and strain survival across parcels; instantaneous handoff at flight completion; a projected local strip with zero inflow from other nasal regions.',
            'mucus': cfg['mucus'], 'biology': cfg['biology'], 'strains': cfg['strains']}
    return cfg, source, meta


def run_handoff(archive_path, output_dir, config, roi_config, mucus_overrides=None):
    sys.path.insert(0, str(SIM))
    from mucus import run_mucus
    cfg, source, meta = make_handoff(archive_path, config, roi_config, mucus_overrides)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = run_mucus(cfg, source)
    initial = result['initial_deposited_cfu']
    if not np.allclose(initial, meta['initial_cfu_expected'], rtol=1e-12, atol=1e-10):
        raise RuntimeError('NC04-to-mucus handoff lost or duplicated CFU.')
    if result['max_relative_balance_error'] > 1e-7 or result['minimum_cfu'] < -1e-7:
        raise RuntimeError('Mucus solution failed its count-balance or positivity criterion.')
    result['strain_names'] = np.array([s['name'] for s in cfg['strains']])
    result['global_times_s'] = result['mucus_times_s'] + meta['spray_handoff_time_s']
    np.savez_compressed(out/'retention.npz', **result)
    with (out/'retention.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['time_after_handoff_s', 'time_after_spray_start_s', 'strain',
                         'mobile_cfu', 'attached_cfu', 'total_cfu', 'cleared_cfu',
                         'net_bio_cfu', 'exposure_cfu_h'])
        for t, time_s in enumerate(result['mucus_times_s']):
            for i, strain in enumerate(cfg['strains']):
                mobile = float(result['mobile_cfu'][t, i].sum())
                attached = float(result['attached_cfu'][t, i].sum())
                writer.writerow([time_s, result['global_times_s'][t], strain['name'],
                                 mobile, attached, mobile+attached, result['cleared_cfu'][t, i],
                                 result['net_bio_cfu'][t, i], result['exposure_cfu_h'][t, i]])
    meta['max_relative_balance_error'] = result['max_relative_balance_error']
    meta['final_retained_cfu'] = (result['mobile_cfu'][-1].sum(axis=(1, 2))
                                  + result['attached_cfu'][-1].sum(axis=1)).tolist()
    meta['final_attached_cfu'] = result['attached_cfu'][-1].sum(axis=1).tolist()
    meta['exposure_cfu_h'] = result['exposure_cfu_h'][-1].tolist()
    (out/'model_inputs.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    return meta

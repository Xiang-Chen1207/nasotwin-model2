"""Screen NC04 spray settings and couple ROI deposition to mucus transport.

A fixed standard ROI ranks the spray settings. Independent seeds measure Monte
Carlo variation, and the selected parcel fields initialize the local mucus model."""
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import nc04_angles as solver
from nc04_mucus_bridge import run_handoff

BASE = Path(__file__).resolve().parent
CONFIG = BASE / 'assets/nc04/optimization_config.json'
OUT = BASE / 'outputs/nc04_optimized'
SPRAY_KEYS = ('rr_characteristic_diameter_um', 'cone_full_angle_deg', 'speed_m_s')


def make_candidates(config):
    baseline = dict(config['baseline'])
    keys = tuple(baseline)
    cases = [dict(baseline, case='baseline')]
    seen = {tuple(baseline[k] for k in keys)}

    def add(values, name):
        identity = tuple(values[k] for k in keys)
        if identity in seen:
            return False
        seen.add(identity)
        cases.append(dict(values, case=name))
        return True

    for key, values in config['search_values'].items():
        for value in values:
            add(dict(baseline, **{key: value}), f'one_{key}_{value:g}')
    rng = np.random.default_rng(config['design_seed'])
    total = int(np.prod([len(config['search_values'][k]) for k in keys]))
    target = min(config['joint_candidates'], total - len(cases))
    count = 0
    while count < target:
        values = {k: float(rng.choice(config['search_values'][k])) for k in keys}
        if add(values, f'joint_{count+1:02d}'):
            count += 1
    return cases


def run_candidate(job):
    case, count, seed, config, entrance, directory = job
    axis = solver.axis_native(case['pitch_deg'], case['yaw_deg'])
    entrance = np.asarray(entrance, dtype=float)
    tip = entrance + case['depth_mm'] * axis
    overrides = {key: case[key] for key in SPRAY_KEYS}
    name = f'{case["case"]}_seed{seed}'
    extra = dict(case, seed=seed, parcels=count, tip_native_mm=tip.tolist(),
                 trajectory_path=str(Path(directory) / (name + '.npz')))
    # Test the insertion path from the external side, then each release segment.
    start = solver.to_simulation(entrance - 10 * axis)[None, :]
    end = solver.to_simulation(tip)[None, :]
    fraction, _, _ = solver.COLLISION.intersect(start, end)
    if np.any(np.isfinite(fraction)):
        print(f'{name}: excluded, insertion axis crosses a wall', flush=True)
        return dict(extra, valid=False, reason='Insertion axis crosses an original surface triangle.')
    position, _, _, _ = solver.injection(case['pitch_deg'], case['yaw_deg'], count, seed,
                                         nozzle_tip=tip, spray_overrides=overrides)
    starts = np.broadcast_to(end, position.shape).copy()
    fraction, _, _ = solver.COLLISION.intersect(starts, position)
    blocked = int(np.count_nonzero(np.isfinite(fraction)))
    if blocked:
        print(f'{name}: excluded, {blocked} blocked release segments', flush=True)
        return dict(extra, valid=False, blocked_release_segments=blocked,
                    reason='Tip-to-release segments cross original surface triangles.')
    row = solver.simulate(
        (case['pitch_deg'], case['yaw_deg'], count, seed,
         config['flight_time_s'], config['maximum_time_step_s']),
        nozzle_tip=tip, case_name=name, output_dir=directory,
        spray_overrides=overrides, sample_count=0)
    row.update(extra)
    row['reason'] = '' if row['valid'] else 'Some release points are outside the fluid volume.'
    return row


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_confirmation(rows, case_names, seeds):
    by_key = {(r['case'], r['seed']): r for r in rows}
    summaries = []
    for name in case_names:
        group = [by_key[name, seed] for seed in seeds]
        if not all(r['valid'] for r in group):
            summaries.append({'case': name, 'valid': False,
                              'reason': 'Not feasible for every independent release sample.'})
            continue
        item = {'case': name, 'valid': True, 'seeds': seeds}
        for region in solver.REGIONS:
            values = np.array([r[region + '_volume_pct'] for r in group])
            item[region + '_mean_pct'] = float(values.mean())
            item[region + '_min_pct'] = float(values.min())
            item[region + '_max_pct'] = float(values.max())
            item[region + '_sample_std_pct'] = float(values.std(ddof=1))
        if all(by_key['baseline', seed]['valid'] for seed in seeds):
            difference = [by_key[name, seed]['standard_volume_pct']
                          - by_key['baseline', seed]['standard_volume_pct'] for seed in seeds]
            item['paired_standard_gain_percentage_points'] = difference
            item['paired_gain_positive_all_seeds'] = all(value > 0 for value in difference)
        summaries.append(item)
    return summaries


def plot_comparison(summary, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    groups = [row for row in summary['confirmation_summary'] if row['valid']]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout='constrained')
    means = np.array([row['standard_mean_pct'] for row in groups])
    errors = np.array([[row['standard_mean_pct'] - row['standard_min_pct'] for row in groups],
                       [row['standard_max_pct'] - row['standard_mean_pct'] for row in groups]])
    axes[0].bar(np.arange(len(groups)), means, color='#3587a4', yerr=errors, capsize=4)
    axes[0].set_xticks(np.arange(len(groups)), [r['case'] for r in groups], rotation=20, ha='right')
    axes[0].set(ylabel='Direct deposition / emitted liquid volume (%)',
                title='Fixed standard ROI: mean and full seed range')
    for label, item in summary['mucus_handoffs'].items():
        with np.load(Path(item['output_dir']) / 'retention.npz') as values:
            mobile = values['mobile_cfu'].sum(axis=(2, 3))
            attached = values['attached_cfu'].sum(axis=2)
            # Sum the four strain populations with their shared transport parameters.
            total = (mobile + attached).sum(axis=1)
            axes[1].plot(values['mucus_times_s']/60, total, label=label)
            axes[1].plot(values['mucus_times_s']/60, attached.sum(axis=1), linestyle='--', alpha=.7)
    axes[1].set(xlabel='Time after local mucus handoff (min)', ylabel='Four-strain total (CFU)',
                title='Local mucus scenario: solid total, dashed attached')
    axes[1].legend()
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.suptitle('NC04 | still-air deposition and reference-parameter retention', fontsize=12)
    fig.savefig(output_dir / 'comparison.png', dpi=200)
    fig.savefig(output_dir / 'comparison.svg')
    plt.close(fig)


def complete_mucus_comparisons(summary, out):
    """Compare mucus dynamics for every confirmed spray setting.

    Reuse completed case/seed handoffs when resuming the postprocessing stage.
    """
    config = summary['configuration']
    roi = summary['spray_and_roi_configuration']
    mucus_config = json.loads((BASE.parent / 'config.json').read_text())
    repeats = summary.setdefault('mucus_all_seed_handoffs', [])
    handoffs = summary.setdefault('mucus_handoffs', {})
    existing = {(row['case'], row['seed']) for row in repeats}
    names = [row['case'] for row in summary['confirmation_summary'] if row['valid']]
    for name in names:
        for seed in config['confirm_seeds']:
            if (name, seed) in existing:
                continue
            archive = out / 'confirm' / f'{name}_seed{seed}.npz'
            target = out / 'mucus' / f'{name}_seed{seed}'
            metadata = run_handoff(archive, target, mucus_config, roi, config['mucus'])
            item = dict(metadata, case=name, seed=seed, output_dir=str(target))
            repeats.append(item)
            if seed == config['confirm_seeds'][0]:
                handoffs[name] = item
            print(f'MUCUS: {name}, seed={seed}, initial={metadata["initial_cfu_expected"]}', flush=True)
    (out / 'optimization_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    plot_comparison(summary, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--mucus-only', action='store_true',
                        help='Reuse completed deposition results and finish missing mucus comparisons.')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.mucus_only:
        summary = json.loads((out / 'optimization_summary.json').read_text())
        complete_mucus_comparisons(summary, out)
        print('NC04 MUCUS COMPARISONS COMPLETE: ' + str(out), flush=True)
        return
    config = json.loads(args.config.read_text())
    insertion = json.loads((BASE / 'assets/nc04/insertion_depth_config.json').read_text())
    roi = json.loads(solver.CONFIG.read_text())
    for phase in ('screen', 'confirm', 'mucus'):
        (out / phase).mkdir(parents=True, exist_ok=True)
    cases = make_candidates(config)
    write_csv(out / 'candidate_settings.csv', cases)
    print(f'SCREEN: {len(cases)} cases, {config["screen_parcels"]} parcels each', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=solver.initialize_worker) as pool:
        jobs = [(case, config['screen_parcels'], config['screen_seed'], config,
                 insertion['entrance_native_mm'], str(out / 'screen')) for case in cases]
        screen = list(pool.map(run_candidate, jobs))
        write_csv(out / 'screen_results.csv', screen)
        valid = [row for row in screen if row['valid']]
        if not valid or not next(r for r in screen if r['case'] == 'baseline')['valid']:
            raise RuntimeError('No valid baseline comparison; inspect screen_results.csv.')
        # Rank settings by the fixed standard ROI deposition endpoint.
        ranking = sorted(valid, key=lambda r: (-r['standard_volume_pct'], r['case']))
        top_value = ranking[0]['standard_volume_pct']
        ties = [row['case'] for row in ranking if row['standard_volume_pct'] == top_value]
        selected_name = 'baseline' if 'baseline' in ties else ties[0]
        names = list(dict.fromkeys(['baseline', selected_name]
                                  + [r['case'] for r in ranking[:config['confirm_top_k']]]))
        selected_cases = [case for name in names for case in cases if case['case'] == name]
        print('CONFIRM: frozen screening candidates ' + ', '.join(names), flush=True)
        jobs = [(case, config['confirm_parcels'], seed, config,
                 insertion['entrance_native_mm'], str(out / 'confirm'))
                for case in selected_cases for seed in config['confirm_seeds']]
        confirmed = list(pool.map(run_candidate, jobs))
    write_csv(out / 'confirmation_results.csv', confirmed)
    repeated = summarize_confirmation(confirmed, names, config['confirm_seeds'])
    write_csv(out / 'confirmation_summary.csv', repeated)
    # Carry every confirmed setting into the downstream mucus comparison.
    # Keep the screening winner associated with the original screening stage.
    summary = {
        'configuration': config, 'spray_and_roi_configuration': roi,
        'entrance_native_mm': insertion['entrance_native_mm'],
        'screen_cases': len(cases), 'valid_screen_cases': len(valid),
        'excluded_screen_cases': [r for r in screen if not r['valid']],
        'screen_top_ties': ties, 'frozen_screen_selected_case': selected_name,
        'selection_resolved_by_screen': len(ties) == 1 and top_value > 0,
        'selected_settings': next(case for case in cases if case['case'] == selected_name),
        'confirmation_summary': repeated,
        'mucus_handoffs': {}, 'mucus_all_seed_handoffs': [],
        'interpretation': [
            'Standard ROI is fixed. Narrow/wide masks assess annotation sensitivity only.',
            'The automatically selected spray setting is fixed before three independent confirmation seeds.',
            'Independent Monte Carlo seeds describe numerical variation under the fixed model settings.',
            'Motion model: still-air droplets with gravity and Schiller-Naumann drag, integrated to first wall contact; deposited parcels supply the local mucus model.',
            'Nozzle: virtual point source satisfying the configured spatial path-clearance constraints.',
            'ROI: candidate olfactory envelope defined by the recorded NC04 geometric coordinates.',
            'Liquid-to-CFU mapping assumes uniform bacterial concentration and the configured survival fraction.',
            'Mucus handoff projects only standard-ROI hits onto a local strip of equal wall area.',
            'Local retention domain: the selected ROI mucus strip, with zero inflow from other nasal regions.',
            'Mucus speed, diffusivity, adhesion and growth use the listed transferred or reference parameters; local calibration requires matching measurements.',
            'Comparison scope: the best deposition setting found within this NC04 mesh, still-air model and sampled parameter range.',
        ],
    }
    complete_mucus_comparisons(summary, out)
    print('NC04 SIMULATION COMPLETE: ' + str(out), flush=True)


if __name__ == '__main__':
    main()

"""Couple three selected spray seeds to mucus transport and fitted growth inputs."""
from pathlib import Path
import argparse
import csv
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "coupling"
SIM = ROOT / "simulation"
for key, subdir in {
    "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
    "XDG_CACHE_HOME": "cache", "MPLCONFIGDIR": "mpl",
}.items():
    p = ROOT / ".build" / f"coupling-runtime-{os.getuid()}" / subdir
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)
os.environ["FIPY_SOLVERS"] = "scipy"

import numpy as np
from scipy.integrate import cumulative_simpson
from scipy.sparse.linalg import expm_multiply

SEEDS = [7717, 8831, 9941]
GROUPS = ["WT", "Lp-NAC", "Lp-SFN", "Lp-Q"]
SCENARIOS = ["full", "no_growth", "no_attachment", "faster_clearance"]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_fits(workbook=None):
    from plot_common import growth, culture_metrics
    times, groups = growth(workbook)
    rows = []
    for group, records in groups.items():
        for record in records:
            r, k, lag = record["p"]
            rows.append(dict(strain=group, record=record["name"], b0=record["b0"],
                             r_h_inv=r, k_od=k, lambda_h=lag,
                             observations=len(times), observation_start_h=float(times[0]),
                             observation_end_h=float(times[-1])))
    write_csv(OUT / "growth_fits.csv", rows)
    write_csv(OUT / "culture_metrics.csv", culture_metrics(times, groups))
    print(f"Saved {len(rows)} growth fits and observed culture metrics.", flush=True)


def relative_growth(t, row):
    """Return the growth multiplier and logarithmic rate in inverse seconds."""
    r, k, lag, b0 = [float(row[x]) for x in ("r_h_inv", "k_od", "lambda_h", "b0")]
    hours = np.asarray(t, dtype=float) / 3600
    if lag <= 1e-6:
        # Evaluate the Logistic limit with its finite initial growth rate.
        adjusted = hours
        startup = np.ones_like(hours)
    else:
        adjusted = hours + lag * np.expm1(-hours / lag)
        startup = -np.expm1(-hours / lag)
    b = k / (1 + (k / b0 - 1) * np.exp(-r * adjusted))
    factor = b / b0
    rate = r * startup * (1 - b / k) / 3600
    return factor, rate


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-fits", action="store_true")
    parser.add_argument("--workbook", type=Path, help="Growth workbook used by --prepare-fits.")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--fits", type=Path, default=ROOT / "outputs/coupling/growth_fits.csv")
    parser.add_argument("--trajectory-dir", type=Path, default=ROOT / "spray_boundary_pilot/trajectories")
    parser.add_argument("--config", type=Path, default=SIM / "config.json")
    parser.add_argument("--roi-config", type=Path, default=SIM / "anatomical/assets/nc04/olfactory_roi.json")
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    if args.prepare_fits:
        prepare_fits(args.workbook)
        return

    with args.fits.open() as stream:
        fit_rows = list(csv.DictReader(stream))
    if len(fit_rows) != len(GROUPS) or sorted(row['strain'] for row in fit_rows) != sorted(GROUPS):
        raise ValueError("Growth-fit CSV must contain one row for each of WT, Lp-NAC, Lp-SFN, and Lp-Q.")
    sys.path.insert(0, str(SIM))
    from mucus import transport_operator, coupled_operator
    cfg = json.loads(args.config.read_text())
    roi = json.loads(args.roi_config.read_text())
    positions, weights, area = [], [], None
    for seed in SEEDS:
        path = args.trajectory_dir / f"d+15_p037p5_seed{seed}.npz"
        with np.load(path) as archive:
            index = list(archive["roi_names"]).index("standard")
            mask = archive["roi_hit_mask"][index].astype(bool)
            positions.append(archive["positions_native_mm"][mask, 0])
            weights.append(archive["parcel_volume_weight"][mask])
            area = float(archive["roi_area_m2"][index])
    limits = roi["bounds_mm"]["standard"]
    start = min(limits[0], min(p.min() for p in positions))
    end = max(limits[1], max(p.max() for p in positions))
    params = dict(cfg["mucus"], length_m=(end-start)/1000,
                  width_m=area/((end-start)/1000), duration_s=3600, output_frames=721,
                  velocity_m_s=1/60000)
    biology = dict(cfg["biology"], attachment_Da=[0., 1.], baseline_Da=1.)
    nx, nz = params["nx"], params["nz"]
    n = nx * nz
    times = np.linspace(0, 3600, 721)
    initial = np.zeros((n + nx + 3, len(SEEDS)))
    emitted = 1e6
    for j, (x, w) in enumerate(zip(positions, weights)):
        counts, _ = np.histogram((x-start)/1000, bins=np.linspace(0, params["length_m"], nx+1),
                                 weights=w * emitted)
        initial[(nz-1)*nx:nz*nx, j] = counts
    initial_counts = initial.sum(axis=0)
    expected = emitted * np.array([w.sum() for w in weights])
    if not np.allclose(initial_counts, expected, rtol=1e-12, atol=1e-8):
        raise RuntimeError("Deposition-to-mucus transfer did not preserve the initial dose.")

    transport_cases = {}
    for name, da, speed in [
        ("baseline", 1., 1.), ("no_attachment", 0., 1.), ("faster_clearance", 1., 2.)
    ]:
        p = dict(params, velocity_m_s=params["velocity_m_s"] * speed)
        transport, outflow, dx, dz = transport_operator(p)
        matrix, _ = coupled_operator(transport, outflow, p, biology, da, 0., 0.)
        solution = expm_multiply(matrix, initial, start=0, stop=3600, num=len(times),
                                 endpoint=True, traceA=float(matrix.diagonal().sum()))
        mobile = solution[:, :n].reshape(len(times), nz, nx, len(SEEDS))
        attached = solution[:, n:n+nx]
        retained = mobile.sum(axis=(1, 2)) + attached.sum(axis=1)
        cleared = solution[:, -3]
        error = np.max(np.abs(retained + cleared - initial_counts) / initial_counts)
        if error > 1e-7 or solution[:, :n+nx].min() < -1e-7:
            raise RuntimeError(f"Invalid transport result for {name}: balance error {error:g}")
        flux = np.einsum("tns,n->ts", solution[:, :n], outflow)
        flux += biology["direct_attached_clearance_s_inv"] * attached.sum(axis=1)
        transport_cases[name] = dict(
            mobile_axial=mobile.sum(axis=1).transpose(0, 2, 1),
            attached_axial=attached.transpose(0, 2, 1),
            flux=flux, cleared=cleared, exposure=solution[:, -1],
        )
        print(f"{name}: retained60={retained[-1].mean():.3f} CFU; balance={error:.2g}", flush=True)

    records = [r["record"] for r in fit_rows]
    # time, scenario, growth record, spray seed
    shape = (len(times), len(SCENARIOS), len(records), len(SEEDS))
    mobile_total = np.empty(shape)
    attached_total = np.empty(shape)
    outflow_total = np.empty(shape)
    births_total = np.empty(shape)
    auc_total = np.empty(shape)
    full_maps_mobile, full_maps_attached = [], []
    max_balance_error = 0.
    raw_metrics = []
    for c, scenario in enumerate(SCENARIOS):
        case = transport_cases["baseline" if scenario in ("full", "no_growth") else scenario]
        mobile0 = case["mobile_axial"].sum(axis=2)
        attached0 = case["attached_axial"].sum(axis=2)
        for j, row in enumerate(fit_rows):
            factor, rate = relative_growth(times, row)
            if scenario == "no_growth":
                factor, rate = np.ones_like(times), np.zeros_like(times)
            mobile = mobile0 * factor[:, None]
            attached = attached0 * factor[:, None]
            retained = mobile + attached
            # Integrate the instantaneous outflow flux with its growth factor.
            if scenario == "no_growth":
                out = case["cleared"]
                born = np.zeros_like(out)
                exposure = case["exposure"]
            else:
                # Add the integrated growth-dependent flux correction to the
                # zero-growth cumulative outflow integral state.
                correction = cumulative_simpson(
                    case["flux"] * (factor[:, None]-1), x=times, axis=0, initial=0)
                out = case["cleared"] + correction
                born = cumulative_simpson(rate[:, None] * retained, x=times, axis=0, initial=0)
                exposure = case["exposure"] + cumulative_simpson(
                    (mobile0+attached0) * (factor[:, None]-1),
                    x=times/3600, axis=0, initial=0)
            error = np.max(np.abs(retained + out - initial_counts - born) / initial_counts)
            max_balance_error = max(max_balance_error, float(error))
            if error > 1e-5:
                raise RuntimeError(f"Growth-coupled count balance error: {scenario} {row['record']} {error:g}")
            mobile_total[:, c, j] = mobile
            attached_total[:, c, j] = attached
            outflow_total[:, c, j] = out
            births_total[:, c, j] = born
            auc_total[:, c, j] = exposure
            if scenario == "full":
                full_maps_mobile.append(case["mobile_axial"] * factor[:, None, None])
                full_maps_attached.append(case["attached_axial"] * factor[:, None, None])
            for s, seed in enumerate(SEEDS):
                raw_metrics.append(dict(
                    scenario=scenario, strain=row["strain"], record=row["record"], seed=seed,
                    initial_cfu=initial_counts[s], free_60min_cfu=mobile[-1, s],
                    attached_60min_cfu=attached[-1, s], retained_60min_cfu=retained[-1, s],
                    retained_60min_pct=100*retained[-1, s]/initial_counts[s],
                    outflow_60min_cfu=out[-1, s], growth_added_60min_cfu=born[-1, s],
                    auc_cfu_h=exposure[-1, s],
                    normalized_auc_min=60*exposure[-1, s]/initial_counts[s],
                ))
    # Store spatial fields every 30 seconds; compute flux integrals on the 5-second grid.
    keep = np.arange(0, len(times), 6)
    np.savez_compressed(
        OUT / "coupled_results.npz",
        times_s=times, map_times_s=times[keep], x_mm=(np.arange(nx)+.5)*params["length_m"]/nx*1000,
        scenarios=np.array(SCENARIOS), records=np.array(records),
        strains=np.array([r["strain"] for r in fit_rows]), seeds=np.array(SEEDS),
        source_emitted_cfu_per_strain=np.array(emitted),
        initial_cfu=initial_counts, mobile_cfu=mobile_total,
        attached_cfu=attached_total, outflow_cfu=outflow_total,
        growth_added_cfu=births_total, auc_cfu_h=auc_total,
        full_mobile_axial_cfu=np.stack(full_maps_mobile, axis=1)[keep],
        full_attached_axial_cfu=np.stack(full_maps_attached, axis=1)[keep],
    )
    write_csv(OUT / "metrics_by_record_seed.csv", raw_metrics)
    metrics = [k for k in raw_metrics[0] if k not in ("scenario", "strain", "record", "seed")]
    aggregate = []
    for scenario in SCENARIOS:
        for strain in GROUPS:
            source = [r for r in raw_metrics if r["scenario"] == scenario and r["strain"] == strain]
            result = dict(scenario=scenario, strain=strain)
            for metric in metrics:
                seed_means = [np.mean([r[metric] for r in source if r["seed"] == seed]) for seed in SEEDS]
                result[metric+"_mean"] = float(np.mean(seed_means))
                result[metric+"_seed_min"] = float(min(seed_means))
                result[metric+"_seed_max"] = float(max(seed_means))
            aggregate.append(result)
    write_csv(OUT / "metrics_summary.csv", aggregate)
    summary = dict(
        depth_mm=15, pitch_deg=37.5, yaw_deg=10, cone_full_deg=10,
        seed_count=3, parcels_per_seed=10000, emitted_cfu_per_strain=emitted,
        initial_cfu_mean=float(initial_counts.mean()),
        posterior_origin_mm=start, mucus=params, biology=biology,
        scenarios=SCENARIOS, growth_transfer_eta=1, culture_phase_h=0,
        growth_record_count=len(fit_rows),
        growth_records_per_strain={group: 1 for group in GROUPS},
        growth_data_observations_per_strain={row['strain']: int(row['observations'])
                                           for row in fit_rows if row.get('observations')},
        growth_logistic_limit_threshold_h=1e-6,
        max_relative_coupled_balance_error=max_balance_error,
    )
    (OUT / "results_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print("COUPLED RESULTS COMPLETE", OUT, "balance", max_balance_error, flush=True)


if __name__ == "__main__":
    main()

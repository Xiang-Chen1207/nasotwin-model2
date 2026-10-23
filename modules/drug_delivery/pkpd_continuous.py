"""Connect continuous bacterial supply, transport, and cellular response.

Couple amount-conserving transport to dynamic PD fitted to ELISA endpoints
using the configured assay duration, response times, and exposure scale.
The trajectory output corresponds to Wiki Figure N4c.
"""
from pathlib import Path
import os
import argparse
import json
import csv

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
RUNTIME.mkdir(exist_ok=True)
for key in ("TMPDIR", "TEMP", "TMP", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
    os.environ[key] = str(RUNTIME)

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.integrate import solve_ivp
from transport import TransportParameters, simulate
from fit_endpoint_models import extract_groups, subset
import matplotlib.pyplot as plt

CHANNELS = [("E01", "IL-6", "NAC", "RAW"),
            ("E02", "TNF-α", "NAC", "RAW"),
            ("E03", "IL-6", "SFN", "RAW"),
            ("E04", "IL-6", "SFN", "BV2")]


def fit_dynamic_pd(groups, eid, endpoint, drug, cell, assumption):
    """Fit Emax_infinity/K conditional on T, tau_effect and tau_response.

    Under constant culture-medium concentration c, Ce(s)=c*(1-exp(-s/tau_e)).
    The fraction suppressed at T is Emax * integral of the response kernel
    against Ce/(K+Ce), evaluated at the configured assay duration.
    """
    data = subset(groups, eid, endpoint)
    factor = 1000.0 if data[0]["dose_unit"] == "mM" else 1.0
    dose = np.array([g["dose"] * factor for g in data])
    means = np.array([g["mean"] for g in data])
    baseline = float(means[dose == 0][0])
    mask = dose > 0
    c = dose[mask]
    observed = 1 - means[mask] / baseline
    duration = assumption["reference_assay_duration_h"]
    tau_e, tau_r = assumption["tau_effect_h"], assumption["tau_response_h"]
    nodes, weights = leggauss(96)
    t = (nodes + 1) * duration / 2
    weights = weights * duration / 2
    effective = c[:, None] * (-np.expm1(-t / tau_e))
    kernel = weights * np.exp(-(duration - t) / tau_r) / tau_r

    def endpoint_gain(k):
        return (effective / (k + effective)) @ kernel

    def estimate(log_k):
        response = endpoint_gain(np.exp(log_k))
        emax = float(np.clip(np.dot(response, observed) / np.dot(response, response), 0, 1))
        sse = float(np.sum((emax * response - observed) ** 2))
        return emax, sse

    bounds = (float(np.log(c.max() * 1e-4)), float(np.log(c.max() * 1e4)))
    optimum = minimize_scalar(lambda z: estimate(z)[1], bounds=bounds, method="bounded")
    if not optimum.success:
        raise RuntimeError(optimum.message)
    k = float(np.exp(optimum.x))
    emax, _ = estimate(optimum.x)
    predicted = baseline * (1 - emax * endpoint_gain(k))
    return dict(experiment=eid, endpoint=endpoint, drug=drug, cell=cell,
                Emax_infinity=emax, K_medium_uM=k, baseline_pg_mL=baseline,
                tau_effect_h=tau_e, tau_response_h=tau_r,
                reference_assay_duration_h=duration,
                duration_status="configured_reference_duration",
                calibration_dose_uM=c.tolist(),
                observed_pg_mL=means[mask].tolist(),
                predicted_at_assumed_time_pg_mL=predicted.tolist(),
                conditional_fit_RMSE_pg_mL=float(np.sqrt(np.mean((predicted-means[mask])**2))),
                K_below_lowest_nonzero_dose=bool(k < c.min()),
                Emax_at_upper_bound=bool(emax > 1-1e-7),
                independent_validation=False)


def dynamic_response(time, medium_equivalent_input, calibration):
    tau_e = calibration["tau_effect_h"]
    tau_r = calibration["tau_response_h"]
    k, emax = calibration["K_medium_uM"], calibration["Emax_infinity"]

    def rhs(t, state):
        ce, normalized_readout = state
        driver = np.interp(t, time, medium_equivalent_input)
        target = 1 - emax * ce / (k + ce)
        return [(driver - ce) / tau_e, (target - normalized_readout) / tau_r]

    solution = solve_ivp(rhs, (time[0], time[-1]), [0, 1], t_eval=time,
                         method="DOP853", rtol=1e-8, atol=1e-10, max_step=0.1)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y


def required_production(target_reduction, calibration, gain_h_per_mL,
                        exposure_scale=1.0, reference_biomass=None):
    """Invert the steady-state response to source and productivity (Wiki Eq. 29).

    Return molecular supply in nmol/h and, when biomass is supplied, production
    in nmol per biomass unit per hour. The target is an absolute reduction.
    """
    target = float(target_reduction)
    emax = float(calibration["Emax_infinity"])
    concentration_scale = float(calibration["K_medium_uM"])
    gain = float(gain_h_per_mL)
    exposure = float(exposure_scale)
    if (not np.isfinite([target, emax, concentration_scale, gain, exposure]).all()
            or not 0 < target < emax <= 1
            or min(concentration_scale, gain, exposure) <= 0):
        raise ValueError('Require 0 < target < Emax <= 1 and positive K, gain, and exposure scale')
    rate = concentration_scale / (exposure * gain) * target / (emax - target)
    result = {"required_source_nmol_h": rate}
    if reference_biomass is not None:
        biomass = float(reference_biomass)
        if not np.isfinite(biomass) or biomass <= 0:
            raise ValueError('Reference biomass must be finite and positive')
        result["required_productivity_nmol_per_biomass_h"] = rate / biomass
    return result


def plot_trajectories(out):
    """Plot supply, exposure, and response from the trajectory output files."""
    result = json.loads((out / "conditional_pkpd_results.json").read_text(encoding="utf-8"))
    with (out / "illustrative_trajectories.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    data = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    time = data["time_h"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    fig.suptitle("Continuous-supply PK–PD: reference scenario", fontsize=15)
    for j, drug in enumerate(result["assumed_configuration"]["drug_order"]):
        axes[j, 0].plot(time, data[f"{drug}_source_nmol_h"], color="#228B8D")
        axes[j, 0].set(title=f"{drug} · Production", ylabel="Production (nmol/h)")
        axes[j, 0].set_ylim(bottom=0)
        for suffix, label in (("nose_uM", "Nasal"), ("plasma_uM", "Plasma"),
                              ("brain_free_uM", "Target")):
            axes[j, 1].plot(time, data[f"{drug}_{suffix}"], label=label)
        axes[j, 1].set(title=f"{drug} · Exposure", ylabel="Concentration (μM)")
        axes[j, 1].legend(fontsize=9)
        for cal in result["conditional_PD_calibrations"]:
            if cal["drug"] == drug:
                axes[j, 2].plot(time, data[f"{cal['experiment']}_relative_readout"],
                                label=f"{cal['cell']} · {cal['endpoint']}")
        axes[j, 2].set(title=f"{drug} · Response", ylabel="Normalized response")
        axes[j, 2].set_ylim(0, 1.05)
        axes[j, 2].legend(fontsize=9)
        for ax in axes[j]:
            ax.set_xlabel("Time (h)")
            ax.grid(axis="y", alpha=0.15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "svg"):
        fig.savefig(out / f"illustrative_pkpd.{ext}", dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    # Accept the legacy configuration status; current exports use the reference-scenario status.
    if cfg["status"] not in {"REFERENCE_SCENARIO_PK_PD", "ILLUSTRATION_ONLY_NOT_CALIBRATED_PK"}:
        raise ValueError("Use REFERENCE_SCENARIO_PK_PD for the reference-parameter PK workflow.")
    cfg["status"] = "REFERENCE_SCENARIO_PK_PD"
    if cfg["drug_order"] != ["SFN", "NAC"]:
        raise ValueError("Expected SFN/NAC ordering.")
    end = float(cfg["duration_h"])
    source = cfg["source"]
    knots = np.array(source["activity_time_h"], dtype=float)
    activity = np.array(source["relative_active_biomass"], dtype=float)
    reference_rate = np.array(source["reference_production_nmol_h"], dtype=float)
    if (knots.ndim != 1 or len(knots) < 2 or activity.shape != (len(knots), 2)
            or reference_rate.shape != (2,) or not np.isfinite(activity).all()
            or not np.isfinite(reference_rate).all() or not np.isfinite(knots).all()
            or np.any(np.diff(knots) <= 0) or knots[0] != 0 or knots[-1] != end
            or np.min(activity) < 0 or np.min(reference_rate) < 0 or end <= 0):
        raise ValueError("Supply needs nonnegative SFN/NAC activity profiles over the simulation interval.")
    assumption = cfg["pd_assumptions"]
    for name in ("reference_assay_duration_h", "tau_effect_h", "tau_response_h"):
        if not np.isfinite(assumption[name]) or assumption[name] <= 0:
            raise ValueError(f"Configure a positive {name}.")
    scale = np.array(assumption["brain_free_to_medium_equivalent_scale"], dtype=float)
    if scale.shape != (2,) or not np.isfinite(scale).all() or np.min(scale) <= 0:
        raise ValueError("Configure two positive exposure scaling factors.")

    def supply(t):
        return reference_rate * np.array([np.interp(t, knots, activity[:, j]) for j in range(2)])

    parameters = TransportParameters(**cfg["transport"])
    pk = simulate(None, parameters, mode="finite_supply", start=0, end=end,
                  source=supply, initial_nasal=[0, 0, 0], points=961,
                  rtol=1e-8, atol=1e-10, max_step=0.1)
    # Compare the amount-balance residual with the cumulative input scale.
    balance_error = float(np.max(np.abs(pk["ledger"])))
    amount_scale = max(1.0, float(np.max(pk["amounts"][11:13])))
    if balance_error > 1e-7 * amount_scale:
        raise RuntimeError("Amount balance is inconsistent.")

    groups, _ = extract_groups()
    calibrations = [fit_dynamic_pd(groups, *channel, assumption) for channel in CHANNELS]
    time = pk["time_h"]
    responses = {}
    diagnostics = []
    for cal in calibrations:
        j = cfg["drug_order"].index(cal["drug"])
        driver = pk["brain_free_uM"][j] * scale[j]
        effective, readout = dynamic_response(time, driver, cal)
        responses[cal["experiment"]] = (effective, readout)
        low, high = min(cal["calibration_dose_uM"]), max(cal["calibration_dose_uM"])
        diagnostics.append(dict(
            experiment=cal["experiment"],
            fraction_of_output_times_driver_below_lowest_nonzero_calibration_dose=float(np.mean(driver < low)),
            fraction_of_output_times_driver_above_highest_calibration_dose=float(np.mean(driver > high)),
            note="Reports how the simulated driver overlaps the cellular calibration-dose range. Exposure mapping and response time constants use configured scenario values."))

    out = ROOT / "pkpd_continuous_results"
    out.mkdir(exist_ok=True)
    fields = ["time_h"]
    for drug in cfg["drug_order"]:
        fields.extend([f"{drug}_source_nmol_h", f"{drug}_nose_uM",
                       f"{drug}_plasma_uM", f"{drug}_brain_free_uM"])
    for cal in calibrations:
        fields.extend([cal["experiment"]+"_Ce_uM", cal["experiment"]+"_relative_readout"])
    with (out / "illustrative_trajectories.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, t in enumerate(time):
            row = {"time_h": t}
            for j, drug in enumerate(cfg["drug_order"]):
                row.update({f"{drug}_source_nmol_h": supply(t)[j],
                            f"{drug}_nose_uM": pk["amounts"][j, i] / parameters.V_nose,
                            f"{drug}_plasma_uM": pk["amounts"][5+j, i] / parameters.V_plasma,
                            f"{drug}_brain_free_uM": pk["brain_free_uM"][j, i]})
            for cal in calibrations:
                ce, readout = responses[cal["experiment"]]
                row.update({cal["experiment"]+"_Ce_uM": ce[i],
                            cal["experiment"]+"_relative_readout": readout[i]})
            writer.writerow(row)
    pk_metrics = []
    for j, drug in enumerate(cfg["drug_order"]):
        concentration = pk["brain_free_uM"][j]
        peak_index = int(np.argmax(concentration))
        auc = np.sum(np.diff(time) * (concentration[:-1] + concentration[1:]) / 2)
        pk_metrics.append(dict(drug=drug, status=cfg["status"],
                               simulated_duration_h=end,
                               cumulative_source_nmol=float(pk["amounts"][11+j, -1]),
                               target_AUC_0_T_uM_h=float(auc),
                               target_Cmax_within_window_uM=float(concentration[peak_index]),
                               time_of_window_max_h=float(time[peak_index]),
                               window_max_at_last_sample=bool(peak_index == len(time)-1)))
    result = {
        "status": cfg["status"], "input_mode": "continuous_bacterial_supply_only",
        "assumed_configuration": cfg, "conditional_PD_calibrations": calibrations,
        "exposure_support": diagnostics, "max_amount_balance_error_nmol": balance_error,
        "illustrative_PK_metrics": pk_metrics,
        "interpretation": "Continuous-supply PK–PD simulation under the reference transport and timing configuration. Cellular response channels are calibrated and reported separately; the configured PK parameters define the exposure scenario. Each drug-response channel is integrated independently.",
    }
    (out / "conditional_pkpd_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    plot_trajectories(out)
    print("Completed: continuous source -> parallel-route PK -> effect compartment -> response.")
    print("PK and timing use reference scenario values; cellular response channels are reported separately.")
    for cal in calibrations:
        print(f"{cal['experiment']} {cal['drug']} {cal['cell']} {cal['endpoint']}: "
              f"conditional K={cal['K_medium_uM']:.5g} uM, "
              f"Emax_inf={cal['Emax_infinity']:.5g}, "
              f"endpoint RMSE={cal['conditional_fit_RMSE_pg_mL']:.5g} pg/mL")
    print(f"Amount-balance residual: {balance_error:.3g} nmol; outputs: {out}")


if __name__ == "__main__":
    main()

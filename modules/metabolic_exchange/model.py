"""Endpoint-constrained metabolic dynamics with constant effective activity.

Concentrations use mM and nitrogen pools use mM-N. Time and exposure are
dimensionless. Analytic solutions connect Gln consumption, ammonia and BCAA.
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


EXPERIMENTS = {
    "E05": ("gln_residual", "reported_axis_unit_unresolved"),
    "E06": ("gln_degraded", "mM"),
    "E07": ("bcaa", "mM"),
    "E08": ("ammonium_residual", "mM"),
}


def read_observations(source):
    """Extract individual R-column observations from E05-E08."""
    text = Path(source).read_text(encoding="utf-8")
    rows = []
    for exp, (analyte, unit) in EXPERIMENTS.items():
        match = re.search(
            rf"^### {exp}\b.*?(?=^### E\d+|\Z)", text, re.M | re.S
        )
        if match is None:
            raise ValueError(f"Missing {exp} in source table")
        header = None
        for line in match.group().splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if "R1" in cells and "分组" in cells:
                header = cells
                continue
            if header is None or len(cells) != len(header) or cells[0].startswith("---"):
                continue
            group = cells[header.index("分组")]
            if not (group.startswith("Wild type") or group.startswith("Lp-")):
                continue
            added = 15.0 if "15mM" in group else 5.0 if "5mM" in group else 0.0
            strain = "WT" if group.startswith("Wild type") else (
                "Lp-Q" if group.startswith("Lp-Q") else "Lp-NH3"
            )
            for i, column in enumerate(header):
                if not re.fullmatch(r"R\d+", column) or cells[i] in {"NA", ""}:
                    continue
                rows.append({
                    "experiment": exp, "analyte": analyte, "strain": strain,
                    "added_gln_mM": added, "source_group": group,
                    "subcolumn": column, "value": float(cells[i]), "unit": unit,
                    "evidence": "measured_processed_value",
                    "replicate_type": "unconfirmed",
                })
    return pd.DataFrame(rows)


def endpoint_statistics(obs):
    keys = ["experiment", "analyte", "added_gln_mM", "strain", "unit"]
    stats = obs.groupby(keys, sort=False)["value"].agg(
        n_values="count", mean="mean", sd="std"
    ).reset_index()
    effects = []
    for (exp, added), frame in stats.groupby(["experiment", "added_gln_mM"], sort=False):
        wt = frame[frame.strain == "WT"].iloc[0]
        eng = frame[frame.strain != "WT"].iloc[0]
        effects.append({
            "experiment": exp, "added_gln_mM": added,
            "analyte": wt.analyte, "engineered_strain": eng.strain,
            "wt_mean": wt["mean"], "engineered_mean": eng["mean"],
            "ratio_to_wt": eng["mean"] / wt["mean"],
            "change_percent": 100 * (eng["mean"] / wt["mean"] - 1),
            "difference_engineered_minus_wt": eng["mean"] - wt["mean"],
            "unit": wt.unit, "interpretation": "between_group_endpoint_contrast",
        })
    return stats, pd.DataFrame(effects)


def group_mean(obs, exp, strain):
    return float(obs[(obs.experiment == exp) & (obs.strain == strain)].value.mean())


def fit_gln(obs):
    """Fit D=(G_basal+C_added)*(1-exp(-lambda_s)) to the E06 endpoints.

    Unweighted least squares estimates the shared basal pool and strain-specific
    effective activities from all 10 observations.
    """
    data = obs[obs.experiment == "E06"].copy()
    added = data.added_gln_mM.to_numpy()
    observed = data.value.to_numpy()
    is_q = (data.strain == "Lp-Q").to_numpy()

    def residual(par):
        basal, wt, q = par
        rates = np.where(is_q, q, wt)
        return (basal + added) * (-np.expm1(-rates)) - observed

    fit = least_squares(
        residual, [1.3, 0.13, 0.17],
        bounds=([1e-8, 1e-8, 1e-8], [100.0, 10.0, 10.0]),
        xtol=1e-12, ftol=1e-12, gtol=1e-12,
    )
    if not fit.success:
        raise RuntimeError(f"Gln calibration failed: {fit.message}")
    data["predicted_mM"] = observed + residual(fit.x)
    data["residual_mM"] = observed - data.predicted_mM
    return {
        "gln_basal_mM": float(fit.x[0]),
        "lambda_g_wt": float(fit.x[1]),
        "lambda_g_q": float(fit.x[2]),
        "rmse_mM": float(np.sqrt(np.mean(residual(fit.x) ** 2))),
    }, data


def speciation(reference):
    pka = 0.09018 + 2729.92 / (reference["temperature_C"] + 273.2)
    f_nh3 = 1 / (1 + 10 ** (pka - reference["pH"]))
    mode = reference["ammonia_observation"]
    if mode not in {"NH4", "TAN"}:
        raise ValueError("ammonia_observation must be NH4 or TAN")
    return pka, f_nh3, (1 - f_nh3 if mode == "NH4" else 1.0)


def calibrate_ammonia_bcaa(obs, reference):
    """Calibrate ammonia and BCAA activities to E07/E08 group means.

    The configured theta allocates newly accumulated BCAA-N between ammonia
    and the organic nitrogen pool. Calibration uses the configured initial
    pools and jointly matches the two assays at group level.
    """
    _, _, observation_fraction = speciation(reference)
    a0 = reference["initial_total_ammonia_mM"]
    b0 = reference["initial_bcaa_mM"]
    o0 = reference["initial_other_organic_n_mM_N"]
    theta = reference["bcaa_new_n_fraction_from_ammonia"]
    if not 0 <= theta <= 1 or min(a0, o0) <= 0 or b0 < 0:
        raise ValueError("Invalid initial pools or nitrogen-source fraction")
    result = {}
    for strain, suffix in [("WT", "wt"), ("Lp-NH3", "n")]:
        a_end = group_mean(obs, "E08", strain) / observation_fraction
        b_end = group_mean(obs, "E07", strain)
        if not 0 < a_end < a0 or not b0 <= b_end:
            raise ValueError("Assumed starting pools incompatible with removal/accumulation model")
        newly_b = b_end - b0
        yield_n = theta * newly_b / (a0 - a_end)
        organic_used = (1 - theta) * newly_b
        if not 0 <= yield_n <= 1 or not 0 <= organic_used < o0:
            raise ValueError("Assumed nitrogen allocation exceeds available substrate")
        result[f"lambda_a_{suffix}"] = float(np.log(a0 / a_end))
        result[f"yield_b_{suffix}"] = float(yield_n)
        result[f"lambda_o_{suffix}"] = float(-np.log1p(-organic_used / o0))
    return result


def effective_rates(gln, nitrogen, reference, q_fraction, exposure=1.0, wild_type=False):
    """Combine strain activities using mixture fractions and relative exposure."""
    if not 0 <= q_fraction <= 1 or exposure < 0:
        raise ValueError("q_fraction must be in [0,1]; exposure must be nonnegative")
    r = q_fraction
    q_transfer = reference["q_activity_transfer"]
    n_transfer = reference["n_activity_transfer"]
    gw, gq = gln["lambda_g_wt"], gln["lambda_g_q"]
    aw, an = nitrogen["lambda_a_wt"], nitrogen["lambda_a_n"]
    ow, on = nitrogen["lambda_o_wt"], nitrogen["lambda_o_n"]
    yw, yn = nitrogen["yield_b_wt"], nitrogen["yield_b_n"]
    if wild_type:
        return exposure*q_transfer*gw, exposure*n_transfer*aw, exposure*n_transfer*ow, yw
    a_mix = r * aw + (1-r) * an
    yield_mix = (r*aw*yw + (1-r)*an*yn) / a_mix
    return (
        exposure*q_transfer*(r*gq + (1-r)*gw),
        exposure*n_transfer*a_mix,
        exposure*n_transfer*(r*ow + (1-r)*on),
        yield_mix,
    )


def trajectory(tau, gln, nitrogen, reference, q_fraction=0.5, exposure=1.0, wild_type=False):
    """Exact solution of five-pool linear N balance, with zero external flux.

    States: Gln (two N), TAN, BCAA (one N), other available organic N,
    retained/unresolved N (including glutamate-equivalent N and biomass).
    Carbon and energy availability are held constant in this nitrogen model.
    """
    tau = np.asarray(tau, dtype=float)
    kg, ka, ko, yb = effective_rates(gln, nitrogen, reference, q_fraction, exposure, wild_type)
    g0 = gln["gln_basal_mM"] + reference["gln_supplement_mM"]
    a0 = reference["initial_total_ammonia_mM"]
    b0 = reference["initial_bcaa_mM"]
    o0 = reference["initial_other_organic_n_mM_N"]
    nu = reference["ammonia_release_per_gln"]
    if not 0 <= nu <= 1:
        raise ValueError("ammonia_release_per_gln must be in [0,1]")
    g = g0 * np.exp(-kg*tau)
    if abs(ka-kg) < 1e-10:
        released_remaining = nu*kg*g0*tau*np.exp(-kg*tau)
    else:
        released_remaining = nu*kg*g0*(np.exp(-kg*tau)-np.exp(-ka*tau))/(ka-kg)
    a = a0*np.exp(-ka*tau) + released_remaining
    used_a = a0 + nu*(g0-g) - a
    organic = o0*np.exp(-ko*tau)
    b = b0 + yb*used_a + (o0-organic)
    retained = (2-nu)*(g0-g) + (1-yb)*used_a
    _, f_nh3, obs_fraction = speciation(reference)
    result = pd.DataFrame({
        "tau": tau, "Gln_mM": g, "TAN_mM": a, "BCAA_mM": b,
        "other_organic_N_mM_N": organic, "retained_N_mM_N": retained,
        "ammonia_assay_equivalent_mM": a*obs_fraction,
        "free_NH3_mM": a*f_nh3,
        "total_tracked_N_mM_N": 2*g+a+b+organic+retained,
    })
    if reference.get("illustrative_endpoint_hours") is not None:
        result["illustrative_time_h"] = tau * reference["illustrative_endpoint_hours"]
    return result


def ammonia_peak(gln, nitrogen, reference, r, dose):
    """Analytic maximum on tau in [0,1], including an interior stationary point."""
    kg, ka, _, _ = effective_rates(gln, nitrogen, reference, r, dose)
    nu = reference["ammonia_release_per_gln"]
    g0 = gln["gln_basal_mM"] + reference["gln_supplement_mM"]
    a0 = reference["initial_total_ammonia_mM"]
    candidates = [0.0, 1.0]
    if kg > 0 and ka > 0 and nu > 0:
        if abs(ka-kg) < 1e-10:
            t = (nu*g0-a0)/(nu*kg*g0)
            if 0 < t < 1:
                candidates.append(t)
        else:
            c = nu*kg*g0/(ka-kg)
            ratio = -ka*(a0-c)/(kg*c)
            if ratio > 0:
                t = np.log(ratio)/(ka-kg)
                if 0 < t < 1:
                    candidates.append(t)
    curve = trajectory(candidates, gln, nitrogen, reference, r, dose)
    return float(curve.TAN_mM.max())


def design_scan(gln, nitrogen, reference, design):
    rows = []
    g0 = gln["gln_basal_mM"] + reference["gln_supplement_mM"]
    a0 = reference["initial_total_ammonia_mM"]
    fractions = np.linspace(0, 1, design["q_fraction_points"])
    exposures = np.linspace(design["relative_exposure_min"], design["relative_exposure_max"],
                            design["relative_exposure_points"])
    for dose in exposures:
        for r in fractions:
            end = trajectory([1.0], gln, nitrogen, reference, float(r), float(dose)).iloc[0]
            peak = ammonia_peak(gln, nitrogen, reference, float(r), float(dose))
            removal = 1-end.Gln_mM/g0
            feasible = (
                removal >= design["target_gln_removal_fraction"]
                and peak/a0 <= design["max_peak_ammonia_to_initial"] + 1e-12
                and end.TAN_mM/a0 <= design["max_endpoint_ammonia_to_initial"] + 1e-12
            )
            rows.append({
                "q_fraction": float(r), "relative_exposure": float(dose),
                "gln_removal_fraction": removal, "endpoint_TAN_mM": end.TAN_mM,
                "peak_TAN_mM": peak, "peak_TAN_to_initial": peak/a0,
                "endpoint_BCAA_mM": end.BCAA_mM, "feasible_assumed_targets": feasible,
                "evidence": "conditional_simulation",
            })
    return pd.DataFrame(rows)


def choose_candidate(scan):
    """Select minimum feasible exposure, then maximize Gln removal."""
    eligible = scan[scan.feasible_assumed_targets]
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["relative_exposure", "gln_removal_fraction"], ascending=[True, False]
    ).iloc[0].to_dict()


def scenario_sensitivity(obs, gln, reference, design):
    variants = [
        ("Reference", {}),
        ("No Gln-derived ammonia", {"ammonia_release_per_gln": 0.0}),
        ("Half ammonia release", {"ammonia_release_per_gln": 0.5}),
        ("No ammonia-to-BCAA coupling", {"bcaa_new_n_fraction_from_ammonia": 0.0}),
        ("All new BCAA N from ammonia", {"bcaa_new_n_fraction_from_ammonia": 1.0}),
        ("Initial TAN 3.5 mM", {"initial_total_ammonia_mM": 3.5}),
        ("Initial TAN 6 mM", {"initial_total_ammonia_mM": 6.0}),
        ("Initial BCAA 0 mM", {"initial_bcaa_mM": 0.0}),
        ("Initial BCAA 0.4 mM", {"initial_bcaa_mM": 0.4}),
        ("Gln activity transfer x0.5", {"q_activity_transfer": 0.5}),
        ("Gln activity transfer x2", {"q_activity_transfer": 2.0}),
        ("N activity transfer x0.5", {"n_activity_transfer": 0.5}),
        ("N activity transfer x2", {"n_activity_transfer": 2.0}),
    ]
    rows = []
    for name, updates in variants:
        ref = dict(reference, **updates)
        nitrogen = calibrate_ammonia_bcaa(obs, ref)
        end = trajectory([1.0], gln, nitrogen, ref).iloc[0]
        peak = ammonia_peak(gln, nitrogen, ref, 0.5, 1.0)
        candidate = choose_candidate(design_scan(gln, nitrogen, ref, design))
        rows.append({
            "scenario": name, "Gln_mM_at_1to1": end.Gln_mM,
            "TAN_mM_at_1to1": end.TAN_mM, "BCAA_mM_at_1to1": end.BCAA_mM,
            "peak_TAN_mM_at_1to1": peak,
            "candidate_q_fraction": np.nan if candidate is None else candidate["q_fraction"],
            "candidate_exposure": np.nan if candidate is None else candidate["relative_exposure"],
            "evidence": "one_factor_assumption_sensitivity_not_confidence_interval",
        })
    return pd.DataFrame(rows)

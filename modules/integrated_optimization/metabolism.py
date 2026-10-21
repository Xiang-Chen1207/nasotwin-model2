"""Map retained bacterial histories to five nitrogen-pool dynamics.

The reference reaction compartment contains Gln, ammonia, BCAA, other organic
nitrogen, and retained nitrogen. Endpoint loss parameters are scaled by
retained CFU / exposure_ref, where exposure_ref has units CFU*h.
Lp-Q and Lp-NH3 drive these pools through their strain-specific and WT rates.
NAC and SFN production is evaluated by the separate PK-PD branch.
"""

import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def load_parameters(root: Path) -> dict:
    """Read the metabolic module export selected by config.json inputs."""
    root = Path(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    source = root / config["inputs"]["metabolic_parameters"]
    return json.loads(source.read_text(encoding="utf-8"))


def _initial(parameters):
    gln = parameters["gln"]
    ref = parameters["reference"]
    return np.array(
        [
            gln["gln_basal_mM"] + ref["gln_supplement_mM"],
            ref["initial_total_ammonia_mM"],
            ref["initial_bcaa_mM"],
            ref["initial_other_organic_n_mM_N"],
            0.0,
        ],
        dtype=float,
    )


def _validate(t, n_q, n_n, exposure_ref, eta_q, eta_n, nu):
    t = np.asarray(t, dtype=float)
    n_q = np.asarray(n_q, dtype=float)
    n_n = np.asarray(n_n, dtype=float)
    if (
        t.ndim != 1
        or t.size < 2
        or not np.all(np.isfinite(t))
        or t[0] != 0.0
        or t[-1] > 1.0 + 1e-12
        or np.any(np.diff(t) <= 0.0)
    ):
        raise ValueError("Metabolic times must increase from 0 within the first hour.")
    if (
        n_q.shape != n_n.shape
        or n_q.shape[-1:] != t.shape
        or not np.all(np.isfinite(n_q))
        or not np.all(np.isfinite(n_n))
        or np.any(n_q < 0.0)
        or np.any(n_n < 0.0)
    ):
        raise ValueError("Both nonnegative CFU inputs must have time as their last axis.")
    if not (
        np.isfinite(exposure_ref)
        and exposure_ref > 0.0
        and np.isfinite(eta_q)
        and eta_q >= 0.0
        and np.isfinite(eta_n)
        and eta_n >= 0.0
        and np.isfinite(nu)
        and 0.0 <= nu <= 1.0
    ):
        raise ValueError("Use positive CFU*h reference, nonnegative activity, and 0<=nu<=1.")
    return t, n_q, n_n


def _rates(n_q, n_n, parameters, exposure_ref, eta_q, eta_n):
    g = parameters["gln"]
    n = parameters["nitrogen"]
    ref = parameters["reference"]
    q_scale = eta_q * ref["q_activity_transfer"] / exposure_ref
    n_scale = eta_n * ref["n_activity_transfer"] / exposure_ref
    kg = q_scale * (g["lambda_g_q"] * n_q + g["lambda_g_wt"] * n_n)
    ka = n_scale * (n["lambda_a_wt"] * n_q + n["lambda_a_n"] * n_n)
    ko = n_scale * (n["lambda_o_wt"] * n_q + n["lambda_o_n"] * n_n)
    kab = n_scale * (
        n["lambda_a_wt"] * n["yield_b_wt"] * n_q
        + n["lambda_a_n"] * n["yield_b_n"] * n_n
    )
    return kg, ka, ko, kab


def _feasible(removal, peak, end, parameters):
    design = parameters["design"]
    a0 = parameters["reference"]["initial_total_ammonia_mM"]
    return (
        (removal >= design["target_gln_removal_fraction"])
        & (peak <= a0 * design["max_peak_ammonia_to_initial"] + 1e-10)
        & (end <= a0 * design["max_endpoint_ammonia_to_initial"] + 1e-10)
    )


def simulate(
    t: np.ndarray,
    n_q: np.ndarray,
    n_n: np.ndarray,
    parameters: dict,
    exposure_ref: float,
    eta_q: float = 1.0,
    eta_n: float = 1.0,
    nu: float = 1.0,
    *,
    metrics_only: bool = False,
) -> dict:
    """Integrate one candidate with piecewise-linear CFU histories.

    ``eta_q`` transfers the Gln assay activity and ``eta_n`` transfers the
    ammonia/BCAA assay activity. These pathway factors multiply the
    corresponding rates for both strains. The WT Gln rate is assigned to
    Lp-NH3 and the WT ammonia/BCAA rates to Lp-Q through linear mixing.

    A DOP853 solve retains the full history. Ammonia maxima include endpoints,
    the adaptive solver grid, a fine auxiliary grid, and derivative roots
    refined with Brent's method. Rates follow the time-varying CFU histories.
    """
    t, n_q, n_n = _validate(t, n_q, n_n, exposure_ref, eta_q, eta_n, nu)
    if n_q.ndim != 1:
        raise ValueError("simulate accepts one history; use scan_metrics for batches.")
    initial = _initial(parameters)

    def rates_at(time):
        return _rates(
            np.interp(time, t, n_q),
            np.interp(time, t, n_n),
            parameters,
            exposure_ref,
            eta_q,
            eta_n,
        )

    def derivative(time, state):
        kg, ka, ko, kab = rates_at(time)
        g, a, _, organic, _ = state
        loss_g = kg * g
        loss_a = ka * a
        gain_b_from_a = kab * a
        loss_o = ko * organic
        return np.array(
            [
                -loss_g,
                nu * loss_g - loss_a,
                gain_b_from_a + loss_o,
                -loss_o,
                (2.0 - nu) * loss_g + loss_a - gain_b_from_a,
            ]
        )

    solution = solve_ivp(
        derivative,
        (0.0, float(t[-1])),
        initial,
        method="DOP853",
        dense_output=True,
        rtol=2e-9,
        atol=2e-11,
        max_step=0.02,
    )
    if not solution.success:
        raise RuntimeError(f"Metabolic integration failed: {solution.message}")
    states = solution.sol(t)
    peak_grid = np.unique(
        np.concatenate(
            [
                t,
                solution.t,
                np.linspace(0.0, t[-1], max(2, int(np.ceil(t[-1] / 0.0025)) + 1)),
            ]
        )
    )
    peak_states = solution.sol(peak_grid)
    kg, ka, _, _ = rates_at(peak_grid)
    slopes = nu * kg * peak_states[0] - ka * peak_states[1]
    peaks = list(peak_states[1])
    peak_times = list(peak_grid)

    def ammonia_slope(time):
        kg_here, ka_here, _, _ = rates_at(time)
        state = solution.sol(time)
        return nu * kg_here * state[0] - ka_here * state[1]

    brackets = np.flatnonzero((slopes[:-1] > 0.0) & (slopes[1:] < 0.0))
    for index in brackets:
        stationary = brentq(
            ammonia_slope,
            peak_grid[index],
            peak_grid[index + 1],
            xtol=1e-12,
        )
        peaks.append(float(solution.sol(stationary)[1]))
        peak_times.append(float(stationary))
    peak_index = int(np.argmax(peaks))
    peak = float(peaks[peak_index])
    removal = float(1.0 - states[0, -1] / initial[0])
    total_n = 2.0 * states[0] + states[1:].sum(axis=0)
    initial_total_n = 2.0 * initial[0] + initial[1:].sum()
    result = {
        "gln_removal": removal,
        "ammonia_peak": peak,
        "ammonia_peak_time_h": float(peak_times[peak_index]),
        "ammonia_end": float(states[1, -1]),
        "bcaa_end": float(states[2, -1]),
        "nitrogen_error": float(np.max(np.abs(total_n - initial_total_n))),
        "feasible": bool(_feasible(removal, peak, states[1, -1], parameters)),
    }
    if not metrics_only:
        result.update(
            time_h=t.copy(),
            G=states[0],
            A=states[1],
            B=states[2],
            O=states[3],
            R=states[4],
        )
    return result


def _constant_ammonia(g, a, kg, ka, duration, nu):
    """Stable exact TAN update under constant rates, including kg == ka."""
    duration = np.asarray(duration)
    d = np.abs(ka - kg)
    kernel = np.empty_like(np.asarray(g + a + duration), dtype=float)
    np.divide(
        -np.expm1(-d * duration),
        d,
        out=kernel,
        where=d > 1e-14,
    )
    kernel = np.where(d > 1e-14, kernel, duration)
    return a * np.exp(-ka * duration) + (
        nu * kg * g * np.exp(-np.minimum(kg, ka) * duration) * kernel
    )


def scan_metrics(
    t: np.ndarray,
    unit_q: np.ndarray,
    unit_n: np.ndarray,
    doses_q: np.ndarray,
    doses_n: np.ndarray,
    parameters: dict,
    exposure_ref: float,
    eta_q: float = 1.0,
    eta_n: float = 1.0,
    nu: float = 1.0,
    *,
    max_step_h: float = 1.0 / 600.0,
) -> dict:
    """Fast batch screen from unit-retention histories and emitted CFU arrays.

    ``unit_q`` and ``unit_n`` are 1-D retained CFU per emitted CFU over ``t``;
    ``doses_q`` and ``doses_n`` are equal-length candidate emitted-CFU arrays.
    Each input interval is subdivided to at most ``max_step_h``; linearly
    interpolated midpoint rates are held constant within each subinterval.
    All five pools and interior ammonia maxima then have analytic updates.
    This second-order time discretization retains changing strain ratios and
    ammonia peaks. Use ``simulate`` to confirm/report selected candidates.
    """
    unit_q = np.asarray(unit_q, dtype=float)
    unit_n = np.asarray(unit_n, dtype=float)
    doses_q = np.asarray(doses_q, dtype=float)
    doses_n = np.asarray(doses_n, dtype=float)
    if (
        unit_q.ndim != 1
        or unit_n.ndim != 1
        or doses_q.ndim != 1
        or doses_q.shape != doses_n.shape
        or np.any(doses_q < 0.0)
        or np.any(doses_n < 0.0)
    ):
        raise ValueError("Use 1-D unit-retention histories and equal-length CFU arrays.")
    n_q = doses_q[:, None] * unit_q[None, :]
    n_n = doses_n[:, None] * unit_n[None, :]
    t, n_q, n_n = _validate(t, n_q, n_n, exposure_ref, eta_q, eta_n, nu)
    if not np.isfinite(max_step_h) or max_step_h <= 0.0:
        raise ValueError("max_step_h must be finite and positive.")
    initial = _initial(parameters)
    size = n_q.shape[0]
    g, a, b, organic, retained = [
        np.full(size, value, dtype=float) for value in initial
    ]
    peak = a.copy()
    peak_time = np.zeros(size)
    nitrogen_error = np.zeros(size)
    total_initial = 2.0 * initial[0] + initial[1:].sum()

    for interval in range(t.size - 1):
        interval_length = t[interval + 1] - t[interval]
        count = max(1, int(np.ceil(interval_length / max_step_h)))
        dt = interval_length / count
        for substep in range(count):
            fraction = (substep + 0.5) / count
            q = n_q[:, interval] + fraction * (
                n_q[:, interval + 1] - n_q[:, interval]
            )
            n = n_n[:, interval] + fraction * (
                n_n[:, interval + 1] - n_n[:, interval]
            )
            kg, ka, ko, kab = _rates(q, n, parameters, exposure_ref, eta_q, eta_n)
            start = t[interval] + substep * dt
            new_g = g * np.exp(-kg * dt)
            new_a = _constant_ammonia(g, a, kg, ka, dt, nu)

            # A' has at most one interior zero under constant rates. Its
            # analytic expression uses log1p to remain stable when the two
            # rates approach equality.
            v0 = nu * kg * g
            slope_start = v0 - ka * a
            slope_end = nu * kg * new_g - ka * new_a
            stationary_mask = (slope_start > 0.0) & (slope_end < 0.0)
            stationary = np.zeros(size)
            if np.any(stationary_mask):
                index = np.flatnonzero(stationary_mask)
                delta = ka[index] - kg[index]
                source = v0[index]
                scaled = delta * (source - ka[index] * a[index]) / (
                    source * kg[index]
                )
                stable = np.abs(delta) > 1e-12
                roots = np.empty_like(delta)
                roots[stable] = np.log1p(scaled[stable]) / delta[stable]
                roots[~stable] = (
                    (source[~stable] - ka[index][~stable] * a[index][~stable])
                    / (source[~stable] * kg[index][~stable])
                )
                stationary[index] = np.clip(roots, 0.0, dt)
                interior_a = _constant_ammonia(
                    g[index],
                    a[index],
                    kg[index],
                    ka[index],
                    stationary[index],
                    nu,
                )
                improve = interior_a > peak[index]
                changed = index[improve]
                peak[changed] = interior_a[improve]
                peak_time[changed] = start + stationary[changed]
            improve = new_a > peak
            peak[improve] = new_a[improve]
            peak_time[improve] = start + dt

            used_a = a + nu * (g - new_g) - new_a
            yield_mix = np.divide(kab, ka, out=np.zeros_like(ka), where=ka > 0.0)
            new_organic = organic * np.exp(-ko * dt)
            b += yield_mix * used_a + organic - new_organic
            retained += (2.0 - nu) * (g - new_g) + (1.0 - yield_mix) * used_a
            g, a, organic = new_g, new_a, new_organic
            nitrogen_error = np.maximum(
                nitrogen_error,
                np.abs(2.0 * g + a + b + organic + retained - total_initial),
            )

    removal = 1.0 - g / initial[0]
    return {
        "gln_removal": removal,
        "ammonia_peak": peak,
        "ammonia_peak_time_h": peak_time,
        "ammonia_end": a,
        "bcaa_end": b,
        "nitrogen_error": nitrogen_error,
        "feasible": _feasible(removal, peak, a, parameters),
    }

"""Compute independent NAC and SFN dynamics for a one-hour bacterial source.

The input parameters specify transport and four PD calibrations. Production
amplitude q * n_emitted has units nmol/h; the retention kernel h(t) gives the
source R(t) = q * n_emitted * h(t). Production ends at 1 h, while drug and
response dynamics continue to the configured evaluation time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


SOURCE_END_H = 1.0


def load_parameters(root: Path) -> dict:
    """Read the drug-delivery export selected by config.json inputs."""
    root = Path(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    source = root / config["inputs"]["pkpd_parameters"]
    return json.loads(source.read_text(encoding="utf-8"))


def prepare_unit_response(
    time_h: np.ndarray,
    kernel_t: np.ndarray,
    kernel: np.ndarray,
    drug: str,
    parameters: dict,
    exposure_factor: float = 1.0,
) -> dict:
    """Solve linear PK/effect compartments once for unit production amplitude.

    exposure_factor multiplies the target-to-medium-equivalent scale kappa.
    Drug amounts follow the transport coefficients in the input parameters.
    Eight-point Gauss-Legendre integration on intervals no longer than 0.1 h
    supplies the nonlinear response convolution used by both forward and
    inverse calculations.
    """
    time_h = np.asarray(time_h, dtype=float)
    kernel_t = np.asarray(kernel_t, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if (
        time_h.ndim != 1 or len(time_h) < 2 or time_h[0] != 0
        or not np.isfinite(time_h).all() or np.any(np.diff(time_h) <= 0)
    ):
        raise ValueError("time_h must increase from zero and contain at least two times.")
    if (
        kernel_t.ndim != 1 or len(kernel_t) < 2 or kernel.shape != kernel_t.shape
        or not np.isfinite(kernel_t).all() or not np.isfinite(kernel).all()
        or np.any(np.diff(kernel_t) <= 0) or kernel_t[0] != 0
        or not np.isclose(kernel_t[-1], SOURCE_END_H)
        or np.min(kernel) < 0
    ):
        raise ValueError("Retention kernel must be nonnegative and cover exactly 0–1 h.")
    if not np.isfinite(exposure_factor) or exposure_factor <= 0:
        raise ValueError("exposure_factor must be finite and positive.")
    cfg = parameters["assumed_configuration"]
    transport = cfg["transport"]
    if any(transport[key] != 0 for key in ("kf", "kr", "k_conjugate_loss")):
        raise ValueError("Independent linear drug branches require zero conjugate chemistry.")
    j = cfg["drug_order"].index(drug)
    calibrations = {
        cal["experiment"]: cal for cal in parameters["conditional_PD_calibrations"]
        if cal["drug"] == drug
    }
    if not calibrations:
        raise ValueError(f"No PD channels for {drug}.")
    scale = (
        cfg["pd_assumptions"]["brain_free_to_medium_equivalent_scale"][j]
        * exposure_factor
    )
    vn, vp, vb = (transport[key] for key in ("V_nose", "V_plasma", "V_brain"))
    ps_ep, fd, fs, tau_d, ps_pb, kpb, clp, clb, kloss = (
        transport[key][j] for key in (
            "PS_ep", "f_direct", "f_systemic", "tau_direct", "PS_pb",
            "K_pb", "CL_plasma", "CL_brain", "k_nasal_loss",
        )
    )
    tau_effect = np.array([cal["tau_effect_h"] for cal in calibrations.values()])

    def rhs(t, y, source_active):
        nose, transit, plasma, target = y[:4]
        source = np.interp(t, kernel_t, kernel) if source_active else 0.0
        uptake = ps_ep * nose / vn
        direct = transit / tau_d
        cp, cb = plasma / vp, target / vb
        exchange = ps_pb * (cp - cb / kpb)
        deriv = np.empty_like(y)
        deriv[:4] = (
            source - uptake - kloss * nose,
            fd * uptake - direct,
            fs * uptake - exchange - clp * cp,
            direct + exchange - clb * cb,
        )
        deriv[4:] = (scale * cb - y[4:]) / tau_effect
        return deriv

    end = time_h[-1]
    # Include every retained-biomass knot and the exact source discontinuity.
    breaks = np.unique(np.r_[0, kernel_t[(kernel_t > 0) & (kernel_t < end)], end])
    state = np.zeros(4 + len(calibrations))
    pieces = []
    for start, stop in zip(breaks[:-1], breaks[1:]):
        active = start < SOURCE_END_H
        solution = solve_ivp(
            lambda t, y: rhs(t, y, active), (start, stop), state,
            method="DOP853", rtol=1e-9, atol=1e-12, max_step=0.1,
            dense_output=True,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        pieces.append((start, stop, solution.sol))
        state = solution.y[:, -1]

    def evaluate(query):
        query = np.asarray(query, dtype=float)
        result = np.empty((4 + len(calibrations), query.size))
        for start, stop, dense in pieces:
            mask = (query >= start) & (query < stop if stop < end else query <= stop)
            if np.any(mask):
                result[:, mask] = dense(query[mask])
        return result

    # A shared refined grid permits exact exponential decay between output
    # samples and accurate quadrature even when the requested output is sparse.
    integration_grid = np.unique(np.r_[
        time_h, breaks, np.linspace(0, end, max(2, int(np.ceil(end / 0.1)) + 1))
    ])
    left, right = integration_grid[:-1], integration_grid[1:]
    nodes, weights = leggauss(8)
    quad_times = (
        (left[:, None] + right[:, None]) / 2
        + (right - left)[:, None] * nodes[None, :] / 2
    )
    quad_widths = (right - left)[:, None] * weights[None, :] / 2
    quad_states = evaluate(quad_times.ravel())
    states = evaluate(time_h)
    channels = {}
    for row, (key, cal) in enumerate(calibrations.items()):
        tau_r = cal["tau_response_h"]
        ce_quad = quad_states[4 + row].reshape(quad_times.shape)
        local_weights = (
            quad_widths * np.exp(-(right[:, None] - quad_times) / tau_r) / tau_r
        )
        endpoint_weights = (
            quad_widths * np.exp(-(end - quad_times) / tau_r) / tau_r
        )
        channels[key] = {
            "calibration": cal,
            "unit_Ce_uM": states[4 + row],
            "unit_quadrature_Ce_uM": ce_quad,
            "local_response_weights": local_weights,
            "endpoint_response_weights": endpoint_weights,
            "interval_decay": np.exp(-(right - left) / tau_r),
        }
    return {
        "drug": drug,
        "time_h": time_h,
        "source_nmol_h": np.where(
            time_h <= SOURCE_END_H, np.interp(time_h, kernel_t, kernel), 0.0
        ),
        "nose_uM": states[0] / vn,
        "plasma_uM": states[2] / vp,
        "target_uM": states[3] / vb,
        "integration_grid": integration_grid,
        "output_indices": np.searchsorted(integration_grid, time_h),
        "exposure_factor": float(exposure_factor),
        "target_to_medium_equivalent_scale": float(scale),
        "channels": channels,
    }


def _endpoint_effect(amplitude: float, channel: dict) -> float:
    cal = channel["calibration"]
    ce = amplitude * channel["unit_quadrature_Ce_uM"]
    suppression = cal["Emax_infinity"] * ce / (cal["K_medium_uM"] + ce)
    return float(np.sum(channel["endpoint_response_weights"] * suppression))


def simulate(
    time_h: np.ndarray,
    kernel_t: np.ndarray,
    kernel: np.ndarray,
    amplitude: float,
    drug: str,
    parameters: dict,
    exposure_factor: float = 1.0,
    *,
    unit_response: dict | None = None,
) -> dict:
    """Return full dynamics; all channels remain independent."""
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("Production amplitude must be finite and nonnegative.")
    unit = unit_response
    if unit is None:
        unit = prepare_unit_response(
            time_h, kernel_t, kernel, drug, parameters, exposure_factor
        )
    result = {"time_h": unit["time_h"], "channels": {}}
    for key in ("source_nmol_h", "nose_uM", "plasma_uM", "target_uM"):
        result[key] = amplitude * unit[key]
    result["amplitude_q_times_emitted_CFU_nmol_h"] = float(amplitude)
    result["exposure_factor"] = unit["exposure_factor"]
    for key, channel in unit["channels"].items():
        cal = channel["calibration"]
        ce_quad = amplitude * channel["unit_quadrature_Ce_uM"]
        suppression = cal["Emax_infinity"] * ce_quad / (cal["K_medium_uM"] + ce_quad)
        increments = np.sum(channel["local_response_weights"] * suppression, axis=1)
        effect_grid = np.zeros(len(unit["integration_grid"]))
        for i, (decay, increment) in enumerate(zip(channel["interval_decay"], increments)):
            effect_grid[i + 1] = decay * effect_grid[i] + increment
        effect = effect_grid[unit["output_indices"]]
        result["channels"][key] = {
            "Ce_uM": amplitude * channel["unit_Ce_uM"],
            "relative_readout": 1 - effect,
            "effect": effect,
        }
    return result


def required_amplitude(
    time_h: np.ndarray,
    kernel_t: np.ndarray,
    kernel: np.ndarray,
    drug: str,
    parameters: dict,
    targets: dict[str, float],
    exposure_factor: float = 1.0,
    *,
    unit_response: dict | None = None,
) -> dict:
    """Find the minimum q*n_emitted meeting all requested E(T) targets.

    The PK, response timing, and exposure parameters determine the required
    amplitude. A None amplitude denotes a target above the channel limit or
    outside the finite numerical search range.
    """
    unit = unit_response
    if unit is None:
        unit = prepare_unit_response(
            time_h, kernel_t, kernel, drug, parameters, exposure_factor
        )
    unknown = set(targets) - set(unit["channels"])
    if unknown:
        raise ValueError(f"Channels {sorted(unknown)} do not belong to {drug}.")
    per_channel = {}
    feasible = True
    amplitudes = []
    for key, target in targets.items():
        target = float(target)
        if not np.isfinite(target) or target < 0:
            raise ValueError("Effect targets must be finite and nonnegative.")
        channel = unit["channels"][key]
        cal = channel["calibration"]
        ce_quad = channel["unit_quadrature_Ce_uM"]
        ceiling = float(cal["Emax_infinity"] * np.sum(
            channel["endpoint_response_weights"] * (ce_quad > 0)
        ))
        entry = {
            "drug": drug,
            "cell": cal["cell"],
            "endpoint": cal["endpoint"],
            "target_effect": target,
            "dynamic_effect_ceiling": ceiling,
            "amplitude": None,
            "feasible": False,
        }
        if target == 0:
            amplitude = 0.0
        elif target >= ceiling:
            entry["reason"] = "target_at_or_above_dynamic_effect_ceiling"
            per_channel[key] = entry
            feasible = False
            continue
        else:
            high = cal["K_medium_uM"] / float(np.max(ce_quad))
            for _ in range(80):
                if _endpoint_effect(high, channel) >= target:
                    break
                high *= 10
            else:
                entry["reason"] = "finite_amplitude_bracket_not_found"
                per_channel[key] = entry
                feasible = False
                continue
            # Solve a dimensionless bracket to retain relative precision for
            # production amplitudes spanning many orders of magnitude.
            fraction = brentq(
                lambda x: _endpoint_effect(x * high, channel) - target,
                0, 1, xtol=1e-13, rtol=1e-12,
            )
            amplitude = float(fraction * high)
        entry.update(
            amplitude=amplitude, feasible=True,
            effect_at_own_requirement=_endpoint_effect(amplitude, channel),
        )
        per_channel[key] = entry
        amplitudes.append(amplitude)
    combined = max(amplitudes, default=0.0) if feasible else None
    if feasible:
        for key, entry in per_channel.items():
            entry["effect_at_joint_drug_requirement"] = _endpoint_effect(
                combined, unit["channels"][key]
            )
    return {
        "drug": drug,
        "amplitude": combined,
        "amplitude_unit": "nmol/h = q[nmol/(CFU h)] * emitted_CFU",
        "feasible": feasible,
        "per_channel": per_channel,
        "endpoint_time_h": float(unit["time_h"][-1]),
        "source_end_h": SOURCE_END_H,
        "exposure_factor": unit["exposure_factor"],
        "interpretation": (
            "Model-derived production requirement under the reference transport and timing configuration; "
            "dose conversion uses configured productivity, target-normalized by default and replaceable with measured values. Response channels are evaluated separately."
        ),
    }

"""Convert nasal-model coupling output to first-hour per-emitted-CFU kernels.

Each strain has one fitted growth trajectory, coupled to each spray seed.
Average mobile and attached populations across seeds and divide by the emitted
single-strain dose. NH3 uses the WT kernel multiplied by nh3_factor.
"""

import json
from pathlib import Path

import numpy as np


STRAIN_LABELS = {"WT": "WT", "Q": "Lp-Q", "NAC": "Lp-NAC", "SFN": "Lp-SFN"}


def load_retention(root: Path) -> dict:
    """Read the upstream NPZ selected by config.json inputs.retention."""
    root = Path(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    source = config["inputs"]["retention"]
    with np.load(root / source, allow_pickle=False) as coupling:
        result = retention_from_coupling(coupling)
    result["metadata"]["source"] = source
    return result


def retention_from_coupling(coupling) -> dict:
    """Convert an NPZ mapping with time/scenario/record/seed axes in memory.

    ``source_emitted_cfu_per_strain`` is the emitted input before deposition.
    ``initial_cfu`` in the nasal output instead describes the deposited dose.
    The returned free/attached kernels retain every first-hour output time.
    """
    times_h = np.asarray(coupling["times_s"], dtype=float) / 3600.0
    first_hour = (times_h >= 0.0) & (times_h <= 1.0)
    time_h = times_h[first_hour]
    if time_h.size < 2 or time_h[0] != 0.0 or not np.isclose(time_h[-1], 1.0):
        raise ValueError("Nasal coupling output must cover the full first hour.")
    scenario = list(np.asarray(coupling["scenarios"]).astype(str)).index("full")
    strains = np.asarray(coupling["strains"]).astype(str)
    emitted = float(np.asarray(coupling["source_emitted_cfu_per_strain"]).item())
    if not np.isfinite(emitted) or emitted <= 0.0:
        raise ValueError("The source emitted CFU must be finite and positive.")
    mobile = np.asarray(coupling["mobile_cfu"], dtype=float)
    attached = np.asarray(coupling["attached_cfu"], dtype=float)
    kernels, record_counts = {}, {}
    for strain, label in STRAIN_LABELS.items():
        records = np.flatnonzero(strains == label)
        if records.size != 1:
            raise ValueError(f"Expected one fitted growth trajectory for {label}.")
        record = int(records[0])
        kernels[strain] = np.column_stack((
            mobile[first_hour, scenario, record, :].mean(axis=-1),
            attached[first_hour, scenario, record, :].mean(axis=-1),
        )) / emitted
        record_counts[strain] = int(records.size)
    return {
        "time_h": time_h,
        "kernels": kernels,
        "metadata": {
            "source_script": "../nasal/scripts/run_best_coupled_results.py",
            "scenario": "full",
            "seeds": np.asarray(coupling["seeds"]).astype(int).tolist(),
            "source_emitted_cfu_per_strain": emitted,
            "normalization": "retained CFU per emitted CFU, including deposition",
            "aggregation": "one growth trajectory per strain; mean across spray seeds",
            "growth_record_counts": record_counts,
            "time_points": int(time_h.size),
            "nh3_proxy": "WT total retention multiplied by nh3_factor",
            "source_window_h": [0.0, 1.0],
            "after_source_window": "Zero active production source after one hour.",
            "population_scaling": "Single-strain kernels scale linearly with emitted CFU.",
        },
    }


def retained(
    data: dict,
    strain: str,
    t,
    nh3_factor: float = 1.0,
) -> np.ndarray:
    """Interpolate total active retention per emitted CFU at time(s) in hours.

    ``strain`` is WT, Q, NH3, NAC, or SFN. Scalar input returns a zero-dimensional
    array; array input preserves its shape. Before dosing and after one hour the
    source contribution is zero. NH3 uses WT retention scaled by nh3_factor.
    """
    proxy = "WT" if strain == "NH3" else strain
    total = data["kernels"][proxy].sum(axis=1)
    values = np.interp(
        np.asarray(t, dtype=float),
        data["time_h"],
        total,
        left=0.0,
        right=0.0,
    )
    if strain == "NH3":
        values = values * nh3_factor
    return np.asarray(values)

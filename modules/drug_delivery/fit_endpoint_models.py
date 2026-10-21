"""Fit NAC/SFN dose responses from post-induction experimental endpoints.

Read the shared Markdown table, fit each assay separately, and export model
parameters, endpoint summaries, and supplementary dose-response figures.
Run with python -B.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
RUNTIME.mkdir(exist_ok=True)
for key in ("TMPDIR", "TEMP", "TMP", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
    os.environ[key] = str(RUNTIME)

import csv
import json
import re
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import PchipInterpolator
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = ROOT.parents[1] / "data" / "experimental" / "湿实验结果总表.md"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

META = {
    "E01": ("NAC", "RAW", "ELISA", "mM", "LPS"),
    "E02": ("NAC", "RAW", "ELISA", "mM", "LPS"),
    "E03": ("SFN", "RAW", "ELISA", "uM", "LPS"),
    "E04": ("SFN", "BV2", "ELISA", "uM", "LPS"),
    "E09": ("NAC", "BV2", "JC1", "mM", "LPS"),
    "E10": ("SFN", "BV2", "JC1", "uM", "LPS"),
    "E11": ("NAC", "SH-SY5Y", "qPCR", "mM", "VPA"),
    "E12": ("SFN", "SH-SY5Y", "qPCR", "uM", "H2O2"),
    "E13": ("SFN", "SH-SY5Y", "qPCR", "uM", "H2O2"),
    "E14": ("SFN", "RAW", "qPCR", "uM", "LPS"),
    **{f"E{i}": ("SFN", "BV2", "qPCR", "uM", "LPS") for i in range(15, 20)},
}


def extract_groups():
    """Extract assay groups and stored R-column values from the input table."""
    experiment, header = None, None
    groups, records = [], []
    for line in SOURCE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"### (E\d+)\b", line)
        if match:
            experiment, header = match[1], None
        if experiment not in META:
            continue
        if not line.startswith("|"):
            header = None
            continue
        cells = [v.strip() for v in line.strip("|").split("|")]
        if "指标" in cells and "分组" in cells and "R1" in cells:
            header = cells
            continue
        if header is None or all(re.fullmatch(r"[-: ]+", v) for v in cells):
            continue
        if len(cells) != len(header):
            raise ValueError(f"Unexpected numeric row in {experiment}: {line}")
        row = dict(zip(header, cells))
        drug, cell, assay, unit, stimulus = META[experiment]
        label = row["分组"]
        role = "control" if label == "Control" else (
            "CCCP" if label == "CCCP" else "stimulated")
        match = re.search(rf"(\d+(?:\.\d+)?)\s*(mM|[μµu]M)\s*{drug}", label)
        dose = float(match[1]) if match else (0.0 if role == "stimulated" else None)
        if match:
            parsed_unit = "mM" if match[2] == "mM" else "uM"
            if parsed_unit != unit:
                raise ValueError(f"Unexpected concentration unit in {experiment}")
        group = dict(experiment=experiment, cell=cell, drug=drug, assay=assay,
                     stimulus=stimulus, endpoint=row["指标"], group=label,
                     role=role, dose=dose, dose_unit=unit)
        values = []
        for column in header:
            if not re.fullmatch(r"R\d+", column):
                continue
            value = None if row[column] == "NA" else float(row[column])
            records.append({**group, "source_column": column, "value": value})
            if value is not None:
                values.append(value)
        if not values:
            raise ValueError(f"No values in {experiment}: {label}")
        groups.append({**group, "values": values, "mean": float(np.mean(values)),
                       "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                       "stored_value_count": len(values)})
    if set(META) != {g["experiment"] for g in groups}:
        raise ValueError("A requested experimental table was not parsed.")
    return groups, records


def save_csv(name, rows):
    with (RESULTS / name).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subset(groups, experiment, endpoint=None, role="stimulated"):
    selected = [g for g in groups if g["experiment"] == experiment
                and (endpoint is None or g["endpoint"] == endpoint)
                and g["role"] == role]
    return sorted(selected, key=lambda g: g["dose"] or 0)


def fit_function(x, y, baseline, family, scale):
    """Unweighted least squares on group means; h=1 throughout.

    Solve the saturating amplitude analytically for each K, then optimize K
    over the numerical search interval.
    """
    x, y = np.asarray(x), np.asarray(y)
    if family == "log_linear":
        beta = float(np.dot(x, y - baseline) / np.dot(x, x))
        return {"family": family, "baseline": baseline, "beta": beta}

    def amplitude_and_sse(log_k):
        k = np.exp(log_k)
        h = x / (k + x)
        if family == "inhibitory_emax":
            target = 1 - y / baseline
            amplitude = np.clip(np.dot(h, target) / np.dot(h, h), 0, 1)
            predicted = baseline * (1 - amplitude * h)
        else:
            amplitude = np.dot(h, y - baseline) / np.dot(h, h)
            predicted = baseline + amplitude * h
        return float(amplitude), float(np.sum((y - predicted) ** 2))

    low, high = np.log(scale * 1e-4), np.log(scale * 1e4)
    optimum = minimize_scalar(lambda z: amplitude_and_sse(z)[1],
                              bounds=(low, high), method="bounded")
    if not optimum.success:
        raise RuntimeError(optimum.message)
    amplitude, _ = amplitude_and_sse(optimum.x)
    return {"family": family, "baseline": baseline, "amplitude": amplitude,
            "K": float(np.exp(optimum.x)), "h": 1,
            "K_at_search_boundary": bool(min(optimum.x - low, high - optimum.x) < 0.03)}


def predict(model, dose):
    dose = np.asarray(dose)
    if model["family"] == "log_linear":
        return model["baseline"] + model["beta"] * dose
    h = dose / (model["K"] + dose)
    if model["family"] == "inhibitory_emax":
        return model["baseline"] * (1 - model["amplitude"] * h)
    return model["baseline"] + model["amplitude"] * h


def fit_assay(groups, experiment, endpoint, families, log_scale=False):
    data = subset(groups, experiment, endpoint)
    x = np.array([g["dose"] for g in data])
    # Fit qPCR responses using the mean of log2-transformed expression values.
    y = np.array([np.mean(np.log2(g["values"])) if log_scale else g["mean"]
                  for g in data])
    baseline = float(y[x == 0][0])
    positive = x > 0
    candidates = []
    for family in families:
        model = fit_function(x[positive], y[positive], baseline, family, float(x.max()))
        model["fit_rmse"] = float(np.sqrt(np.mean(
            (predict(model, x[positive]) - y[positive]) ** 2)))
        holdouts = []
        for index in np.flatnonzero(positive):
            train = positive.copy()
            train[index] = False
            fitted = fit_function(x[train], y[train], baseline, family, float(x.max()))
            estimate = float(predict(fitted, x[index]))
            holdouts.append(dict(dose=float(x[index]), observed=float(y[index]),
                                 predicted=estimate,
                                 extrapolation=bool(x[index] > max(x[train]))))
        model["leave_one_dose_out_rmse"] = float(np.sqrt(np.mean(
            [(r["predicted"] - r["observed"]) ** 2 for r in holdouts])))
        model["held_out_doses"] = holdouts
        if "K" in model:
            model["K_inside_tested_positive_range"] = bool(
                min(x[positive]) <= model["K"] <= max(x))
        candidates.append(model)
    # Prefer candidate models with K inside the numerical search interval.
    eligible = [m for m in candidates if not m.get("K_at_search_boundary", False)]
    selected = min(eligible or candidates, key=lambda m: m["leave_one_dose_out_rmse"])
    return dict(experiment=experiment, endpoint=endpoint, dose_unit=data[0]["dose_unit"],
                response_scale="log2_expression" if log_scale else "pg_per_mL",
                selection_rule="lowest leave-one-positive-dose-out RMSE among non-boundary candidates",
                selected=selected, candidates=candidates,
                tested_doses=x.tolist(), fixed_endpoint_time_hours=None)


def jc1_groups(groups, experiment):
    output = []
    for red in [g for g in groups if g["experiment"] == experiment and g["endpoint"] == "RED"]:
        green = next(g for g in groups if g["experiment"] == experiment
                     and g["endpoint"] == "GREEN" and g["group"] == red["group"])
        output.append({k: v for k, v in red.items() if k not in ("values", "mean", "sd")} |
                      {"red_percent": red["mean"], "green_percent": green["mean"],
                       "AM_ratio": red["mean"] / green["mean"]})
    return output


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "legend.frameon": False})
COLORS = {"SFN": "#CA7D32", "NAC": "#228B8D"}


def finish(fig, name):
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for extension in ("png", "svg"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


def point_panel(ax, data, title, ylabel, fit=None, log_scale=False):
    color = COLORS[data[0]["drug"]]
    dose = np.array([g["dose"] for g in data])
    for group in data:
        values = np.log2(group["values"]) if log_scale else group["values"]
        offsets = np.linspace(-0.006, 0.006, len(values)) * dose.max()
        ax.scatter(group["dose"] + offsets, values, s=22, color=color, alpha=0.65,
                   edgecolors="white", linewidth=0.3, zorder=3)
    center = [np.mean(np.log2(g["values"])) if log_scale else g["mean"] for g in data]
    ax.scatter(dose, center, s=33, marker="D", facecolors="white",
               edgecolors=color, zorder=4, label="Mean")
    if fit is not None:
        grid = np.linspace(0, dose.max(), 200)
        ax.plot(grid, predict(fit["selected"], grid), color=color, lw=1.6, label="Fit")
    else:
        ax.plot(dose, center, color=color, ls=":", lw=1, label="Guide only")
    ax.set(title=title, xlabel=f'{data[0]["drug"]} ({data[0]["dose_unit"].replace("uM", "μM")})',
           ylabel=ylabel)
    ax.set_xticks(dose)
    ax.grid(axis="y", alpha=0.15)


def plot_jc1(ax, rows, drug):
    data = sorted([r for r in rows if r["role"] == "stimulated"], key=lambda r: r["dose"])
    x, y = np.array([r["dose"] for r in data]), np.array([r["AM_ratio"] for r in data])
    grid = np.linspace(0, x.max(), 200)
    ax.plot(grid, PchipInterpolator(x, y, extrapolate=False)(grid),
            color=COLORS[drug], ls="--", lw=1.3, label="Interpolation only")
    ax.scatter(x, y, color=COLORS[drug], s=32, zorder=3)
    ax.axhline(y[0], color="#888888", lw=0.8, ls=":")
    ax.set(title=f"{drug} · BV2 · JC-1", ylabel="RED% / GREEN%",
           xlabel=f'{drug} ({"μM" if drug == "SFN" else "mM"})', xticks=x)
    ax.grid(axis="y", alpha=0.15)


def make_figures(groups, fits, jc):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
    fig.suptitle("Cytokine responses", fontsize=15)
    for ax, eid, endpoint in zip(axes.flat, ("E01", "E02", "E03", "E04"),
                                 ("IL-6", "TNF-α", "IL-6", "IL-6")):
        data = subset(groups, eid, endpoint)
        fit = fits[(eid, endpoint)]
        point_panel(ax, data, f"{data[0]['drug']} · {data[0]['cell']} · {endpoint}",
                    "Concentration (pg/mL)", fit)
        control = subset(groups, eid, endpoint, "control")[0]["mean"]
        ax.axhline(control, color="#888888", ls="--", lw=0.9, label="Control")
        ax.set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=9)
    finish(fig, "01_cytokine_response")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    fig.suptitle("SFN · BV2 responses", fontsize=15)
    plot_jc1(axes.flat[0], jc["E10"], "SFN")
    for ax, eid, gene in zip(list(axes.flat)[1:], ("E15", "E16", "E17", "E18", "E19"),
                             ("Nfe2l2", "Hmox1", "Nos2", "Il6", "Tnf")):
        point_panel(ax, subset(groups, eid, gene), gene,
                    "log₂ relative expression", fits.get((eid, gene)), log_scale=True)
        control = subset(groups, eid, gene, "control")[0]
        ax.axhline(np.mean(np.log2(control["values"])), color="#888888", ls="--", lw=0.8)
    finish(fig, "02_sfn_bv2_endpoints")

    fig = plt.figure(figsize=(13, 7.5))
    grid = fig.add_gridspec(2, 6)
    axes = [fig.add_subplot(grid[0, :3]), fig.add_subplot(grid[0, 3:]),
            fig.add_subplot(grid[1, :2]), fig.add_subplot(grid[1, 2:4]),
            fig.add_subplot(grid[1, 4:])]
    fig.suptitle("NAC · Cellular responses", fontsize=15)
    plot_jc1(axes[0], jc["E09"], "NAC")
    for ax, gene in zip(axes[1:], ("NOTCH1", "HES1", "BECN1", "JAG1")):
        point_panel(ax, subset(groups, "E11", gene), f"SH-SY5Y · {gene}",
                    "Relative expression")
        control = subset(groups, "E11", gene, "control")[0]["mean"]
        ax.axhline(control, color="#888888", ls="--", lw=0.8)
    finish(fig, "03_nac_endpoints")

    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.8), sharex=True)
    fig.suptitle("SFN · BV2 dose responses", fontsize=15)
    data = subset(groups, "E04", "IL-6")
    y0 = data[0]["mean"]
    doses = np.array([g["dose"] for g in data])
    reductions = np.array([100 * (1 - g["mean"] / y0) for g in data])
    axes[0].plot(doses, reductions, "o-", color=COLORS["SFN"])
    axes[0].set(ylabel="IL-6 reduction vs LPS (%)", title="IL-6")
    plot_jc1(axes[1], jc["E10"], "SFN")
    axes[1].set_title("JC-1")
    for ax in axes:
        ax.axvspan(5, 10, color="#739BA5", alpha=0.12)
        ax.set_xticks([0, 2.5, 5, 10, 15, 20])
    finish(fig, "04_sfn_dose_comparison")


def main():
    groups, records = extract_groups()
    save_csv("endpoint_values.csv", records)
    save_csv("endpoint_groups.csv", [{k: v for k, v in g.items() if k != "values"} for g in groups])
    fits = {}
    for eid, gene in (("E01", "IL-6"), ("E02", "TNF-α"), ("E03", "IL-6"), ("E04", "IL-6")):
        fits[(eid, gene)] = fit_assay(groups, eid, gene, ["inhibitory_emax"])
    for eid, gene in (("E16", "Hmox1"), ("E18", "Il6"), ("E19", "Tnf")):
        fits[(eid, gene)] = fit_assay(groups, eid, gene,
                                    ["log_linear", "log_saturating"], log_scale=True)
    jc = {eid: jc1_groups(groups, eid) for eid in ("E09", "E10")}
    save_csv("jc1_ratios.csv", jc["E09"] + jc["E10"])
    comparison = []
    il6 = subset(groups, "E04", "IL-6")
    for g in il6:
        paired = next(r for r in jc["E10"] if r["role"] == "stimulated" and r["dose"] == g["dose"])
        comparison.append(dict(SFN_uM=g["dose"], IL6_pg_mL=g["mean"],
                               IL6_reduction_percent=100 * (1 - g["mean"] / il6[0]["mean"]),
                               JC1_AM=paired["AM_ratio"],
                               comparison_type="dose-aligned separate assays; timing unconfirmed"))
    save_csv("sfn_dose_comparison.csv", comparison)
    proximity = []
    for gene in ("NOTCH1", "HES1", "BECN1", "JAG1"):
        control = subset(groups, "E11", gene, "control")[0]["mean"]
        for g in subset(groups, "E11", gene):
            ratio = g["mean"] / control
            proximity.append(dict(gene=gene, NAC_mM=g["dose"], ratio_to_control=ratio,
                                  absolute_log2_distance=abs(float(np.log2(ratio)))))
    save_csv("nac_control_proximity.csv", proximity)
    (RESULTS / "model_results.json").write_text(json.dumps(
        {"treatment_order": "after model induction", "endpoint_time_hours": None,
         "stimulus_removed_before_treatment": None, "independent_validation": False,
         "replicate_identity": "unconfirmed", "models": list(fits.values()),
         "notes": ["Baseline fixed to matching stimulated zero-dose endpoint.",
                   "Leave-one-dose-out is internal, uses the same zero-dose reference in every fold.",
                   "Evaluation setting: model selection and evaluation share the same leave-one-dose-out folds; reported errors describe internal evaluation.",
                   "Calibration scope: separate cellular dose–response endpoints. Dynamics, transport, drug interactions, viability and ROS require their own calibration inputs."]},
        ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    make_figures(groups, fits, jc)
    for key, value in fits.items():
        model = value["selected"]
        print(f"{key}: {model['family']}; fit RMSE={model['fit_rmse']:.4g}; "
              f"LODO RMSE={model['leave_one_dose_out_rmse']:.4g}; "
              f"K={model.get('K', 'n/a')}; amplitude={model.get('amplitude', 'n/a')}")
    print(json.dumps({"numeric_values": sum(r["value"] is not None for r in records),
                      "preserved_missing_slots": sum(r["value"] is None for r in records),
                      "assays": len(META), "groups": len(groups),
                      "fitted_endpoints": len(fits), "figures": 4,
                      "sfn_comparison": comparison}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

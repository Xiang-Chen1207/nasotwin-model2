"""Draw Wiki Figures N4a, N4b, and N4d from model inputs and calibration outputs."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
RUNTIME.mkdir(parents=True, exist_ok=True)
for key in ("TMPDIR", "TMP", "TEMP", "MPLCONFIGDIR", "XDG_CACHE_HOME"):
    os.environ[key] = str(RUNTIME)

import csv
import json
import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGURES = ROOT / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
COLORS = {"NAC": "#228B8D", "SFN": "#CA7D32"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})


def save(fig, name):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=240,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = plt.subplots(figsize=(13.2, 4.2))
    ax.set(xlim=(-0.2, 14.6), ylim=(-0.3, 4.4))
    ax.axis("off")

    def box(x, y, width, height, label, color):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=color, edgecolor="#43616A", linewidth=1.1))
        ax.text(x + width / 2, y + height / 2, label,
                ha="center", va="center", fontsize=12, linespacing=1.5)

    def arrow(a, b, connection="arc3", dashed=False):
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=14,
            linewidth=1.4, color="#43616A", connectionstyle=connection,
            linestyle="--" if dashed else "-"))

    box(0, 1.7, 2.2, 1.0, "Continuous source\n$q(t)B(t)$", "#D9EEEE")
    box(3.0, 1.7, 1.9, 1.0, "Nasal pool\n$C_n$", "#E7EEF2")
    box(5.9, 2.9, 2.1, 0.9, "Direct transit", "#E7EEF2")
    box(5.9, 0.6, 2.1, 0.9, "Plasma\n$C_p$", "#E7EEF2")
    box(9.0, 1.7, 2.1, 1.0, "Target exposure\n$C_b$", "#F6E8D5")
    box(12.0, 1.7, 2.2, 1.0, "PD response\n$C_e\\;\\rightarrow\\; I$", "#F6E8D5")
    box(12.0, 0.0, 2.2, 0.8, "ELISA endpoints", "#F0F1F2")
    arrow((2.25, 2.2), (2.95, 2.2))
    arrow((4.95, 2.5), (5.85, 3.3))
    arrow((4.95, 1.9), (5.85, 1.1))
    arrow((8.05, 3.3), (8.95, 2.5))
    arrow((8.05, 1.1), (8.95, 1.9))
    arrow((9.0, 1.72), (8.0, 0.8), "arc3,rad=-0.20")
    arrow((11.15, 2.2), (11.95, 2.2))
    arrow((13.1, 0.85), (13.1, 1.65), dashed=True)
    fig.tight_layout(pad=0.4)
    save(fig, "model_overview")


def endpoint_prediction(doses_uM, calibration):
    nodes, weights = leggauss(96)
    duration = calibration["reference_assay_duration_h"]
    time = (nodes + 1) * duration / 2
    weights = weights * duration / 2
    ce = np.asarray(doses_uM)[:, None] * (
        1 - np.exp(-time / calibration["tau_effect_h"]))
    kernel = np.exp(-(duration - time) / calibration["tau_response_h"])
    kernel /= calibration["tau_response_h"]
    gain = np.sum(weights * kernel * ce / (calibration["K_medium_uM"] + ce), axis=1)
    return calibration["baseline_pg_mL"] * (
        1 - calibration["Emax_infinity"] * gain)


def pd_calibration():
    result = json.loads((ROOT.parent / "pkpd_continuous_results/conditional_pkpd_results.json").read_text())
    with (ROOT.parent / "results/endpoint_values.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))
    for ax, cal in zip(axes.flat, result["conditional_PD_calibrations"]):
        drug, eid = cal["drug"], cal["experiment"]
        color = COLORS[drug]
        scale = 1000 if drug == "NAC" else 1
        selected = [r for r in rows if r["experiment"] == eid and r["value"]]
        observations = [r for r in selected if r["role"] == "stimulated"]
        doses = sorted({float(r["dose"]) for r in observations})
        for dose in doses:
            values = [float(r["value"]) for r in observations if float(r["dose"]) == dose]
            offsets = np.linspace(-0.006, 0.006, len(values)) * max(doses)
            ax.scatter(dose + offsets, values, s=24, color=color,
                       alpha=0.7, linewidths=0.4, edgecolors="white", zorder=3)
            ax.scatter(dose, np.mean(values), s=36, marker="D",
                       facecolor="white", edgecolor=color, zorder=4,
                       label="Mean" if dose == 0 else None)
        grid = np.linspace(0, max(doses), 250)
        ax.plot(grid, endpoint_prediction(grid * scale, cal),
                color=color, lw=1.8, label="Conditional fit")
        controls = [float(r["value"]) for r in selected if r["role"] == "control"]
        ax.axhline(np.mean(controls), color="#888888", ls="--", lw=1, label="Control")
        unit = "mM" if drug == "NAC" else "μM"
        ax.set(title=f"{drug} · {cal['cell']} · {cal['endpoint']}",
               xlabel=f"{drug} ({unit})", ylabel="Concentration (pg/mL)",
               xticks=doses, ylim=(0, None))
        ax.grid(axis="y", alpha=0.15)
    axes[0, 0].legend(fontsize=9)
    fig.tight_layout()
    save(fig, "dynamic_pd_calibration")


def production_design():
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    rho = np.linspace(0, 12, 350)
    ax.plot(rho, rho / (1 + rho), color="#228B8D", lw=2.2)
    for x, y, label in ((1, 0.5, "50%"), (4, 0.8, "80%"), (9, 0.9, "90%")):
        ax.plot([x, x], [0, y], ls=":", lw=1, color="#A4AFB1")
        ax.scatter(x, y, s=45, color="#CA7D32", zorder=3)
        ax.annotate(label, (x, y), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=11)
    ax.set(xlim=(0, 12), ylim=(0, 1.05), xticks=[0, 1, 4, 9, 12],
           xlabel=r"Normalized production, $\rho=\kappa G R/K$",
           ylabel=r"Fraction of maximum response, $E/E_{\max,\infty}$")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    save(fig, "supply_response_design")


if __name__ == "__main__":
    overview()
    pd_calibration()
    production_design()
    print("Created 3 figures (PNG, SVG, PDF) from saved data and analytical equations.")

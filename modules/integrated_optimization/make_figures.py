"""Plot workflow, productivity requirements, dynamics, and scenario allocations.

Read generated JSON and CSV results and export each figure as PDF, PNG, and SVG.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import textwrap

import numpy as np

from metabolism import load_parameters


COLORS = {
    "Q": "#0072B2",
    "NH3": "#E69F00",
    "NAC": "#009E73",
    "SFN": "#CC79A7",
    "optimized": "#0072B2",
    "equal_split": "#D55E00",
    "metabolic_40_60": "#009E73",
}
STRATEGIES = ("optimized", "equal_split")
STRATEGY_LABELS = {
    "optimized": "Allocation by functional demand",
    "equal_split": "Four equal shares",
    "metabolic_40_60": "Metabolic 40:60 reference",
}
LINESTYLES = {"optimized": "-", "equal_split": "--", "metabolic_40_60": "-."}


def _configure(root: Path):
    cache = root / "build" / "runtime" / "mpl"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.10,
        }
    )
    return plt


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _is_true(value):
    return str(value).strip().lower() in {"true", "1", "1.0", "yes"}


def _numeric(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _save(plt, fig, directory: Path, name: str):
    for suffix in ("pdf", "png", "svg"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=200)
    plt.close(fig)
    print(name)


def _panel(ax, letter, label=""):
    ax.text(
        0,
        1.045,
        f"{letter}  {label}".rstrip(),
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _workflow(plt, destination):
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10.7, 2.6))
    ax.set_xlim(0, 12.3)
    ax.set_ylim(0, 3.5)
    ax.axis("off")

    def box(x, y, w, h, text, color="#555555"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.1,
            edgecolor=color,
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(start, end, connection="arc3,rad=0"):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                color="#555555",
                linewidth=1.0,
                connectionstyle=connection,
            )
        )

    box(0.10, 1.25, 2.08, 0.90, "Total dose and allocation\nQ / NH3 / NAC / SFN")
    box(2.65, 1.25, 2.15, 0.90, "Nasal Dynamics\nRetained active bacteria")
    box(
        5.30,
        2.25,
        3.00,
        0.90,
        "Metabolic Exchange\nQ / NH3 · Gln and ammonia",
        COLORS["Q"],
    )
    box(
        5.30,
        0.25,
        3.00,
        0.90,
        "Drug Delivery & Response\nNAC / SFN · Production → PK–PD",
        COLORS["NAC"],
    )
    box(
        9.05,
        1.25,
        3.00,
        0.90,
        "Meet all design targets\nMinimize total dose",
    )
    arrow((2.22, 1.70), (2.61, 1.70))
    arrow((4.84, 1.90), (5.26, 2.70))
    arrow((4.84, 1.50), (5.26, 0.70))
    arrow((8.34, 2.70), (9.01, 1.90))
    arrow((8.34, 0.70), (9.01, 1.50))
    fig.subplots_adjust(left=0.015, right=0.995, top=0.99, bottom=0.035)
    _save(plt, fig, destination, "01_integration_workflow")


def _allocation_feasibility(plt, destination, rows, summary):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    cmap = ListedColormap(["#ECECEC", "#56B4E9"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    scenarios = {row["name"]: row for row in summary["scenarios"]}
    scenarios["robust_selected"] = summary["robust"]
    reference_cfu = float(summary["reference"]["emitted_cfu"])
    for ax, name, letter, label in zip(
        axes,
        ("nominal", "robust_selected"),
        ("a", "b"),
        ("Nominal scenario", "All selected scenarios"),
    ):
        selected = [row for row in rows if row["scenario"] == name]
        if not selected:
            _panel(ax, letter, label)
            ax.text(0.5, 0.5, "Search-grid result:\nno feasible allocation",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_xlabel("Fraction allocated to metabolic strains")
            continue
        x = np.unique(_numeric(selected, "metabolic_fraction"))
        y = np.unique(_numeric(selected, "total_dose_ref"))
        grid = np.full((len(y), len(x)), np.nan)
        x_positions = {value: index for index, value in enumerate(x)}
        y_positions = {value: index for index, value in enumerate(y)}
        for row in selected:
            grid[
                y_positions[float(row["total_dose_ref"])],
                x_positions[float(row["metabolic_fraction"])],
            ] = float(_is_true(row["feasible"]))
        ax.pcolormesh(x, y, np.ma.masked_invalid(grid), cmap=cmap, norm=norm, shading="nearest")
        if len(y) and y.min() > 0 and y.max() / y.min() > 20:
            ax.set_yscale("log")
        result = scenarios[name]
        if result["feasible"]:
            total = float(result["total_cfu"])
            metabolic = sum(float(result["allocations_cfu"][s]) for s in ("Q", "NH3"))
            ax.plot(
                metabolic / total,
                total / reference_cfu,
                marker="*",
                markersize=11,
                markerfacecolor="#D55E00",
                markeredgecolor="white",
                markeredgewidth=0.7,
                linestyle="none",
                zorder=5,
            )
        _panel(ax, letter, label)
        ax.set_xlabel("Fraction allocated to metabolic strains")
        ax.set_xlim(x.min(), x.max())
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
    axes[0].set_ylabel("Total allocation (reference doses)")
    handles = [
        Patch(facecolor="#ECECEC", label="Constraints not met"),
        Patch(facecolor="#56B4E9", label="Feasible grid point"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#D55E00",
            markeredgecolor="white",
            markersize=10,
            label="Minimum total allocation",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.25, wspace=0.15)
    _save(plt, fig, destination, "02_feasible_allocation")


def _productivity(plt, destination, rows, summary):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.7))
    reference_cfu = float(summary["reference"]["emitted_cfu"])
    for ax, drug, letter in zip(axes, ("NAC", "SFN"), ("a", "b")):
        selected = sorted(
            (row for row in rows if row["drug"] == drug),
            key=lambda row: float(row["allocated_cfu"]),
        )
        allocated = _numeric(selected, "allocated_cfu")
        for key, label, linestyle, color in (
            ("required_q_nominal", "Nominal exposure", "-", COLORS[drug]),
            ("required_q_lower_exposure", "Lower exposure", "--", "#555555"),
        ):
            productivity = _numeric(selected, key)
            valid = (allocated > 0) & (productivity > 0) & np.isfinite(productivity)
            ax.loglog(allocated[valid], productivity[valid], color=color, linestyle=linestyle, label=label)
        _panel(ax, letter, drug)
        ax.set_xlabel("Allocated emitted bacteria (CFU)")
        ax.set_ylabel(r"Required productivity (nmol CFU$^{-1}$ h$^{-1}$)")
        ax.grid(which="major", color="#E0E0E0", linewidth=0.5)
        required_q = float(summary["drug_requirements"][drug]["amplitude"]) / reference_cfu
        ax.plot(
            reference_cfu,
            required_q,
            marker="o",
            markersize=4,
            color=COLORS[drug],
            markeredgecolor="white",
            markeredgewidth=0.6,
            zorder=5,
        )
        ax.text(
            0.97,
            0.97,
            f"At {reference_cfu:.2g} emitted CFU:\n"
            f"required q = {required_q:.3g}\n"
            r"nmol CFU$^{-1}$ h$^{-1}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2},
        )
        ax.legend(frameon=False, loc="lower left")
    fig.subplots_adjust(left=0.095, right=0.98, top=0.90, bottom=0.18, wspace=0.30)
    _save(plt, fig, destination, "03_productivity_requirements")


def _dynamics(plt, destination, metabolic_rows, drug_rows, summary, metabolic_parameters):
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))
    for strategy in STRATEGIES:
        selected = sorted(
            (
                row
                for row in metabolic_rows
                if row["scenario"] == "nominal" and row["strategy"] == strategy
            ),
            key=lambda row: float(row["time_h"]),
        )
        time = _numeric(selected, "time_h")
        for ax, key in ((axes[0, 0], "A"), (axes[0, 1], "G")):
            ax.plot(
                time,
                _numeric(selected, key),
                color=COLORS[strategy],
                linestyle=LINESTYLES[strategy],
                label=STRATEGY_LABELS[strategy],
            )
        for ax, drug, channel in (
            (axes[1, 0], "NAC", "E02"),
            (axes[1, 1], "SFN", "E03"),
        ):
            chosen = sorted(
                (
                    row
                    for row in drug_rows
                    if row["scenario"] == "nominal"
                    and row["strategy"] == strategy
                    and row["drug"] == drug
                    and row["channel"] == channel
                ),
                key=lambda row: float(row["time_h"]),
            )
            ax.plot(
                _numeric(chosen, "time_h"),
                _numeric(chosen, "effect"),
                color=COLORS[strategy],
                linestyle=LINESTYLES[strategy],
            )
    axes[0, 0].set_ylabel("Ammonia in reference pool (mM)")
    axes[0, 1].set_ylabel("Glutamine in reference pool (mM)")
    initial_ammonia = float(metabolic_parameters["reference"]["initial_total_ammonia_mM"])
    initial_gln = (
        float(metabolic_parameters["gln"]["gln_basal_mM"])
        + float(metabolic_parameters["reference"]["gln_supplement_mM"])
    )
    peak_limit = initial_ammonia * float(
        metabolic_parameters["design"]["max_peak_ammonia_to_initial"]
    )
    gln_target = initial_gln * (
        1.0 - float(metabolic_parameters["design"]["target_gln_removal_fraction"])
    )
    axes[0, 0].axhline(peak_limit, color="#666666", linewidth=1, linestyle=":")
    axes[0, 0].axhline(initial_ammonia, color="#666666", linewidth=1, linestyle="-.")
    axes[0, 1].axhline(gln_target, color="#666666", linewidth=1, linestyle=":")
    for ax, channel in ((axes[1, 0], "E02"), (axes[1, 1], "E03")):
        ax.axhline(
            float(summary["reference"]["targets"][channel]),
            color="#666666",
            linewidth=1,
            linestyle=":",
        )
        ax.set_ylabel("Fractional channel response")
    for ax, letter, label in (
        (axes[0, 0], "a", "Metabolic ammonia"),
        (axes[0, 1], "b", "Metabolic glutamine"),
        (axes[1, 0], "c", "NAC · RAW TNF-α (E02)"),
        (axes[1, 1], "d", "SFN · RAW IL-6 (E03)"),
    ):
        _panel(ax, letter, label)
        ax.set_xlabel("Time (h)")
        ax.set_xlim(left=0)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[strategy],
            linestyle=LINESTYLES[strategy],
            label=STRATEGY_LABELS[strategy],
        )
        for strategy in STRATEGIES
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        ncol=2,
        title="Allocation",
        title_fontsize=8,
        bbox_to_anchor=(0.5, 0.075),
    )
    fig.legend(
        handles=[
            Line2D(
                [0], [0], color="#666666", linewidth=1, linestyle=":",
                label="Design target / peak limit",
            ),
            Line2D(
                [0], [0], color="#666666", linewidth=1, linestyle="-.",
                label="Initial ammonia / end-point ceiling",
            ),
        ],
        frameon=False,
        loc="lower center",
        ncol=2,
        title="Reference lines",
        title_fontsize=8,
        bbox_to_anchor=(0.5, 0),
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.25, hspace=0.54, wspace=0.30)
    _save(plt, fig, destination, "04_equal_budget_dynamics")


def _scenario_allocations(plt, destination, summary):
    from matplotlib.patches import Patch

    results = list(summary["scenarios"]) + [summary["robust"]]
    reference_cfu = float(summary["reference"]["emitted_cfu"])
    positions = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(9.0, 4.15))
    bottom = np.zeros(len(results))
    for strain in ("Q", "NH3", "NAC", "SFN"):
        values = np.asarray(
            [
                float(result["allocations_cfu"][strain]) / reference_cfu
                if result["feasible"]
                else 0.0
                for result in results
            ]
        )
        ax.bar(
            positions,
            values,
            bottom=bottom,
            width=0.60,
            color=COLORS[strain],
            edgecolor="white",
            linewidth=0.65,
            label=f"Lp-{strain}",
        )
        bottom += values
    ymax = max(float(bottom.max()) * 1.20, 1.0)
    for position, result, total in zip(positions, results, bottom):
        if result["feasible"]:
            ax.text(position, total + ymax * 0.015, f"{total:.2f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.plot(position, ymax * 0.035, marker="x", color="#555555", markersize=7)
            ax.text(
                position,
                ymax * 0.08,
                "No feasible\nallocation",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
    labels = [
        "\n".join(textwrap.wrap(str(result.get("label", result["name"])), width=18))
        for result in results
    ]
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Allocation (reference doses)")
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[Patch(facecolor=COLORS[strain], label=f"Lp-{strain}") for strain in ("Q", "NH3", "NAC", "SFN")],
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.83, bottom=0.20)
    _save(plt, fig, destination, "05_scenario_allocations")


def main(root=None):
    """Read result tables and write PDF, PNG and SVG versions of each figure."""
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    plt = _configure(root)
    results = root / "results"
    destination = root / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    with (results / "optimization_results.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    metabolic_parameters = load_parameters(root)
    _workflow(plt, destination)
    _productivity(plt, destination, _read_csv(results / "supply_requirements.csv"), summary)
    _dynamics(
        plt,
        destination,
        _read_csv(results / "metabolic_trajectories.csv"),
        _read_csv(results / "drug_trajectories.csv"),
        summary,
        metabolic_parameters,
    )
    _scenario_allocations(plt, destination, summary)


if __name__ == "__main__":
    main()

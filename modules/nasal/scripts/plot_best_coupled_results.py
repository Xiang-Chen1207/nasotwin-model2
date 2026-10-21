"""Plot spatial fields, retained populations, and mechanism comparisons."""
from pathlib import Path
import argparse
import os
import csv
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "coupling"
FIGURES = ROOT / "figures"
REPORTS = ROOT / "outputs" / "coupling_figures"
for key, subdir in {
    "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
    "XDG_CACHE_HOME": "cache", "MPLCONFIGDIR": "mpl",
}.items():
    p = ROOT / ".build" / f"coupled-plots-{os.getuid()}" / subdir
    p.mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(p)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"WT": "#526575", "Lp-NAC": "#007F86", "Lp-SFN": "#CD762B", "Lp-Q": "#7962A9"}
GROUPS = list(COLORS)
SCENARIOS = ["full", "no_growth", "no_attachment", "faster_clearance"]
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.labelsize": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#B2BDC7", "axes.labelcolor": "#25394A", "text.color": "#25394A",
    "xtick.color": "#526575", "ytick.color": "#526575", "axes.linewidth": .8,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
})


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURES / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=.12)
    plt.close(fig)
    print("Saved", name, flush=True)


def title(ax, letter, text):
    ax.set_title(text, loc="left", fontsize=11, pad=14, color="#405868")
    ax.text(-.13, 1.06, letter, transform=ax.transAxes, fontsize=15, fontweight="bold")


def main():
    global DATA, FIGURES, REPORTS
    parser = argparse.ArgumentParser(description="Plot saved nasal coupling results.")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=FIGURES)
    parser.add_argument("--report-dir", type=Path, default=REPORTS)
    args = parser.parse_args()
    DATA, FIGURES, REPORTS = args.data_dir, args.output_dir, args.report_dir
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    with np.load(DATA / "coupled_results.npz") as z:
        data = {k: z[k] for k in z.files}
    with (DATA / "metrics_summary.csv").open() as f:
        summary = {(r["scenario"], r["strain"]): r for r in csv.DictReader(f)}
    with (DATA / "metrics_by_record_seed.csv").open() as f:
        raw = list(csv.DictReader(f))
    config = json.loads((DATA / "results_summary.json").read_text())
    t = data["times_s"] / 60

    def series(key, scenario, group):
        block = data[key][:, SCENARIOS.index(scenario)]
        ids = np.flatnonzero(data["strains"] == group)
        return np.take(block, ids, axis=1).mean(axis=1)

    # Results 3: Lp-NAC is the representative growth input; each spray seed is retained.
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1.08]})
    free = series("mobile_cfu", "full", "Lp-NAC")
    attached = series("attached_cfu", "full", "Lp-NAC")
    retained = free + attached
    for values, color, label in [
        (retained, "#243D4B", "Total retained"),
        (free, "#008F95", "Free"),
        (attached, "#8957AE", "Attached"),
    ]:
        axes[0, 0].fill_between(t, values.min(axis=1)/1000, values.max(axis=1)/1000,
                               color=color, alpha=.10, lw=0)
        axes[0, 0].plot(t, values.mean(axis=1)/1000, color=color, lw=2, label=label)
    for key, color, label in [
        ("outflow_cfu", "#758A96", "Cumulative departure"),
        ("growth_added_cfu", "#CD762B", "Cumulative growth"),
    ]:
        values = series(key, "full", "Lp-NAC")
        axes[0, 1].fill_between(t, values.min(axis=1)/1000, values.max(axis=1)/1000,
                               color=color, alpha=.14, lw=0)
        axes[0, 1].plot(t, values.mean(axis=1)/1000, color=color, lw=2, label=label)
    for ax in axes[0]:
        ax.set(xlabel="Time after deposition (min)", ylabel=r"Population ($10^3$ CFU)",
               xlim=(0, 60), ylim=(0, None), xticks=[0, 15, 30, 45, 60])
        ax.grid(axis="y", color="#E6EBEF", lw=.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, loc="best")
    ids = np.flatnonzero(data["strains"] == "Lp-NAC")
    dx = float(data["x_mm"][1] - data["x_mm"][0])
    for ax, key in zip(axes[1], ["full_mobile_axial_cfu", "full_attached_axial_cfu"]):
        field = np.take(data[key], ids, axis=1).mean(axis=(1, 2)) / dx / 1000
        image = ax.pcolormesh(
            data["x_mm"], data["map_times_s"]/60, field,
            cmap="viridis_r", vmin=0, vmax=np.ceil(field.max()*2)/2, shading="nearest",
            rasterized=True,
        )
        ax.set(xlabel="Distance along target mucus region (mm)",
               ylabel="Time after deposition (min)", xlim=(20, 47), ylim=(0, 60),
               xticks=[20, 25, 30, 35, 40, 45, 47], yticks=[0, 15, 30, 45, 60])
        cb = fig.colorbar(image, ax=ax, fraction=.047, pad=.025)
        cb.set_label(r"Axial population ($10^3$ CFU/mm)", fontsize=10)
        cb.ax.tick_params(labelsize=9)
    for ax, letter, text in zip(axes.flat, "ABCD", [
        "Population states", "Departure and growth",
        "Free bacteria", "Attached bacteria",
    ]):
        title(ax, letter, text)
    save(fig, "fig08_best_coupled_dynamics")

    # Results 4: seed-paired comparisons using one fitted curve per strain.
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
    effects = []
    for j, group in enumerate(GROUPS):
        xpos = np.arange(4) + (j-1.5)*.15
        for c, scenario in enumerate(SCENARIOS):
            values, effects_auc = [], []
            for seed in data["seeds"]:
                subset = [r for r in raw if r["strain"] == group and int(r["seed"]) == seed]
                rows = [r for r in subset if r["scenario"] == scenario]
                base = [r for r in subset if r["scenario"] == "full"]
                values.append(np.mean([float(r["retained_60min_pct"]) for r in rows]))
                auc = np.mean([float(r["auc_cfu_h"]) for r in rows])
                auc_full = np.mean([float(r["auc_cfu_h"]) for r in base])
                effects_auc.append(100*(auc/auc_full-1))
            for ax, points in zip(axes, [values, effects_auc]):
                mean = np.mean(points)
                ax.errorbar(xpos[c], mean,
                            yerr=[[mean-min(points)], [max(points)-mean]],
                            fmt=["o", "s", "D", "^"][j], ms=6, color=COLORS[group],
                            capsize=3, elinewidth=1.2, markeredgecolor="white", markeredgewidth=.6,
                            label=group if c == 0 else None, zorder=4)
            effects.append(dict(
                scenario=scenario, strain=group,
                auc_change_pct_mean=float(np.mean(effects_auc)),
                auc_change_pct_seed_min=float(min(effects_auc)),
                auc_change_pct_seed_max=float(max(effects_auc)),
            ))
    labels = ["Full\nmodel", "No\ngrowth", "No\nattachment", "2× mucus\nspeed"]
    for ax in axes:
        ax.set(xticks=range(4), xticklabels=labels, xlim=(-.55, 3.55))
        ax.grid(axis="y", color="#E6EBEF", lw=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0, pad=9)
    axes[0].set(ylabel="Retained at 60 min / initial dose (%)", ylim=(-2, 66))
    axes[1].set(ylabel="Population AUC change from full model (%)", ylim=(-65, 6))
    axes[1].axhline(0, color="#9DAEB9", lw=1, ls="--", zorder=1)
    title(axes[0], "A", "Endpoint retention")
    title(axes[1], "B", "Change in 0–60 min exposure")
    axes[0].legend(frameon=False, ncol=4, fontsize=8.5, loc="upper center")
    save(fig, "fig09_best_mechanism_comparison")
    with (REPORTS / "mechanism_effects.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)

    caption3 = (
        "Coupled bacterial transport and retention at the selected spray setting. "
        "Insertion depth is 15 mm, pitch 37.5°, yaw 10°, and full cone angle 10°. "
        "A, free, attached, and total retained populations. B, cumulative departure from "
        "the target region and cumulative net growth. C and D, axial distributions of free "
        "and attached populations in the downstream 20–47 mm of the 47 mm mucus strip; "
        "x = 47 mm is the outlet. Free cells are integrated over mucus thickness; both "
        "maps show CFU per axial millimeter. Each heatmap has its own linear color scale, "
        "with darker colors indicating larger populations. Lp-NAC supplies the representative "
        "growth input. Each seed uses its original deposition field and dose; the emitted dose "
        "is 10^6 CFU with survival fraction one. The fitted Lp-NAC curve drives each spray "
        "seed; curves and heatmaps show the three-seed mean, and bands show the full seed range. "
        "Bands describe Monte Carlo variation across the three spray seeds. "
        "Growth and outflow are accumulated separately, preserving initial population plus "
        "growth equals retained population plus departure. Nasal dynamics are model scenarios "
        "with culture-relative growth transfer coefficient one and culture phase zero."
    )
    caption4 = (
        "Mechanistic comparisons share the selected spray setting and each seed's initial "
        "deposition field. A, population retained at 60 min relative to the initial local dose. "
        "B, percentage change in the 0–60 min population AUC relative to the full model, "
        "paired within each spray seed. Full model: attachment Da = 1, mucus speed 1 mm/min, "
        "and fitted relative growth. No growth: growth transfer coefficient zero. "
        "No attachment: attachment Da = 0 with zero initial attached population. "
        "2× mucus speed: 2 mm/min, with all other settings unchanged. "
        "Each point is the mean over three spray seeds using one fitted curve per strain; "
        "whiskers show the full seed range. "
        "All strains share transport and attachment parameters. Direct clearance of attached "
        "bacteria is zero. These comparisons evaluate separate parameter scenarios "
        "under shared transport settings and strain-specific culture growth inputs."
    )
    for number, caption in [(3, caption3), (4, caption4)]:
        (REPORTS / f"result{number}_caption_EN.md").write_text(caption+"\n")

    def m(scenario, group, metric):
        return float(summary[scenario, group][metric+"_mean"])

    nac_full = m("full", "Lp-NAC", "retained_60min_cfu")
    attach_fraction = 100*m("full", "Lp-NAC", "attached_60min_cfu")/nac_full
    text = r"""
\subsection{Predicted redistribution and local retention at the selected spray setting}
\label{sec:retention-results}
\label{sec:coupling-results}
The biological calculations start from the three original deposition fields at
15~mm insertion and \(37.5^\circ\) pitch. The mean initial target population is
""" + f"{config['initial_cfu_mean']:,.0f}" + r"""~CFU per single-strain scenario,
for an emitted dose of \(10^6\,\mathrm{CFU}\) and survival fraction one.
Figure~\ref{fig:best-dynamics} uses Lp-NAC as the representative growth input.
Free bacteria migrate toward the outlet and exchange reversibly with the
attached population. The spatial panels follow axial cell populations in the
local mucus strip using axial transport and reversible attachment.

At 60~min, the Lp-NAC scenario retains
""" + f"{nac_full:,.0f}" + r"""~CFU, comprising
""" + f"{m('full','Lp-NAC','free_60min_cfu'):,.0f}" + r""" free and
""" + f"{m('full','Lp-NAC','attached_60min_cfu'):,.0f}" + r""" attached CFU.
The retained population is
""" + f"{m('full','Lp-NAC','retained_60min_pct'):.2f}" + r"""\% of the initial local dose,
and """ + f"{attach_fraction:.2f}" + r"""\% of the retained population is attached.
Cumulative departure is """ + f"{m('full','Lp-NAC','outflow_60min_cfu'):,.0f}" + r"""~CFU
and cumulative net growth is """ + f"{m('full','Lp-NAC','growth_added_60min_cfu'):,.0f}" + r"""~CFU.
Growth therefore partly offsets loss, while the retained population remains below
the initial deposited dose. The balance is
\(N_{\mathrm{ret}}+N_{\mathrm{out}}=N_0+N_{\mathrm{growth}}\);
departure and net growth are separate cumulative balance terms.
The predicted persistence uses reversible, unsaturated attachment, a zero
direct-clearance term for attached bacteria and full transfer of the
culture-relative growth drive.

\begin{figure}[htbp]\centering
\includegraphics[width=\linewidth]{figures/fig08_best_coupled_dynamics.pdf}
\caption{\textbf{Coupled dynamics at the selected spray setting.}
A, free, attached, and total retained Lp-NAC populations.
B, cumulative departure and net growth.
C and D, axial free and attached populations in the downstream 20--47~mm
of the 47~mm mucus strip; the outlet is at 47~mm.
Both maps show CFU per axial millimeter and use separate linear color scales.
All panels use 15~mm insertion, \(37.5^\circ\) pitch, \(10^\circ\) yaw,
and a \(10^\circ\) full cone.
The fitted Lp-NAC curve drives every spray seed; lines and maps are the
three-seed mean, and shaded bands show the full seed range.
These ranges quantify Monte Carlo variation across the three spray seeds.
The emitted dose is \(10^6\,\mathrm{CFU}\), survival fraction one,
\(\mathrm{Da}_{\mathrm{att}}=1\), \(U=1\,\mathrm{mm\,min^{-1}}\),
\(\eta=1\), and \(\tau_0=0\).}
\label{fig:best-dynamics}
\end{figure}
\FloatBarrier

Table~\ref{tab:best-strains} compares the four growth inputs under the same
transport and attachment settings. These are separate equal-emitted-dose
scenarios. Their population differences arise from the fitted relative growth
inputs; all four strain scenarios share the transport and attachment parameters.
\begin{table}[htbp]\centering\small
\caption{\textbf{First-hour model outputs at the selected spray setting.}
Means over three spray seeds using one fitted curve per strain.
AUC is in \(\mathrm{CFU}\cdot\mathrm h\).}
\label{tab:best-strains}
\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrr}
\toprule
Growth input & Free at 60 min & Attached at 60 min & Retained at 60 min & AUC\\
\midrule
"""
    for group in GROUPS:
        values = [m("full", group, key) for key in [
            "free_60min_cfu", "attached_60min_cfu", "retained_60min_cfu", "auc_cfu_h"]]
        text += group + " & " + " & ".join(f"{v:,.0f}" for v in values) + r"\\" + "\n"
    text += r"""\bottomrule
\end{tabular*}
\end{table}
\FloatBarrier

Because transport and exchange coefficients are shared, the instantaneous free
and attached fields differ between growth inputs only through the growth factor
\(G_i(t)\) in Eq.~\ref{eq:growth-factor}. Strain comparisons therefore report
growth-driven differences under the shared nasal transport and adhesion settings.
"""
    (REPORTS / "results_sections_EN.tex").write_text(text)
    print("Saved figure assets and English Results 3; mechanism comparison excluded from the manuscript.", flush=True)


if __name__ == "__main__":
    main()

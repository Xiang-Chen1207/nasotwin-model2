"""Export the completed 1 mm by 2.5 degree spray simulation grid."""
from pathlib import Path

import os as _runtime_os
from pathlib import Path as _RuntimePath
_runtime_root = _RuntimePath(__file__).resolve().parents[1] / ".build" / "runtime"
for _key, _folder in {"TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
                      "MPLCONFIGDIR": "mpl", "XDG_CACHE_HOME": "cache",
                      "PYVISTA_USERDATA_PATH": "pyvista"}.items():
    _cache = _runtime_root / _folder
    _cache.mkdir(parents=True, exist_ok=True)
    _runtime_os.environ.setdefault(_key, str(_cache))
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np


BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / "spray_boundary_pilot"
OUT = BASE / "spray_depth_angle_grid_0_15mm"


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    global SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true",
                        help="Redraw the existing grid without rewriting simulation data.")
    parser.add_argument("--max-depth", type=int, default=15)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    SOURCE = args.source_dir
    out = args.output_dir
    if args.plot_only:
        with (out / "deposition_grid.csv").open() as stream:
            plot_grid(list(csv.DictReader(stream)), out)
        return
    depths = np.arange(0, args.max_depth + 1)
    angles = np.arange(30, 50.01, 2.5)
    cfg = json.loads((SOURCE / "scan_config.json").read_text())
    expected = {(int(d), float(a), s)
                for d in depths for a in angles for s in cfg["seeds"]}
    rows = [json.loads(line) for line in
            (SOURCE / "seed_results.jsonl").read_text().splitlines() if line.strip()]
    rows = sorted(
        (r for r in rows if (r["depth_mm"], r["pitch_deg"], r["seed"]) in expected),
        key=lambda r: (r["depth_mm"], r["pitch_deg"], r["seed"]))
    keys = [(r["depth_mm"], r["pitch_deg"], r["seed"]) for r in rows]
    assert len(keys) == len(expected) and set(keys) == expected
    assert all(r["parcels"] == cfg["parcels"] == 10000 for r in rows)

    with (SOURCE / "deposition_grid.csv").open() as stream:
        groups = [r for r in csv.DictReader(stream)
                  if int(r["depth_mm"]) in depths and float(r["pitch_deg"]) in angles]
    groups.sort(key=lambda r: (int(r["depth_mm"]), float(r["pitch_deg"])))
    assert len(groups) == len(depths) * len(angles)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "seed_results.csv", rows)
    write_csv(out / "deposition_grid.csv", groups)
    cfg = {k: v for k, v in cfg.items() if k not in
           {"coarse_pitch_deg", "refine_pitch_step_deg", "refine_margin_deg"}}
    cfg.update(depth_min_mm=0, depth_max_mm=args.max_depth, depth_step_mm=1,
               pitch_min_deg=30, pitch_max_deg=50, pitch_step_deg=2.5)
    cfg["depth_scope"] = (
        "Configured virtual-nozzle analysis range; physical device fit and insertion safety require separate measurements."
    )
    (out / "scan_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    plot_grid(groups, out)
    best = max((r for r in groups if r["status"] == "positive"),
               key=lambda r: float(r["mean_pct"]))
    counts = {s: sum(r["status"] == s for r in groups)
              for s in ["positive", "observed_zero", "infeasible"]}
    print(json.dumps({"settings": len(groups), "seed_records": len(rows),
                      "counts": counts, "output_directory": str(out),
                      "best_depth_mm": int(best["depth_mm"]),
                      "best_pitch_deg": float(best["pitch_deg"]),
                      "best_mean_pct": float(best["mean_pct"])}, indent=2))


def plot_grid(groups, out):
    """Draw the original layout with larger deposition fractions shown darker."""
    depths = np.array(sorted({int(r["depth_mm"]) for r in groups}))
    angles = np.array(sorted({float(r["pitch_deg"]) for r in groups}))
    lookup = {(int(r["depth_mm"]), float(r["pitch_deg"])): r for r in groups}
    best = max((r for r in groups if r["status"] == "positive"),
               key=lambda r: float(r["mean_pct"]))
    values = np.array([[float(lookup[d, a]["mean_pct"])
                        if lookup[d, a]["status"] != "infeasible" else np.nan
                        for d in depths] for a in angles])
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(11.3, 5.0))
    fig.subplots_adjust(left=.09, right=.87, bottom=.18, top=.97)
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad("#CCD5DE")
    upper = max(.5, np.ceil(float(best["mean_pct"]) * 2) / 2)
    im = ax.pcolormesh(
        np.r_[depths - .5, depths[-1] + .5],
        np.r_[angles - 1.25, angles[-1] + 1.25], values,
        cmap=cmap, norm=Normalize(0, upper), shading="flat",
        edgecolors=(1, 1, 1, .28), linewidth=.3)
    for row in groups:
        if row["status"] == "observed_zero":
            ax.add_patch(Rectangle(
                (int(row["depth_mm"]) - .5, float(row["pitch_deg"]) - 1.25),
                1, 2.5, facecolor="#F5F7FA",
                edgecolor=(1, 1, 1, .6), linewidth=.3))
    ax.scatter(float(best["depth_mm"]), float(best["pitch_deg"]), marker="*",
               s=210, c="#FFAC39", edgecolors="white", linewidths=1.2, zorder=5)
    ax.set(xlabel="Insertion depth (mm)", ylabel="Pitch angle (°)",
           xticks=depths[::1 if len(depths) <= 21 else 5], yticks=angles,
           xlim=(depths[0]-.5, depths[-1]+.5), ylim=(angles[0]-1.25, angles[-1]+1.25))
    cax = fig.add_axes([.90, .18, .022, .79])
    cb = fig.colorbar(im, cax=cax, ticks=np.linspace(0, upper, 5))
    cb.set_label("Deposition (%)", labelpad=10)
    ax.legend(
        handles=[Patch(facecolor="#F5F7FA", edgecolor="#AEBBC8",
                       label="Zero hits"),
                 Patch(facecolor="#CCD5DE", label="Infeasible")],
        loc="upper left", bbox_to_anchor=(0, -.15), borderaxespad=0,
        ncol=2, frameon=False, fontsize=9)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(out / f"deposition_heatmap_1mm_2p5deg.{ext}",
                    dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Updated heatmap exports: PNG, SVG, PDF")


if __name__ == "__main__":
    main()

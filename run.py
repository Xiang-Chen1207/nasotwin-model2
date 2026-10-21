#!/usr/bin/env python3
"""Run a NasoTwin workflow from the repository root."""
from pathlib import Path
import argparse
import os
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
WORKFLOWS = {
    "metabolism": ("modules/metabolic_exchange/run_model.py", (),
                   "Fit metabolic endpoints and generate scenario figures"),
    "drug-endpoints": ("modules/drug_delivery/fit_endpoint_models.py", (),
                       "Analyze ELISA, JC-1 and qPCR endpoints"),
    "drug-pkpd": ("modules/drug_delivery/pkpd_continuous.py",
                  ("--config", str(ROOT / "modules/drug_delivery/pkpd_continuous_illustration.json")),
                  "Run the continuous-supply PK-PD illustration"),
    "drug-figures": ("modules/drug_delivery/wiki_figures/build_figures.py", (),
                     "Draw drug figures from supplied result tables"),
    "optimize": ("modules/integrated_optimization/run_optimization.py", (),
                 "Run the integrated grid and deterministic scenarios"),
    "ga": ("modules/integrated_optimization/run_ga_comparison.py", (),
           "Run genetic searches using the saved integrated baseline"),
    "optimization-figures": ("modules/integrated_optimization/make_figures.py", (),
                             "Draw integrated figures from saved results"),
    "nasal-coupling": ("modules/nasal/scripts/run_best_coupled_results.py", (),
                       "Compute bacterial coupling from supplied spray trajectories"),
    "nasal-growth": ("modules/nasal/scripts/run_best_coupled_results.py", ("--prepare-fits",),
                     "Fit the four strain-level OD600 trajectories"),
    "nasal-growth-figures": ("modules/nasal/scripts/gen_fig3_growth.py", (),
                             "Draw Wiki N2e growth curves and calculate culture metrics"),
    "nasal-grid-figure": ("modules/nasal/scripts/export_spray_grid.py",
                          ("--source-dir", str(ROOT / "modules/nasal/outputs/spray_scan"),
                           "--output-dir", str(ROOT / "modules/nasal/outputs/spray_grid")),
                          "Draw Wiki N2d from the spray-deposition table"),
    "nasal-plots": ("modules/nasal/scripts/plot_best_coupled_results.py", (),
                    "Plot saved nasal coupling results"),
    "nasal-scan": ("modules/nasal/simulation/anatomical/nc04_boundary_scan.py",
                   ("--config", str(ROOT / "modules/nasal/configs/final_scan.json")),
                   "Run a fresh spray scan; external NC04 geometry required"),
}


def main():
    descriptions = "\n".join(f"  {name:22} {item[2]}" for name, item in WORKFLOWS.items())
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=descriptions + "\n\nFurther arguments are passed to the selected script."
        " Relative paths are resolved from the repository root.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("workflow", choices=WORKFLOWS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    runtime = ROOT / ".runtime"
    env = os.environ.copy()
    for variable, subdirectory in (
        ("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp"),
        ("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", "cache"),
    ):
        directory = runtime / subdirectory
        directory.mkdir(parents=True, exist_ok=True)
        env[variable] = str(directory)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    script, defaults, _ = WORKFLOWS[args.workflow]
    result = subprocess.run(
        [sys.executable, "-B", str(ROOT / script), *defaults, *args.arguments],
        cwd=ROOT,
        env=env,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

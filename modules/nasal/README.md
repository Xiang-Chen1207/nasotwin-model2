# Nasal deposition and bacterial dynamics

Implementation of the [NasoTwin nasal deposition and bacterial dynamics chapter](https://2026.igem.wiki/sustech-med/model-nasotwin#nasal-deposition-bacterial-dynamics).

This module connects NC04 spray deposition, mucus transport, reversible bacterial
attachment, and culture-derived relative growth. Droplets follow drag and gravity
through a CT-derived nasal surface in still air. First-hit intersections determine
the deposited liquid fraction within fixed candidate olfactory ROI coordinate
envelopes. These candidate regions use the recorded geometric coordinates and boundary
rules. Liquid-to-CFU conversion uses uniform bacterial concentration
and the configured survival fraction.

Deposited parcel weights initialize a local mucus strip with the same ROI wall
area. FiPy assembles anisotropic diffusion and upwind advection; a SciPy sparse
matrix exponential advances mobile and attached populations together with
clearance and exposure integrals. The growth model fits one 47-point trajectory
for each of WT, Lp-NAC, Lp-SFN, and Lp-Q over 0–23.5008 h: four fitting units.
Each fitted trajectory drives all three spray seeds. Outputs retain the strain
and seed axes for downstream exposure and allocation calculations.

## Environment

Use Python 3.10 or newer. Install `requirements.txt` for growth fitting, coupling,
and numerical plots. Install `requirements-3d.txt` for spray computation and
geometry rendering. Spray sweeps use CPU workers; rendering uses OpenGL/EGL.

Run the commands below from this module directory:

```bash
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/.build/tmp"
export XDG_CACHE_HOME="$PWD/.build/cache"
export MPLCONFIGDIR="$PWD/.build/mpl"
export PIP_CACHE_DIR="$PWD/.build/pip"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$PIP_CACHE_DIR"
python -m pip install -r requirements.txt
```

## Inputs

Solver configurations are provided in `simulation/config.json`,
`simulation/anatomical/assets/nc04/{olfactory_roi,optimization_config}.json`,
`configs/final_scan.json`, and the two `scan_config.json` files under the spray
grid directories. Supply the growth measurements, geometry, trajectory archives,
and saved results at the paths below, or use the supported command-line options.
The workflow generates fitted parameters and result files from those inputs.

| Input | Default path | Used by |
|---|---|---|
| Growth workbook | `data/growth_curve.xlsx` | Growth fitting and growth plots |
| Transport and biology parameters | `simulation/config.json` | Mucus coupling |
| ROI bounds and spray parameters | `simulation/anatomical/assets/nc04/olfactory_roi.json` | Spray and mucus transfer |
| Scan configuration | `configs/final_scan.json` | Depth–angle scan |
| Selected spray trajectories | `spray_boundary_pilot/trajectories/d+15_p037p5_seed{7717,8831,9941}.npz` | Selected-spray coupling |
| Growth-fit parameters | `outputs/coupling/growth_fits.csv` | Selected-spray coupling |
| Coupled result arrays and tables | `outputs/coupling/` | Coupled-result plots |
| Final deposition grid | `spray_depth_angle_grid_0_15mm/deposition_grid.csv` | Grid plot |
| Cone comparison tables | `data/reference_retention/cone{10,20}/` | Retention and growth comparison plots |

The growth workbook uses `Sheet1`, rows 1–48 and columns A–K. Row 1 contains
headers; rows 2–48 contain the 47 observations. Column A supplies culture time
in hours, and the raw OD columns have headers `WT`, `Lp-NAC`, `Lp-SFN`, and
`Lp-Q`. Each corresponding relative-OD series is six columns to the right of
its raw OD column. Set `NASOTWIN_GROWTH_WORKBOOK` to use another workbook path
for the growth plots, or pass `--workbook PATH` to the fitting command.

Growth-fit CSV columns are `strain`, `record`, `b0`, `r_h_inv`, `k_od`, and
`lambda_h`, followed by observation count and time-window metadata. It contains
one row for each strain. The workbook's relative-OD series supplies the observed
endpoint ratio; the raw OD series supplies endpoint OD and trapezoidal OD-AUC.
Full-window and 0–6 h integrals use the observed samples, with linear interpolation
at the 6 h boundary. `culture_metrics.csv` reports these values, comparisons
against WT over the same windows, R², and RMSE.

The coupling reads NPZ arrays `roi_names`, `roi_hit_mask`, `positions_native_mm`,
`parcel_volume_weight`, and `roi_area_m2`. The ROI JSON supplies `bounds_mm.standard`;
the simulation JSON supplies the `mucus` and `biology` parameter dictionaries.

Supply the NC04 STL through `--geometry`, `NASOTWIN_NC04_GEOMETRY`, or
`simulation/anatomical/assets/nc04/nc04-surface.stl`. Preserve its native millimetre
coordinates. Geometry source: Kiao Inthavong, RMIT University (2021), version 2,
[NC04 dataset](https://doi.org/10.25439/rmt.13550162.v2), CC BY-NC 4.0.

## Growth fitting and coupling

```bash
python scripts/run_best_coupled_results.py --prepare-fits
python scripts/run_best_coupled_results.py
python scripts/plot_best_coupled_results.py
```

Fitting and coupling write to `outputs/coupling/`. The coupling command also accepts
`--output-dir`, `--trajectory-dir`, `--config`, and `--roi-config`. Its default
parameterization uses a 120 × 12 mucus grid, four mechanism scenarios, four growth
curves, three spray seeds, and a five-second output grid over 60 minutes.

The fitted curve uses adjusted culture time
`A(tau) = tau - lambda * (1 - exp(-tau/lambda))` and
`B(tau) = K / (1 + (K/B0 - 1) * exp(-r*A(tau)))`.
Nonlinear least squares estimates `r`, `K`, and `lambda`, with `B0` fixed at the
first observed OD and a startup lower bound of `1e-8 h`. For coupling, startup
values at or below `1e-6 h` use `A(tau) = tau` and `dA/dtau = 1`, including
at time zero. The growth multiplier is `B(t/3600)/B0`, and the growth drive is
`r * (dA/dtau) * (1 - B/K) / 3600` in inverse seconds. This initial-rate treatment
also enters the cumulative growth integral. The reference emitted dose is
`1e6 CFU` per strain, with survival one, mucus velocity `1 mm/min`, growth-transfer
coefficient one, and culture phase zero.

`coupled_results.npz` stores mobile and attached fields, outflow, growth-added
counts, and exposure integrals. `metrics_by_record_seed.csv` and
`metrics_summary.csv` provide per-record/per-seed and aggregated endpoints;
`results_summary.json` records the model settings and four-curve input metadata.
Population arrays have axes `[time, scenario, strain-record, spray-seed]`, with
reference shape `[721, 4, 4, 3]`. `source_emitted_cfu_per_strain` stores the emitted
dose used to normalize downstream retention; `initial_cfu` stores the deposited
local dose separately. The plotting command reads these four files and writes figures to `figures/` and captions and mechanism
tables to `outputs/coupling_figures/`. It accepts `--output-dir` and `--report-dir`.

## Spray scan

```bash
python -m pip install -r requirements-3d.txt
python simulation/anatomical/nc04_boundary_scan.py \
  --config configs/final_scan.json --geometry /path/to/nc04-surface.stl \
  --workers 4 --coarse-only --output-dir outputs/final_scan
python scripts/export_spray_grid.py --source-dir outputs/final_scan \
  --output-dir outputs/final_grid
python scripts/run_best_coupled_results.py \
  --fits outputs/coupling/growth_fits.csv \
  --trajectory-dir outputs/final_scan/trajectories --output-dir outputs/new_coupling
```

The scan JSON defines the entrance coordinate, yaw, cone angle, characteristic
diameter, speed, depth range/step, `coarse_pitch_deg`, parcels, seeds, flight time,
and maximum timestep. Two-stage scans also define `refine_pitch_step_deg` and
`refine_margin_deg`. A final grid from 0–15 mm and 30–50° at 1 mm / 2.5° spacing
contains 144 settings. Three seeds with 10,000 parcels per setting reproduce the
selected grid specification.

The scanner writes trajectory NPZ files, seed-level results, grouped deposition
fractions, and depth boundaries. Each worker loads the geometry and collision
index. An existing output directory resumes completed cases for the same geometry
and scan configuration. `--coarse-only` runs the configured angle list directly.
The grid exporter reads the scan directory's `scan_config.json`,
`seed_results.jsonl`, and `deposition_grid.csv`.

## Wiki figure mapping

| Wiki figure or table | Entry point | Generated output |
|---|---|---|
| N2d: depth–pitch deposition grid | `scripts/export_spray_grid.py` | `deposition_heatmap_1mm_2p5deg.{png,svg,pdf}` in the selected grid directory |
| N2e: four observed growth curves and Logistic fits | `scripts/gen_fig3_growth.py` | `figures/fig03_growth_fits.{png,svg,pdf}` |
| N2f: selected-spray bacterial dynamics | `scripts/plot_best_coupled_results.py` | `figures/fig08_best_coupled_dynamics.{png,svg,pdf}` |
| Table N2-1: observed culture metrics | `scripts/run_best_coupled_results.py --prepare-fits` | `outputs/coupling/culture_metrics.csv` |
| Fitted strain parameters | `scripts/run_best_coupled_results.py --prepare-fits` | `outputs/coupling/growth_fits.csv` |
| First-hour free, attached, retained, and exposure endpoints | `scripts/run_best_coupled_results.py` | `outputs/coupling/metrics_summary.csv` |

These entries correspond to the [nasal chapter results](https://2026.igem.wiki/sustech-med/model-nasotwin#naso-results).

## Scientific figures

```bash
python scripts/export_spray_grid.py --plot-only
python scripts/gen_fig3_growth.py
python scripts/gen_fig4_windows.py
python scripts/gen_fig5_retention.py
python scripts/gen_fig6_coupling.py
export NASOTWIN_NC04_GEOMETRY=/path/to/nc04-surface.stl
python scripts/make_structure_rotation_gif.py
python scripts/plot_insertion_depth_positions.py
python simulation/anatomical/make_best_grid_gif.py \
  --config spray_depth_angle_grid_0_15mm/scan_config.json
```

The retention comparison scripts read `model_inputs.json` and `retention.csv`
from each cone directory. The spray replay reads the final-grid configuration,
`seed_results.csv`, and `deposition_grid.csv`; it simulates dense trajectories and
renders a three-second sequence. `--simulate-only` and `--render-only` split those
steps. Geometry figures use the ROI configuration and final-grid settings.

## Source layout

- `simulation/mucus.py`: finite-volume transport and reversible attachment.
- `simulation/anatomical/nc04_angles.py`: ROI masks, spray sampling, and trajectories.
- `simulation/anatomical/nc04_boundary_scan.py`: parallel depth–angle scan.
- `simulation/anatomical/nc04_optimization.py`: setting ranking and mucus comparisons.
- `simulation/anatomical/nc04_mucus_bridge.py`: dose-preserving ROI transfer.
- `simulation/anatomical/solve_anatomy.py`: triangle collisions, drag, and integration.
- `simulation/anatomical/prepare_flow.py`: archived velocity interpolation utilities.
- `scripts/`: growth fitting, selected-spray coupling, and figure generation.

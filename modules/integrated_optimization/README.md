# Integrated Optimization

This module implements [NasoTwin Integrated Optimization (M8)](https://2026.igem.wiki/sustech-med/model-nasotwin#integrated-optimization). It minimizes total emitted CFU across Lp-Q, Lp-NH3, Lp-NAC, and Lp-SFN while satisfying three metabolic and four cellular-response constraints.

## Upstream inputs

The three upstream result files are included and read directly. The [nasal](../nasal/README.md), [metabolic-exchange](../metabolic_exchange/README.md), and [drug-delivery](../drug_delivery/README.md) commands refresh them when inputs change. Paths in `config.json` are relative to this module directory.

| Configuration field | Default upstream output | Contents |
|---|---|---|
| `inputs.retention` | `../nasal/outputs/coupling/coupled_results.npz` | Nasal coupling histories with one fitted growth trajectory per strain, coupled to each spray seed. |
| `inputs.metabolic_parameters` | `../metabolic_exchange/results/integrated_parameters.json` | Fitted `gln` and `nitrogen` coefficients, `reference` pools, and `design` thresholds. |
| `inputs.pkpd_parameters` | `../drug_delivery/pkpd_continuous_results/conditional_pkpd_results.json` | Transport settings in `assumed_configuration` and channel calibrations in `conditional_PD_calibrations`. |

`transport.py` selects the nasal `full` scenario and averages mobile and attached populations across spray seeds separately. It divides both components by `source_emitted_cfu_per_strain`, the emitted input before deposition. Each strain has one fitted growth trajectory; the current nasal workflow fits one 47-point culture curve per strain and generates 721 first-hour coupling times. Lp-NH3 uses the WT retention kernel multiplied by `nh3_factor`; NH3-specific retention calibration requires matching measurements.

The [saved grid and scenario results](results/optimization_results.json) and
[GA comparison](results/ga_comparison.csv) accompany the source code and configuration.
The [GA report](results/ga_comparison.json) records the search settings, seeds and
full dynamic constraint checks. The data guide records provenance and checksums.

## Objective and constraints

For strain doses `n_s >= 0`, minimize `sum(n_s)`. The default reference dose is 1,000,000 emitted CFU. Metabolic activity uses the scale `E_ref = reference_emitted_cfu * integral_0^1h h_WT(t) dt`, computed from the current retention kernel by trapezoidal integration.

| Criterion | Default requirement | Evaluation |
|---|---|---|
| Gln removal | `1 - G(1 h)/G0 >= 0.20` | First-hour endpoint |
| Peak total ammonia | `A_peak <= 1.05 * A0` | Maximum over the first hour |
| Endpoint ammonia | `A(1 h) <= A0` | First-hour endpoint |
| E01: NAC–RAW IL-6 | Response >= 0.20 | 24 h |
| E02: NAC–RAW TNF-α | Response >= 0.20 | 24 h |
| E03: SFN–RAW IL-6 | Response >= 0.20 | 24 h |
| E04: SFN–BV2 IL-6 | Response >= 0.20 | 24 h |

Metabolic thresholds and starting pools come from the metabolic export. Response targets and evaluation time are set in `config.json`.

For each drug, the source is `R(t) = q * emitted_CFU * h(t)` during the first hour, followed by zero new production. PK-PD dynamics continue to the evaluation time. The inverse calculation takes the largest required amplitude across that drug's channels. With a null configured productivity, `q_ref = nominal_required_amplitude / reference_emitted_cfu`; a positive value sets productivity directly in nmol/(CFU*h). The productivity boundary is `q_required(n) = required_amplitude / n`. The default reference productivity is normalized to target demand; enter measured strain productivity to update the dose allocation.

## Search

The metabolic grid covers 0.25–6 reference doses in increments of 0.05 and Q fractions from 0 to 1 in increments of 0.025. The selected metabolic allocation is combined with the NAC and SFN supply requirements. Scenario analysis applies the configured activity, retention, exposure, and production multipliers and finds a common allocation across those scenarios.

The nominal genetic search uses four continuous strain doses, each bounded by zero and the feasible grid solution's total. Its settings match the Wiki: 96 individuals, 200 generations, seeds 2026/2027/2028, binary tournaments, and elitist survival. Simulated binary crossover uses index 15, pair probability 0.9, and variable probability 0.5. Polynomial mutation uses index 20 and variable probability 0.25. Feasible candidates are ranked by total input; the remaining candidates are ranked by summed normalized constraint violation.

A `1e-6` inward normalized margin is applied during search. Each seed's selected candidate is evaluated with adaptive metabolic integration, ammonia stationary-point evaluation, and full PK-PD dynamics. Allocations and productivity values are computed from the upstream inputs at runtime.

## Run

Using the included upstream outputs, run from the repository root:

```bash
cd modules/integrated_optimization
mkdir -p build/runtime/tmp build/runtime/mpl build/runtime/cache
export TMPDIR="$PWD/build/runtime/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export MPLCONFIGDIR="$PWD/build/runtime/mpl"
export XDG_CACHE_HOME="$PWD/build/runtime/cache"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python3 -B run_optimization.py
python3 -B run_ga_comparison.py
```

The first command writes grid, scenario, and same-budget results and generates figures. The second uses the saved nominal grid solution for continuous search. Rerun both commands after changing upstream inputs or model configuration. To redraw generated results, run `python3 -B make_figures.py`.

## Wiki and output mapping

| Wiki element | Implementation | Generated output |
|---|---|---|
| N5a: four-strain integration workflow | `make_figures.py::_workflow` | `figures/01_integration_workflow.*` |
| N5b: equal versus grid allocation | `run_optimization.py` and `make_figures.py::_dynamics` | `results/equal_budget_comparison.csv`, metabolic/drug trajectory CSVs, `figures/04_equal_budget_dynamics.*`; response panels use E02 and E03. |
| N5c: required productivity | `pkpd.py::required_amplitude` and `make_figures.py::_productivity` | `results/supply_requirements.csv`, `figures/03_productivity_requirements.*` |
| Table N5-1: seven-constraint comparison | `run_optimization.py` and `run_ga_comparison.py` | `results/optimization_results.json`, `results/ga_comparison.json`, and comparison CSVs. |
| Table N5-2: independent GA runs | `genetic_algorithm.py::minimize` and `run_ga_comparison.py` | `results/ga_comparison.csv` and `results/ga_convergence.csv`. |

`results/metabolic_scan.csv` and `results/allocation_scan.csv` store grid evaluations. `figures/05_scenario_allocations.*` shows scenario allocations. Figures are exported as PDF, PNG, and SVG. The scripts also generate Markdown summaries and LaTeX table fragments in `results/`.

`transport.retention_from_coupling(mapping)` converts an in-memory nasal-output mapping into per-emitted-CFU kernels. `metabolism.simulate` evaluates one allocation, `metabolism.scan_metrics` evaluates batches, and `pkpd.prepare_unit_response` supplies reusable drug responses.

This module is distributed under the repository's [MIT license](../../LICENSE).

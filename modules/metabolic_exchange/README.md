# Metabolic Exchange — M3–M4

Implementation of the [Metabolic Exchange chapter of the SUSTech-MED Wiki](https://2026.igem.wiki/sustech-med/model-nasotwin#metabolic-exchange).

This module connects Gln consumption, ammonia removal and BCAA accumulation
through endpoint calibration and a five-pool nitrogen-balance model. It
computes strain effects, mixture trajectories, design candidates and parameter
sensitivity.

## Input

Provide a processed assay table at
`../../data/experimental/湿实验结果总表.md`, relative to this module, or set
`source_table` in `config.json` to your own table. Paths in the configuration
are resolved relative to the configuration file.

The parser reads Markdown sections headed `### E05` through `### E08`.
Numeric tables use a `分组` column and replicate columns named `R1`, `R2`,
and so on. Here `n` counts the listed observations; replicate type is unspecified in the supplied table format. Group labels begin with `Wild type` or the strain name (`Lp-Q`
or `Lp-NH3`). `NA` denotes a missing observation.

| Assay | Quantity | Role |
|---|---|---|
| E05 | Residual Gln readout | Endpoint summary |
| E06 | Gln degradation, mM | First-order activity calibration |
| E07 | Total BCAA, mM | Joint ammonia/BCAA calibration |
| E08 | Residual ammonium, mM | Joint ammonia/BCAA calibration |

## Methods

- Fit the shared basal Gln pool and strain-specific effective activities using
  unweighted least squares on E06 observations.
- Calibrate ammonia uptake and BCAA production using E07/E08 group means and
  the configured initial pools and nitrogen-source allocation.
- Solve five-pool dynamics analytically with linear activity mixing.
- Search mixture fraction and relative exposure, minimizing feasible exposure
  before maximizing Gln removal.
- Evaluate one-parameter sensitivity scenarios and report endpoint statistics.

Concentrations are expressed in mM and auxiliary nitrogen pools in mM-N.
Time and exposure are dimensionless. The shared interval and cross-experiment activity transfer use the configured reference assumptions. `config.json` defines the initial pools,
activity-transfer factors, aqueous speciation settings and design targets.

## Wiki equations and reference configuration

| Wiki equation | Implementation in `model.py` |
|---|---|
| (8): Gln endpoint fit | `fit_gln` |
| (9)–(11): ammonia/BCAA calibration | `calibrate_ammonia_bcaa` |
| (12)–(14): coupled metabolite dynamics | `trajectory` |
| (15)–(18): activity mixing and nitrogen allocation | `effective_rates` |
| (19): tracked nitrogen balance | `trajectory`, checked by `functional_check` in `run_model.py` |
| (20)–(21): aqueous speciation | `speciation` |

The reference search uses 41 Lp-Q fractions from 0 to 1 at 0.025 spacing
and 16 exposure levels from 0.25 to 4 at 0.25 spacing: 656 designs.
Feasibility requires Gln removal of at least 20%, peak total ammonia at most
105% of its initial value, and endpoint ammonia at most its initial value.
`ammonia_peak` evaluates both endpoints and any interior stationary point;
`choose_candidate` applies the exposure-first selection rule.

Reference initial pools are 4 mM total ammonia, 0.2 mM BCAA and 10 mM-N
organic nitrogen. Gln starts at the fitted basal concentration plus 15 mM.
Ammonia release per Gln is 1, the inorganic-nitrogen allocation is 0.5,
both activity-transfer factors are 1, and speciation uses 30 °C and pH 6.5.

## Run

From the repository root, after preparing the input table:

```bash
python -B run.py metabolism
python -B run.py metabolism --config modules/metabolic_exchange/config.json
```

The existing `--check` option checks the E05–E08 observation counts, endpoint
recovery, nonnegative pools, nitrogen balance and the zero-exposure limit.

## Outputs

`data/measured_observations.csv` holds the parsed observations. `results/`
contains endpoint summaries, effect sizes, fitted parameters, trajectories,
design scans and sensitivity tables. `results/figures/` contains six figure
sets in PNG, SVG and PDF. Subsequent runs update these output files.

`results/integrated_parameters.json` exports `gln`, `nitrogen`, `reference`
and `design` directly from the current run. The integrated optimizer reads
this file through its `config.json` input settings.

| Wiki figure | Generated figure stem |
|---|---|
| ME1: measured endpoints | `01_measured_endpoints` |
| ME2: calibration and joint phenotype | `02_endpoint_model_and_joint_phenotype` |
| ME3: metabolite trajectories | `03_conditional_dynamics` |
| ME4: mixture/exposure design space | `04_conditional_design_space` |
| ME5: parameter sensitivity | `05_assumption_sensitivity` |
| ME6: supporting Gln assay | `06_supporting_gln_assay` |

The Wiki ME0 mechanism illustration accompanies the model narrative. The
quantitative figure scripts generate ME1–ME6 from the supplied assay table.

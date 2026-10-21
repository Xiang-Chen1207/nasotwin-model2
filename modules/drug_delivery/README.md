# Drug Delivery & Response

This module implements the [NasoTwin Drug Delivery & Response chapter](https://2026.igem.wiki/sustech-med/model-nasotwin#drug-delivery-response), covering molecular supply, parallel transport routes, four cellular response channels, and production requirements. A bacterial activity profile drives the molecular source; transport produces target exposure; the effect compartment and response turnover translate exposure into cytokine readouts.

## Wiki methods and figures

| Wiki content | Implementation |
|---|---|
| Equation (22): bacterial molecular supply | `pkpd_continuous.py`: reference production multiplied by interpolated relative activity |
| Equations (23)–(25): compartment transport and balances | `transport.py`: `simulate` |
| Equation (26): effect compartment and response turnover | `pkpd_continuous.py`: `dynamic_response` |
| Equation (27): steady-state delivery gain | `transport.py`: `steady_state_gain` |
| Equation (28): experimental endpoint calibration | `pkpd_continuous.py`: `fit_dynamic_pd`; `wiki_figures/build_figures.py`: `endpoint_prediction` |
| Equation (29): required source and per-unit productivity | `pkpd_continuous.py`: `required_production` |
| Figure N4a: continuous-production workflow | `wiki_figures/build_figures.py`: `overview` → `model_overview` |
| Figure N4b: four-channel dynamic calibration | `wiki_figures/build_figures.py`: `pd_calibration` → `dynamic_pd_calibration` |
| Figure N4c: continuous-supply trajectories | `pkpd_continuous.py`: `plot_trajectories` → `illustrative_pkpd` |
| Equation (30) and Figure N4d: normalized supply–response relation | `wiki_figures/build_figures.py`: `production_design` → `supply_response_design` |

The four response channels are NAC–RAW IL-6, NAC–RAW TNF-α, SFN–RAW IL-6, and SFN–BV2 IL-6. Calibration converts NAC doses from mM to µM, fixes each channel's stimulated untreated mean, and fits Emax and K to treated-group means at the configured assay time. The endpoint calculation uses 96-point Gauss–Legendre quadrature, a logarithmic search for K, and Emax between zero and one. PK and PD integration use DOP853 with relative tolerance `1e-8`, absolute tolerance `1e-10`, and maximum step `0.1 h` in the continuous-supply entry point.

## Run

Use Python 3.9+ with NumPy, SciPy, and Matplotlib. From the repository root, prepare the experimental table described below, then run:

```bash
# Export experimental tables, static fits, and supplementary endpoint figures.
python -B modules/drug_delivery/fit_endpoint_models.py

# Fit dynamic response channels and simulate the Wiki reference scenario.
python -B modules/drug_delivery/pkpd_continuous.py \
  --config modules/drug_delivery/pkpd_continuous_illustration.json
```

`fit_endpoint_models.py` writes tables and static model parameters to `results/` and four sets of PNG/SVG figures to `figures/`. Its static cytokine curves and JC-1/qPCR summaries provide supplementary endpoint views. Figure N4b uses the dynamic calibration produced by `pkpd_continuous.py`.

The continuous-supply command writes `conditional_pkpd_results.json`, `illustrative_trajectories.csv`, and PNG/SVG trajectory figures to `pkpd_continuous_results/`. The JSON contains `assumed_configuration` and `conditional_PD_calibrations`, which also supply the downstream integrated optimization model.

After the two commands above, draw Figures N4a, N4b, and N4d:

```bash
python -B modules/drug_delivery/wiki_figures/build_figures.py
```

The drawing script reads `results/endpoint_values.csv` and `pkpd_continuous_results/conditional_pkpd_results.json`, selects ELISA records for E01–E04, and writes PNG/SVG/PDF figures to `wiki_figures/figures/`.

## External inputs and reference configuration

Provide the external experimental table at `../../data/experimental/湿实验结果总表.md`, relative to this module. The parser reads experiment sections E01–E04 and E09–E19, identified by headings such as `### E01`. Numeric tables use `指标`, `分组`, and `R1`, `R2`, … columns; `NA` marks missing entries. NAC doses use mM and SFN doses use µM. The Wiki figure script uses the exported `experiment`, `role`, `dose`, and `value` columns for E01–E04 directly from the endpoint table.

The included `pkpd_continuous_illustration.json` matches the chapter's reference scenario parameter table:

Cellular endpoint calibrations are reported separately for each response channel.
Transport, exposure mapping and timing use configured scenario values; the PK
parameters have reference-scenario status. Static model selection and evaluation
share leave-one-dose-out folds, with errors reported as internal evaluation.

| Setting | Value |
|---|---|
| Reference molecular source; relative activity | 1 nmol/h per molecule; 1 |
| Simulation interval; initial drug amounts | 0–48 h; zero |
| Nasal, plasma, and target volumes | 1 mL each |
| Epithelial and plasma–target exchange capacities | 0.4 and 0.1 mL/h |
| Direct and systemic route fractions | 0.2 and 0.6 |
| Direct transit time; partition coefficient | 1 h; 1 |
| Plasma and target clearance | 0.3 and 0.1 mL/h |
| Nasal loss; exposure scale | 0.1 h⁻¹; 1 |
| Assay duration; effect and response times | 24 h; 2 h and 4 h |
| Conjugation rates | Zero |

SFN/NAC parameters follow `drug_order = ["SFN", "NAC"]`. Alternate configurations can be supplied through `--config`. The source object contains paired `reference_production_nmol_h`, increasing `activity_time_h` knots from zero to `duration_h`, and paired `relative_active_biomass` values at each knot. Dynamic calibration uses `reference_assay_duration_h`, `tau_effect_h`, `tau_response_h`, and paired `brain_free_to_medium_equivalent_scale` values from `pd_assumptions`. The configuration's `status` identifies its scenario type.

## Production design

For constant supply and a stable linear transport model with zero conjugation, `transport.steady_state_gain(parameters)` returns a two-element SFN/NAC gain array in h/mL. With S = PS_pb, the gain is

```text
eta = PS_ep / (PS_ep + k_nasal_loss * V_nose)
G = eta * [S * f_systemic + (S + CL_plasma) * f_direct]
    / [(S + CL_plasma) * CL_brain + S * CL_plasma / K_pb]
```

`pkpd_continuous.required_production(target_reduction, calibration, gain_h_per_mL, exposure_scale=1.0, reference_biomass=None)` implements

```text
R_required = K / (exposure_scale * G) * target_reduction / (Emax - target_reduction)
q_required = R_required / reference_biomass
```

The target is an absolute readout reduction between zero and the channel's fitted Emax. `calibration` supplies `K_medium_uM` and `Emax_infinity`. The returned `required_source_nmol_h` gives molecular supply; providing a positive `reference_biomass` also returns `required_productivity_nmol_per_biomass_h`. Figure N4d expresses the same relation as `rho = exposure_scale * G * R / K` and `E / Emax = rho / (1 + rho)`.

Data and output locations are resolved relative to the scripts. A relative `--config` path is resolved from the working directory. Each run updates its generated outputs, and runtime caches are stored in the script directory's `.runtime/`.

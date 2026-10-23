# Experimental data and model results

The repository includes the experimental inputs and saved outputs used by the
NasoTwin workflow. [manifest.json](manifest.json) records file checksums,
source checksums and the preparation of each published result.

## Included files

| File | Contents and preparation |
|---|---|
| [Experimental table](experimental/湿实验结果总表.md) | E01–E19 from 19 Prism files: 497 numeric entries and 30 source empty cells. R columns retain their precision, order, repeated values and `NA` entries. Descriptive statistics and units follow the source table. |
| [Experimental images](experimental/images/) | 21 source images, linked from the table and verified against its recorded SHA256 hashes. |
| [Growth workbook](../modules/nasal/data/growth_curve.xlsx) | Original workbook, preserved byte for byte. `Sheet1!A1:E48` contains time and four OD600 series; `G1:K48` contains the supplied relative-OD series. Each strain has 47 observations over 0–23.5007805555556 h. |
| [Nasal coupling](../modules/nasal/outputs/coupling/coupled_results.npz) | Saved transport–growth arrays: 721 times at 5 s intervals, four mechanism scenarios, four strain curves and three spray seeds. The original arrays are preserved. `source_emitted_cfu_per_strain` is set to 1,000,000 CFU from the accompanying run summary to support the current optimizer interface. |
| [Nasal run summary](../modules/nasal/outputs/coupling/results_summary.json) | Original spray, transport, growth-transfer and emitted-dose settings associated with the coupling arrays. `parcels_per_seed` counts simulated spray parcels. |
| [Metabolic parameters](../modules/metabolic_exchange/results/integrated_parameters.json) | Exported by the current metabolic runner from E05–E08 and the repository configuration. `gln` and `nitrogen` contain fitted coefficients; `reference` and `design` carry the model's initial pools, activity settings and design criteria. |
| [PK–PD parameters](../modules/drug_delivery/pkpd_continuous_results/conditional_pkpd_results.json) | Exported by the current continuous-supply workflow. Four cellular response channels are fitted to E01–E04; transport, timing and exposure mapping use the reference configuration. |
| [Grid/scenario optimization](../modules/integrated_optimization/results/optimization_results.json) | Current four-strain allocation results using the included upstream outputs. The export records all seven constraints, reference productivities and scenario settings. |
| [GA comparison](../modules/integrated_optimization/results/ga_comparison.csv) and [search report](../modules/integrated_optimization/results/ga_comparison.json) | Grid baseline and three independent genetic searches, with full dynamic checks of each returned candidate. The report records the seeds, search settings and constraint margins. |

## Data definitions

`R1`–`R4` identify Prism source subcolumns. `NA` preserves missing entries, and
`n` counts available numeric entries. Biological/technical replicate metadata,
the E05 unit conversion definition and the recorded assay dates retain their
documented status in the experimental table. The workbook records WT, Lp-NAC,
Lp-SFN and Lp-Q. The optimizer derives Lp-NH3 retention from WT using its
configured multiplier.

PK transport and timing parameters have reference-scenario status. Cellular
PD parameters are endpoint fits, with `independent_validation: false` retained
in the export. Default per-CFU productivities are normalized to target demand;
measured productivity values can be supplied through the optimizer configuration.
Each response channel is evaluated independently under its own constraint.

The coupling NPZ keeps the original simulation arrays and adds the recorded
emitted-dose scalar. Its `initial_cfu` array denotes deposited bacteria; the
emitted-dose scalar provides the denominator for per-emitted-CFU retention.

## Recompute and verify

With the repository's core dependencies installed, run from the repository root:

```bash
python -B run.py metabolism --check
python -B run.py drug-pkpd
python -B run.py optimize
python -B run.py ga
```

The saved coupling arrays support these commands directly. Growth fitting uses
the included workbook. Regenerating spray and coupling outputs uses NC04 geometry
and the spray trajectories described in the nasal module README.

The published snapshot is checked with:

```bash
python -B tools/verify_published_data.py
```

This check covers checksums, experimental table coverage, workbook shape,
coupling axes and population balance, upstream parameter contracts and the
seven constraints for the grid baseline and all three GA candidates. File
checksums identify this publication snapshot; subsequent computations create
new snapshots with their own file checksums.

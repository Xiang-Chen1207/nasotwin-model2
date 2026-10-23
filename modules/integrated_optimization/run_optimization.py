"""Optimize four-strain allocation from retention, metabolism, and PK-PD inputs.

The grid search minimizes emitted CFU under metabolic and response constraints.
Production requirements follow q_required = amplitude_required / emitted_CFU.
Run ``python3 -B run_optimization.py`` from this module directory.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

import metabolism
import pkpd
from transport import load_retention, retained

ROOT = Path(__file__).resolve().parent
STRAINS = ("Q", "NH3", "NAC", "SFN")
DRUG_CHANNELS = {"NAC": ("E01", "E02"), "SFN": ("E03", "E04")}


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def scalar_metrics(result):
    return {
        key: float(result[key]) for key in (
            "gln_removal", "ammonia_peak", "ammonia_end",
            "bcaa_end", "nitrogen_error",
        )
    }


def main():
    cfg = json.loads((ROOT / "config.json").read_text())
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    retention = load_retention(ROOT)
    mp = metabolism.load_parameters(ROOT)
    pp = pkpd.load_parameters(ROOT)
    t = retention["time_h"]
    kernels = {s: retained(retention, s, t) for s in STRAINS}
    ref = cfg["reference_emitted_cfu"]
    exposure_ref = float(np.trapz(retained(retention, "WT", t), t) * ref)
    drug_time = np.unique(np.r_[np.linspace(0, cfg["evaluation_time_h"], 241), 1.0])
    scenarios = cfg["scenarios"]
    if scenarios[0]["name"] != "nominal":
        raise ValueError("The first scenario must be the nominal reference.")
    targets = cfg["targets"]
    units, demands = {}, {}

    # Exposure-factor variations act on kappa; PK coefficients stay unchanged.
    for factor in sorted({s["exposure_factor"] for s in scenarios} | {0.5, 1.0}):
        for drug in DRUG_CHANNELS:
            unit = pkpd.prepare_unit_response(
                drug_time, t, kernels[drug], drug, pp, exposure_factor=factor
            )
            units[(drug, factor)] = unit
            need = pkpd.required_amplitude(
                drug_time, t, kernels[drug], drug, pp,
                {key: targets[key] for key in DRUG_CHANNELS[drug]},
                exposure_factor=factor, unit_response=unit,
            )
            if not need["feasible"]:
                raise ValueError(f"Unattainable dynamic target for {drug}: {need}")
            demands[(drug, factor)] = need
    qref = {
        drug: (
            cfg["productivity_nmol_per_cfu_h"][drug]
            if cfg["productivity_nmol_per_cfu_h"][drug] is not None
            else demands[(drug, 1.0)]["amplitude"] / ref
        )
        for drug in DRUG_CHANNELS
    }
    if any(q <= 0 for q in qref.values()):
        raise ValueError("Specified production capacities must be positive.")
    grid = cfg["metabolic_grid"]
    ratios, doses = np.meshgrid(
        np.arange(0, 1 + grid["ratio_step"] / 2, grid["ratio_step"]),
        np.arange(grid["dose_min"], grid["dose_max"] + grid["dose_step"] / 2,
                  grid["dose_step"]),
    )
    ratios, doses = ratios.ravel(), doses.ravel()
    nq, nn = ref * doses * ratios, ref * doses * (1 - ratios)
    metric_cache, scans = {}, {}
    metabolic_rows = []

    def scan(scenario, q_doses, n_doses):
        return metabolism.scan_metrics(
            t, kernels["Q"], kernels["NH3"] * scenario["nh3_factor"],
            q_doses, n_doses, mp, exposure_ref,
            eta_q=scenario["eta_q"], eta_n=scenario["eta_n"], nu=scenario["nu"],
        )

    def exact(scenario, q_dose, n_dose):
        return metabolism.simulate(
            t, kernels["Q"] * q_dose,
            kernels["NH3"] * scenario["nh3_factor"] * n_dose,
            mp, exposure_ref, eta_q=scenario["eta_q"],
            eta_n=scenario["eta_n"], nu=scenario["nu"],
        )

    for scenario in scenarios:
        cache_key = tuple(scenario[k] for k in ("eta_q", "eta_n", "nh3_factor", "nu"))
        if cache_key not in metric_cache:
            metric_cache[cache_key] = scan(scenario, nq, nn)
        values = metric_cache[cache_key]
        scans[scenario["name"]] = values
        for i in range(len(doses)):
            metabolic_rows.append({
                "scenario": scenario["name"], "r_q": ratios[i], "dose_ref": doses[i],
                **{key: float(values[key][i]) for key in (
                    "gln_removal", "ammonia_peak", "ammonia_end", "bcaa_end")},
                "feasible": int(values["feasible"][i]),
            })
    write_csv(out / "metabolic_scan.csv", metabolic_rows)

    def choose(group, name, label):
        feasible = np.logical_and.reduce([scans[s["name"]]["feasible"] for s in group])
        worst_removal = np.min(
            [scans[s["name"]]["gln_removal"] for s in group], axis=0
        )
        ids = np.flatnonzero(feasible)
        ids = ids[np.lexsort((-worst_removal[ids], doses[ids]))]
        # The search is evaluated at full dynamic resolution for its selected
        # candidate. If a marginal scan point fails, continue in objective order.
        chosen, details = None, None
        for i in ids:
            evaluations = {s["name"]: exact(s, nq[i], nn[i]) for s in group}
            if all(v["feasible"] for v in evaluations.values()):
                chosen, details = i, evaluations
                break
        result = {"name": name, "label": label, "feasible": chosen is not None}
        if chosen is None:
            result["reason"] = "No common feasible point in the specified metabolic grid."
            return result
        allocation = {"Q": float(nq[chosen]), "NH3": float(nn[chosen])}
        for drug in DRUG_CHANNELS:
            allocation[drug] = float(max(
                demands[(drug, s["exposure_factor"])]["amplitude"]
                / (qref[drug] * s["production_factor"]) for s in group
            ))
        result.update({
            "metabolic": {
                "r_q": float(ratios[chosen]), "dose_ref": float(doses[chosen]),
                **scalar_metrics(details[group[0]["name"]]),
            },
            "metabolic_by_scenario": {
                name: scalar_metrics(value) for name, value in details.items()
            },
            "allocations_cfu": allocation,
            "total_cfu": float(sum(allocation.values())),
            "fractions": {s: allocation[s] / sum(allocation.values()) for s in STRAINS},
            "scenario_names": [s["name"] for s in group],
        })
        return result

    solutions = []
    for scenario in scenarios:
        solution = {**scenario, **choose([scenario], scenario["name"], scenario["label"])}
        solutions.append(solution)
    robust = choose(scenarios, "robust_selected", "Selected-scenario robust")
    nominal = solutions[0]
    if not nominal["feasible"]:
        raise ValueError("No nominal feasible allocation within the configured grid.")

    # Source-rate requirements remain useful independently of any q normalization.
    supply_rows = []
    for drug in DRUG_CHANNELS:
        for emitted in np.geomspace(1e4, 1e8, 81):
            supply_rows.append({
                "drug": drug, "allocated_cfu": float(emitted),
                "required_q_nominal": demands[(drug, 1.0)]["amplitude"] / emitted,
                "required_q_lower_exposure": demands[(drug, 0.5)]["amplitude"] / emitted,
            })
    write_csv(out / "supply_requirements.csv", supply_rows)

    # Re-evaluate the metabolic constraints throughout the allocation map.
    # Evaluate the ammonia peak constraint independently at every grid point.
    map_rows = []
    for solution, group in ((nominal, [scenarios[0]]), (robust, scenarios)):
        if not solution["feasible"]:
            continue
        upper = max(8.0, 1.4 * solution["total_cfu"] / ref)
        fractions, totals = np.meshgrid(np.linspace(0.025, 0.975, 39),
                                       np.linspace(0.25, upper, 110))
        fractions, totals = fractions.ravel(), totals.ravel()
        r = solution["metabolic"]["r_q"]
        meta = fractions * totals * ref
        valid = np.ones(len(meta), dtype=bool)
        for scenario in group:
            valid &= scan(scenario, meta * r, meta * (1 - r))["feasible"]
        drug_need = sum(solution["allocations_cfu"][d] for d in DRUG_CHANNELS)
        valid &= (1 - fractions) * totals * ref >= drug_need - 1e-6
        for i in range(len(meta)):
            map_rows.append({
                "scenario": solution["name"], "metabolic_fraction": fractions[i],
                "total_dose_ref": totals[i], "feasible": int(valid[i]),
            })
    write_csv(out / "allocation_scan.csv", map_rows)

    total = nominal["total_cfu"]
    optimized = nominal["allocations_cfu"]
    meta_total = optimized["Q"] + optimized["NH3"]
    strategies = {
        "optimized": optimized,
        "equal_split": {s: total / 4 for s in STRAINS},
        "metabolic_40_60": {
            **optimized, "Q": 0.4 * meta_total, "NH3": 0.6 * meta_total,
        },
    }
    met_rows, drug_rows, comparisons = [], [], []

    def evaluate_drug(scenario, allocation, drug):
        factor = scenario["exposure_factor"]
        return pkpd.simulate(
            drug_time, t, kernels[drug],
            qref[drug] * scenario["production_factor"] * allocation[drug],
            drug, pp, exposure_factor=factor, unit_response=units[(drug, factor)],
        )

    for strategy, allocation in strategies.items():
        met = exact(scenarios[0], allocation["Q"], allocation["NH3"])
        compare = {
            "strategy": strategy, "total_cfu": sum(allocation.values()),
            **scalar_metrics(met), "metabolic_feasible": bool(met["feasible"]),
        }
        for i, hour in enumerate(t):
            met_rows.append({
                "scenario": "nominal", "strategy": strategy, "time_h": hour,
                **{key: met[key][i] for key in ("G", "A", "B")},
            })
        for drug in DRUG_CHANNELS:
            trajectory = evaluate_drug(scenarios[0], allocation, drug)
            for channel, values in trajectory["channels"].items():
                compare[channel] = float(values["effect"][-1])
                # Record the maximum equivalent exposure for each response channel.
                compare[channel + "_Ce_max_uM"] = float(np.max(values["Ce_uM"]))
                for i, hour in enumerate(drug_time):
                    drug_rows.append({
                        "scenario": "nominal", "strategy": strategy, "drug": drug,
                        "channel": channel, "time_h": hour,
                        "source_nmol_h": trajectory["source_nmol_h"][i],
                        "target_uM": trajectory["target_uM"][i],
                        "Ce_uM": values["Ce_uM"][i], "effect": values["effect"][i],
                    })
        compare["all_targets_met"] = bool(
            compare["metabolic_feasible"]
            and all(compare[key] >= target - 1e-7 for key, target in targets.items())
        )
        comparisons.append(compare)
    write_csv(out / "metabolic_trajectories.csv", met_rows)
    write_csv(out / "drug_trajectories.csv", drug_rows)
    write_csv(out / "equal_budget_comparison.csv", comparisons)

    # Check source cutoff, nitrogen balance, and selected-scenario constraints.
    assert comparisons[0]["all_targets_met"], "Selected nominal design fails its targets."
    assert max(c["nitrogen_error"] for c in comparisons) < 1e-6, "Nitrogen imbalance."
    assert all(abs(c["total_cfu"] - total) < 1e-6 for c in comparisons), "Unequal budgets."
    for solution in solutions + [robust]:
        if not solution["feasible"]:
            continue
        for name in solution["scenario_names"]:
            scenario = next(s for s in scenarios if s["name"] == name)
            for drug in DRUG_CHANNELS:
                trajectory = evaluate_drug(scenario, solution["allocations_cfu"], drug)
                assert np.all(trajectory["source_nmol_h"][drug_time > 1] == 0)
                for channel, values in trajectory["channels"].items():
                    assert values["effect"][-1] >= targets[channel] - 1e-7

    result = {
        "status": "REFERENCE_SCENARIO_ALLOCATION",
        "reference": {
            "emitted_cfu": ref, "exposure_ref_cfu_h": exposure_ref,
            "evaluation_time_h": cfg["evaluation_time_h"], "targets": targets,
            "reference_productivity": qref,
            "productivity_mode": {
                drug: ("specified" if cfg["productivity_nmol_per_cfu_h"][drug] is not None
                       else "target_normalized")
                for drug in DRUG_CHANNELS
            },
        },
        "retention": retention["metadata"],
        "drug_requirements": {
            drug: demands[(drug, 1.0)] for drug in DRUG_CHANNELS
        },
        "scenarios": solutions, "robust": robust,
        "equal_budget_comparison": comparisons,
        "assumptions": [
            "Four strains are modeled independently and jointly allocated under the resource and response constraints.",
            "Fixed 15 mm / 37.5 degree spray; source includes deposition already.",
            "Lp-NH3 retention uses the WT reference kernel and the configured retention multiplier.",
            "Equal per-CFU productivity of free and attached retained bacteria.",
            "Metabolic activity is normalized to WT reference CFU-hours.",
            "Metabolic pools represent a closed reference assay compartment with configured initial concentrations.",
            "The modeled drug-source window is the first hour; bacterial persistence is a separate output.",
            "PK, kappa and PD timing use the configured reference values inherited from Model 3.",
            "Targets of 20 percent are configured cellular-endpoint design criteria.",
            "Default q_ref normalizes each drug to its target requirement at the reference dose.",
            "Four PD channels are computed independently, with a separate response constraint for each.",
            "Scenario feasibility is evaluated for the finite set of listed parameter perturbations.",
            "Dose scaling assumes the supplied linear retention model remains applicable.",
        ],
    }
    (out / "optimization_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    write_tables(result, out)
    write_summary(result, out)
    print("End-to-end numerical check passed: selected constraints, source cutoff, nitrogen balance, equal budgets.")
    print(f"Reference-scenario total: {nominal['total_cfu'] / ref:.4f} reference doses")
    print("Nominal fractions: " + ", ".join(
        f"{s}={100 * nominal['fractions'][s]:.2f}%" for s in STRAINS))
    if robust["feasible"]:
        print(f"Selected-scenario robust total: {robust['total_cfu'] / ref:.4f} reference doses")
        print("Robust fractions: " + ", ".join(
            f"{s}={100 * robust['fractions'][s]:.2f}%" for s in STRAINS))
    else:
        print(robust["reason"])
    for drug in DRUG_CHANNELS:
        print(f"{drug} dynamic requirement q*n={demands[(drug, 1.0)]['amplitude']:.6g}"
              f" nmol/h; reference q={qref[drug]:.6g} nmol/(CFU h)")
    for comparison in comparisons:
        print(f"{comparison['strategy']}: Gln removal={comparison['gln_removal']:.5f}, "
              f"ammonia peak/end={comparison['ammonia_peak']:.5f}/{comparison['ammonia_end']:.5f} mM, "
              f"all targets met={comparison['all_targets_met']}")
    import make_figures
    make_figures.main(ROOT)


def write_tables(result, out):
    ref = result["reference"]["emitted_cfu"]
    comparisons = {row["strategy"]: row for row in result["equal_budget_comparison"]}
    equal, selected = comparisons["equal_split"], comparisons["optimized"]
    nominal = next(s for s in result["scenarios"] if s["name"] == "nominal")
    ga_path = out / "ga_comparison.json"
    ga = json.loads(ga_path.read_text(encoding="utf-8")) if ga_path.exists() else None
    targets = result["reference"]["targets"]
    allocation = nominal["allocations_cfu"]
    met = metabolism.load_parameters(ROOT)
    a0 = met["reference"]["initial_total_ammonia_mM"]
    design = met["design"]
    def response_status(row, channels):
        return "均达到" if all(row[k] >= targets[k] - 1e-7 for k in channels) else "未全部达到"
    columns = [
        ("四株等比例", equal, {s: equal["total_cfu"] / 4 for s in STRAINS}),
        ("网格基线", selected, allocation),
    ]
    if ga is not None:
        best = ga["best"]
        columns.append(("GA 最佳可行", {
            **best["metrics"], "total_cfu": best["total_cfu"],
            "all_targets_met": best["all_targets_met"],
        }, best["allocations_cfu"]))
    caption = (
        "四菌分配与达标情况。四株等比例与网格基线采用相同总量；"
        "GA 列为连续搜索得到的较低总量方案，不属于同预算比较。"
        if ga is not None else "同一预算下的四菌分配与达标情况。"
    )
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{" + caption +
        r"用量以参考喷出剂量（$10^6$ CFU）表示；计算采用所列归一化参考比产率与模型参数，约束判定基于未舍入结果。}",
        r"\begin{tabular}{l" + "c" * len(columns) + r"}\toprule",
        "比较项目 & " + " & ".join(label for label, _, _ in columns) + r"\\\midrule",
    ]
    for strain in STRAINS:
        lines.append(f"Lp-{strain} 用量 & " +
                     " & ".join(f"{dose[strain] / ref:.6f}" for _, _, dose in columns) + r"\\")
    metrics = [
        ("总量（参考剂量）", lambda row: f"{row['total_cfu'] / ref:.6f}"),
        (f"Gln 消耗（$\\geq {100 * design['target_gln_removal_fraction']:.0f}\\%$）",
         lambda row: f"{100 * row['gln_removal']:.5f}\\%"),
        (f"氨峰值（$\\leq {a0 * design['max_peak_ammonia_to_initial']:.1f}$ mM）",
         lambda row: f"{row['ammonia_peak']:.5f}"),
        (f"终点氨（$\\leq {a0 * design['max_endpoint_ammonia_to_initial']:.1f}$ mM）",
         lambda row: f"{row['ammonia_end']:.5f}"),
        ("NAC 两个通道要求", lambda row: response_status(row, DRUG_CHANNELS["NAC"])),
        ("SFN 两个通道要求", lambda row: response_status(row, DRUG_CHANNELS["SFN"])),
        ("全部设计要求", lambda row: "全部达到" if row["all_targets_met"] else "未全部达到"),
    ]
    for label, format_value in metrics:
        lines.append(label + " & " + " & ".join(format_value(row) for _, row, _ in columns) + r"\\")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    (out / "result_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if ga is not None:
        from run_ga_comparison import write_latex_report
        write_latex_report(ga, out)

    lines = [r"\begin{table}[htbp]\centering\small",
              r"\caption{配置的参数扰动情景。表中列出代谢活性、保留、暴露映射与生产能力乘数；暴露因子只改变 $\kappa$，生产因子乘在固定参考比产率上。}",
              r"\begin{tabular}{lrrrrr}\toprule",
              r"情景 & $\eta_G$ & $\eta_A$ & NH3保留 & 暴露映射 & 生产能力\\\midrule"]
    labels = {
        "nominal": "参考", "reduced_activity": "氮处理活性下降",
        "lower_exposure": "暴露映射下降", "combined_stress": "组合扰动",
        "robust_selected": "所选情景稳健方案",
    }
    for scenario in result["scenarios"]:
        values = [scenario[key] for key in (
            "eta_q", "eta_n", "nh3_factor", "exposure_factor", "production_factor")]
        lines.append(labels[scenario["name"]] + " & " +
                     " & ".join(f"{value:.2f}" for value in values) + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}",
              r"\begin{table}[htbp]\centering\small",
              r"\caption{各设定情景下的模型分配方案。用量以参考喷出剂量（$10^6$ CFU）表示；参考行保留网格基线，GA 对参考情景另行比较。稳健方案在列出的全部情景中使用同一组分配并满足约束。}",
              r"\begin{tabular}{lrrrrr}\toprule",
              r"情景 & Q & NH3 & NAC & SFN & 总量\\\midrule"]
    for s in result["scenarios"] + [result["robust"]]:
        if s["feasible"]:
            values = [s["allocations_cfu"][key] / ref for key in STRAINS] + [s["total_cfu"] / ref]
            lines.append(labels[s["name"]] + " & " + " & ".join(f"{v:.4f}" for v in values) + r"\\")
        else:
            lines.append(labels[s["name"]] + r" & \multicolumn{5}{c}{搜索网格内无可行点}\\")
    lines += [r"\bottomrule\end{tabular}\end{table}",
              r"\begin{table}[htbp]\centering\scriptsize",
              r"\caption{参考情景下的完整功能输出。总量为参考剂量，氨为参考区室 mM，响应列为24 h降低比例。前三行同预算，GA 行使用其较低总量；四舍五入不改变按原始数值作出的达标判断。40:60 是代谢子群投放比例，不等同原模型二的有效活性比例。}",
              r"\begin{tabular}{lrrrrrrrr}\toprule",
              r"方案 & 总量 & Gln降低 & 氨峰值 & 终点氨 & E01 & E02 & E03 & E04\\\midrule"]
    labels = {"optimized": "网格基线", "equal_split": "四株等比例",
              "metabolic_40_60": "代谢投放40:60"}
    complete_rows = list(result["equal_budget_comparison"])
    if ga is not None:
        labels["ga_best"] = "GA 最佳可行"
        complete_rows.append({
            "strategy": "ga_best", "total_cfu": ga["best"]["total_cfu"],
            **ga["best"]["metrics"],
        })
    for row in complete_rows:
        values = [f"{row['total_cfu'] / ref:.4f}", f"{row['gln_removal']:.3f}", f"{row['ammonia_peak']:.3f}",
                  f"{row['ammonia_end']:.3f}"] + [f"{row[k]:.3f}" for k in ("E01", "E02", "E03", "E04")]
        lines.append(labels[row["strategy"]] + " & " + " & ".join(values) + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (out / "supplementary_tables.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    findings = (
        f"参考条件下，网格基线方案的总需求为 {nominal['total_cfu'] / ref:.2f} 个参考剂量，"
        f"Gln 消耗为 {100 * selected['gln_removal']:.2f}\\%，氨峰值和终点氨分别为 "
        f"{selected['ammonia_peak']:.3f} 与 {selected['ammonia_end']:.3f}\\,\\mathrm{{mM}}。"
    )
    # Use ordinary text units, so the generated paragraph does not need math mode.
    findings = findings.replace(r"\,\mathrm{mM}", " mM")
    findings += (
        f"同总量的等比例方案消耗了 {100 * equal['gln_removal']:.2f}\\% 的 Gln，"
        f"但氨峰值与终点氨分别达到 {equal['ammonia_peak']:.3f} 和 "
        f"{equal['ammonia_end']:.3f} mM。"
        "这一比较评价功能分配同时满足代谢与响应约束的情况。"
    )
    duplicate = comparisons.get("metabolic_40_60")
    if duplicate and all(np.isclose(selected[k], duplicate[k], rtol=1e-8, atol=1e-9)
                         for k in ("gln_removal", "ammonia_peak", "ammonia_end",
                                   "E01", "E02", "E03", "E04")):
        findings += (
            "\n\n网格基线的代谢子群内部投放比例为 $40{:}60$，与采用该投放比例的对照结果重合，"
            "图2合并显示该曲线，完整对照保留在补充表 S3。"
            "这里的投放比例和模型二的有效活性比例分别按各自定义的尺度解释。"
        )
    (out / "main_findings.tex").write_text(findings + "\n", encoding="utf-8")


def write_summary(result, out):
    ref = result["reference"]["emitted_cfu"]
    lines = [
        "# 四菌整合优化：本次计算结果", "",
        "以下模型设计采用所列跨模块归一化、参考比产率与 PK–PD 参数。",
        "默认产能以每种药物在参考喷出剂量下恰好达到本版响应目标定义；两药相等的参考需求来自该归一化约定。",
        "", "| 情景 | 总量/参考剂量 | Q | NH3 | NAC | SFN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in result["scenarios"] + [result["robust"]]:
        if s["feasible"]:
            values = [s["total_cfu"] / ref] + [s["allocations_cfu"][key] / ref for key in STRAINS]
            lines.append("| " + s["label"] + " | " + " | ".join(f"{x:.4f}" for x in values) + " |")
        else:
            lines.append(f"| {s['label']} | 搜索网格内无可行点 | — | — | — | — |")
    lines += ["", f"参考剂量：{ref:g} CFU。参考有效暴露："
              f"{result['reference']['exposure_ref_cfu_h']:.6g} CFU·h。", "",
              "| 分子 | 动态所需 q×喷出菌量（nmol/h） | 参考比产率（nmol/(CFU·h)） |",
              "|---|---:|---:|"]
    for drug in DRUG_CHANNELS:
        lines.append(f"| {drug} | {result['drug_requirements'][drug]['amplitude']:.6g}"
                     f" | {result['reference']['reference_productivity'][drug]:.6g} |")
    lines += ["", "参考比产率由目标需求定义；可在 config.json 中输入实测比产率，更新生产需求与分配结果。",
              "某株用量为零时，该参数设置下的功能需求可由其他已建模活动满足；结果用于识别设计中的功能冗余。",
              "", "完整输出及条件说明见 optimization_results.json，连续产能需求见 supply_requirements.csv。",
              "改变原 PD 校准的检测时长、效应室或响应时间常数时，需要重新进行条件性校准。",
              ""]
    (out / "RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

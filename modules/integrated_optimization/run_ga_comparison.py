"""Search four continuous strain doses against the saved nominal grid solution.

Run ``python3 -B run_ga_comparison.py`` after ``run_optimization.py``.
The search uses the saved scenario parameters and reference productivities,
then evaluates each seed's best candidate with the full dynamic model.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

import genetic_algorithm
import metabolism
import pkpd
from run_optimization import DRUG_CHANNELS, STRAINS, scalar_metrics, write_csv
from transport import load_retention, retained


ROOT = Path(__file__).resolve().parent
SEEDS = (2026, 2027, 2028)
POPULATION_SIZE = 96
GENERATIONS = 200
# Apply an inward constraint margin for the batch search before final
# evaluation with the adaptive ODE integrator.
SEARCH_MARGIN = 1e-6


def main():
    started = perf_counter()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    saved = json.loads(
        (ROOT / "results" / "optimization_results.json").read_text(encoding="utf-8")
    )
    reference = saved["reference"]
    scenario = next(s for s in saved["scenarios"] if s["name"] == "nominal")
    ref = reference["emitted_cfu"]
    exposure_ref = reference["exposure_ref_cfu_h"]
    targets = reference["targets"]
    qref = reference["reference_productivity"]
    grid = cfg["metabolic_grid"]
    retention = load_retention(ROOT)
    mp = metabolism.load_parameters(ROOT)
    pp = pkpd.load_parameters(ROOT)
    t = retention["time_h"]
    kernels = {s: retained(retention, s, t) for s in STRAINS}
    drug_time = np.unique(np.r_[
        np.linspace(0, reference["evaluation_time_h"], 241), 1.0
    ])
    units = {
        drug: pkpd.prepare_unit_response(
            drug_time, t, kernels[drug], drug, pp,
            exposure_factor=scenario["exposure_factor"],
        )
        for drug in DRUG_CHANNELS
    }
    design = mp["design"]
    a0 = mp["reference"]["initial_total_ammonia_mM"]
    limits = {
        "gln_removal_min": design["target_gln_removal_fraction"],
        "ammonia_peak_max": a0 * design["max_peak_ammonia_to_initial"],
        "ammonia_end_max": a0 * design["max_endpoint_ammonia_to_initial"],
    }
    baseline_x = np.array([scenario["allocations_cfu"][s] / ref for s in STRAINS])
    # Each nonnegative dose is bounded by the feasible baseline total.
    # Every candidate with a smaller total lies within these bounds.
    upper = np.full(4, baseline_x.sum())
    lower = np.zeros(4)

    def batch_evaluate(x):
        met = metabolism.scan_metrics(
            t, kernels["Q"], kernels["NH3"] * scenario["nh3_factor"],
            ref * x[:, 0], ref * x[:, 1], mp, exposure_ref,
            eta_q=scenario["eta_q"], eta_n=scenario["eta_n"], nu=scenario["nu"],
        )
        meta_total = x[:, 0] + x[:, 1]
        residuals = [
            1 - met["gln_removal"] / limits["gln_removal_min"] + SEARCH_MARGIN,
            met["ammonia_peak"] / limits["ammonia_peak_max"] - 1 + SEARCH_MARGIN,
            met["ammonia_end"] / limits["ammonia_end_max"] - 1 + SEARCH_MARGIN,
            (grid["dose_min"] - meta_total) / grid["dose_max"],
            (meta_total - grid["dose_max"]) / grid["dose_max"],
        ]
        for drug, index in (("NAC", 2), ("SFN", 3)):
            amplitude = qref[drug] * scenario["production_factor"] * ref * x[:, index]
            for key, channel in units[drug]["channels"].items():
                cal = channel["calibration"]
                ce = amplitude[:, None] * channel["unit_quadrature_Ce_uM"].ravel()
                suppression = cal["Emax_infinity"] * ce / (cal["K_medium_uM"] + ce)
                effect = suppression @ channel["endpoint_response_weights"].ravel()
                residuals.append(1 - effect / targets[key] + SEARCH_MARGIN)
        violation = np.maximum(np.column_stack(residuals), 0).sum(axis=1)
        return x.sum(axis=1), violation

    def full_evaluate(x):
        met = metabolism.simulate(
            t, ref * x[0] * kernels["Q"],
            ref * x[1] * kernels["NH3"] * scenario["nh3_factor"],
            mp, exposure_ref, eta_q=scenario["eta_q"],
            eta_n=scenario["eta_n"], nu=scenario["nu"],
        )
        metrics = scalar_metrics(met)
        for drug, index in (("NAC", 2), ("SFN", 3)):
            trajectory = pkpd.simulate(
                drug_time, t, kernels[drug],
                qref[drug] * scenario["production_factor"] * ref * x[index],
                drug, pp, exposure_factor=scenario["exposure_factor"],
                unit_response=units[drug],
            )
            metrics.update({
                key: float(values["effect"][-1])
                for key, values in trajectory["channels"].items()
            })
        feasible = bool(
            met["feasible"]
            and np.all(x >= lower) and np.all(x <= upper + 1e-10)
            and grid["dose_min"] <= x[0] + x[1] <= grid["dose_max"]
            and all(metrics[key] >= target - 1e-10 for key, target in targets.items())
        )
        return {
            "allocations_ref": dict(zip(STRAINS, map(float, x))),
            "allocations_cfu": dict(zip(STRAINS, map(float, ref * x))),
            "fractions": dict(zip(STRAINS, map(float, x / x.sum()))),
            "total_ref": float(x.sum()),
            "total_cfu": float(ref * x.sum()),
            "metabolic_q_fraction": float(x[0] / (x[0] + x[1])),
            "metrics": metrics,
            "all_targets_met": feasible,
        }

    baseline = full_evaluate(baseline_x)
    if not baseline["all_targets_met"]:
        raise RuntimeError("Saved baseline is no longer feasible; rebuild the main model first.")
    print(f"Grid baseline: {baseline['total_ref']:.8f} reference doses.", flush=True)
    runs, history = [], []
    for seed in SEEDS:
        print(f"GA seed {seed}: {POPULATION_SIZE} individuals, {GENERATIONS} generations.",
              flush=True)
        answer = genetic_algorithm.minimize(
            batch_evaluate, lower, upper, seed=seed,
            population_size=POPULATION_SIZE, generations=GENERATIONS,
        )
        checked = full_evaluate(answer["best_x"])
        row = {
            "seed": seed,
            "evaluations": answer["evaluations"],
            "search_violation": answer["best_violation"],
            "algorithm_settings": answer["settings"],
            **checked,
        }
        runs.append(row)
        history.extend({"seed": seed, **step} for step in answer["history"])
        print(
            f"Seed {seed}: total={row['total_ref']:.8f}, "
            f"allocation={row['allocations_ref']}, "
            f"full_dynamics_feasible={row['all_targets_met']}", flush=True,
        )

    accepted = [
        row for row in runs
        if row["search_violation"] <= 0 and row["all_targets_met"]
    ]
    if not accepted:
        raise RuntimeError("No GA candidate passed the full dynamic constraints.")
    best = min(accepted, key=lambda row: row["total_ref"])
    improvement = 100 * (baseline["total_ref"] - best["total_ref"]) / baseline["total_ref"]
    report = {
        "status": "CONDITIONAL_NOMINAL_GA_COMPARISON_NOT_GLOBAL_OPTIMALITY_PROOF",
        "reference": reference,
        "scenario": {k: scenario[k] for k in (
            "name", "eta_q", "eta_n", "nh3_factor",
            "exposure_factor", "production_factor", "nu",
        )},
        "method": {
            "algorithm": "real-coded GA, SBX crossover, polynomial mutation, elitist survival",
            "constraint_handling": "feasibility first; normalized sum of positive violations",
            "variables": list(STRAINS),
            "objective": "sum of four nonnegative emitted strain doses in reference units",
            "seeds": list(SEEDS),
            "population_size": POPULATION_SIZE,
            "generations": GENERATIONS,
            "lower_ref": lower.tolist(),
            "upper_ref": upper.tolist(),
            "metabolic_total_bounds_ref": [grid["dose_min"], grid["dose_max"]],
            "search_inward_normalized_margin": SEARCH_MARGIN,
            "initialization": "independent uniform-random population initialization",
            "drug_variables": "NAC and SFN doses are independent search variables",
            "search_evaluator": "full retention history, midpoint metabolic integration, dynamic PD quadrature",
            "final_evaluator": "adaptive metabolic ODE with ammonia extrema; full PK-PD trajectories",
            "domain_interpretation": (
                "Continuous relaxation of the original metabolic grid within its same "
                "ratio and dose ranges; nonnegative drug doses capped by the feasible "
                "baseline total, which cannot exclude a lower-total solution."
            ),
        },
        "constraints": {**limits, "response_min": targets},
        "baseline": baseline,
        "runs": runs,
        "best": best,
        "reduction_vs_grid_percent": improvement,
        "accepted_seed_total_range_ref": [
            min(row["total_ref"] for row in accepted),
            max(row["total_ref"] for row in accepted),
        ],
        "elapsed_seconds": perf_counter() - started,
        "limitations": [
            "Reference-scenario comparison; stress scenarios and common-feasible allocations are reported separately.",
            "Continuous search refines feasible allocations between grid points; the comparison measures total dose relative to that grid baseline.",
            "Three independent seeds describe numerical search repeatability; the reported result is the best feasible solution found.",
            "Interpret allocations under the listed reference productivity, activity-transfer, WT-based NH3 retention and independent-strain assumptions.",
            "Near-boundary feasible solutions have little margin for unmodeled parameter error.",
        ],
    }
    out = ROOT / "results"
    (out / "ga_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for label, result in [("grid", baseline)] + [
        (f"ga_seed_{row['seed']}", row) for row in runs
    ]:
        rows.append({
            "method": label, "total_ref": result["total_ref"],
            **{f"{s}_ref": result["allocations_ref"][s] for s in STRAINS},
            **result["metrics"], "all_targets_met": result["all_targets_met"],
        })
    write_csv(out / "ga_comparison.csv", rows)
    write_csv(out / "ga_convergence.csv", history)
    write_summary(report, out / "GA_COMPARISON_CN.md")
    print(f"Best total: {best['total_ref']:.8f}; reduction vs grid: {improvement:.4f}%.")
    print(f"Full-dynamics feasible GA runs: {len(accepted)}/{len(runs)}.")
    print("Best full-dynamics metrics:", json.dumps(best["metrics"]))
    print("Best strain fractions:", json.dumps(best["fractions"]))
    print("Baseline full-dynamics metrics:", json.dumps(baseline["metrics"]))
    print(f"Elapsed: {report['elapsed_seconds']:.1f} s. Results: {out / 'GA_COMPARISON_CN.md'}")


def write_latex_report(report, out):
    """Render the saved experiment into report text without rerunning search."""
    method = report["method"]
    best, baseline = report["best"], report["baseline"]
    settings = best["algorithm_settings"]
    seeds = "、".join(str(seed) for seed in method["seeds"])
    dose_min, dose_max = method["metabolic_total_bounds_ref"]
    methods = (
        r"令 $x_s=n_s/(10^6\,\mathrm{CFU})$，以四个连续用量为染色体，最小化 "
        r"$F(\mathbf x)=\sum_s x_s$，同时施加前述三项代谢约束和四项响应约束。"
        f"代谢总量范围沿用网格搜索的 $ {dose_min:g}\\leq x_Q+x_N\\leq {dose_max:g}$，"
        r"代谢两株的比例允许在 $0$--$1$ 内连续变化。"
        f"每个变量均取 $0\\leq x_s\\leq {baseline['total_ref']:.2f}$；"
        "上界等于已知可行网格方案的总量。由于目标为非负用量之和，"
        "这个上界不会排除总量更低的方案。产率、保留核及药代参数固定，"
        "NAC/SFN 用量也参与搜索，不固定为反解下限。\n\n"
        f"采用 NumPy 实现的实数编码遗传算法，每次种群为 {method['population_size']}，"
        f"迭代 {method['generations']} 代，随机种子为 {seeds}。"
        "各次初始种群独立均匀随机生成，不注入网格答案。"
        f"使用模拟二进制交叉（SBX，分布指数 {settings['crossover_eta']:g}，"
        f"每对父代交叉概率 {settings['crossover_probability_per_pair']:g}，"
        f"每变量交叉概率 {settings['crossover_probability_per_variable']:g}）及多项式变异"
        f"（分布指数 {settings['mutation_eta']:g}，每变量概率 "
        f"{settings['mutation_probability_per_variable']:g}），"
        "通过二元锦标赛选择父代，并从父代与子代合集中保留精英。"
        "采用可行性优先排序：可行方案按总量排序；"
        "不可行方案按归一化约束违反量之和排序。\n\n"
        "搜索评价保留全部首小时菌量时序，使用分片代谢积分和动态药效求积。"
        r"代谢与响应约束加入 $10^{-6}$ 的归一化向内余量，"
        "用于减小快速评价和最终积分之间数值误差的影响。"
        "各次最终方案再运行自适应代谢 ODE，包含氨峰驻点检查，"
        "并计算完整 PK--PD 轨迹。归一化向内余量用于控制数值评价与最终积分之间的误差。"
        "比较仅覆盖参考情景，所选不利情景的原网格结果单独保留。\n"
    )
    (out / "ga_methods.tex").write_text(methods, encoding="utf-8")
    ratio = " : ".join(f"{100 * best['fractions'][s]:.3f}" for s in STRAINS)
    metrics = best["metrics"]
    low, high = report["accepted_seed_total_range_ref"]
    passed = sum(
        row["all_targets_met"] and row["search_violation"] <= 0 for row in report["runs"]
    )
    findings = (
        "在相同参数和响应要求下，本次搜索得到的最佳可行方案将总量从网格基线的 "
        f"{baseline['total_ref']:.6f} 降至 {best['total_ref']:.6f} 个参考剂量，"
        f"减少 {report['reduction_vs_grid_percent']:.3f}\\%。"
        "按 Q、NH3、NAC、SFN 的顺序，四株占比约为 "
        f"$ {ratio}$（各项以百分数表示），具体用量见表1。"
        f"该方案来自随机种子 {best['seed']}。\n\n"
        f"最终动态计算给出 Gln 消耗 {100 * metrics['gln_removal']:.6f}\\%，"
        f"氨峰值 {metrics['ammonia_peak']:.6f} mM、"
        f"终点氨 {metrics['ammonia_end']:.6f} mM。"
        f"NAC 的限制通道 E02 与 SFN 的限制通道 E03 响应分别为 "
        f"{metrics['E02']:.8f} 和 {metrics['E03']:.8f}，其余通道也达到要求"
        "（补充表 S3）。约束判定使用未舍入结果；"
        "显示的小数位用于说明计算结果的数值边界。\n\n"
        f"{len(report['runs'])} 次独立搜索中，{passed} 次通过完整动态约束检查；"
        f"其可行总量范围为 {low:.6f}--{high:.6f} 个参考剂量"
        "（补充表 S4）。这些独立运行评价该设置下的数值重复性；"
        "结果报告为搜索得到的最佳可行解。\n"
    )
    (out / "ga_findings.tex").write_text(findings, encoding="utf-8")
    lines = [
        r"\begin{table}[htbp]\centering\small",
        r"\caption{遗传算法的三次独立参考情景搜索。总量单位为参考剂量；Q 占比仅针对 Q 与 NH3 两株代谢子群。各次均从随机种群开始，最终方案用完整动态模型检查。}",
        r"\begin{tabular}{rrrrc}\toprule",
        r"随机种子 & 候选评价次数 & 总量 & 代谢子群 Q 占比 & 全部约束\\\midrule",
    ]
    for row in report["runs"]:
        lines.append(
            f"{row['seed']} & {row['evaluations']} & {row['total_ref']:.8f} & "
            f"{100 * row['metabolic_q_fraction']:.3f}\\% & "
            + ("达到" if row["all_targets_met"] else "未全部达到") + r"\\"
        )
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    (out / "ga_seed_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(report, path):
    old, best = report["baseline"], report["best"]
    lines = [
        "# 四变量遗传算法与原网格方案的比较",
        "",
        f"在相同参考参数下，三个随机种子中通过完整动态约束检查的最低总量为 "
        f"**{best['total_ref']:.6f} 个参考剂量**，原网格方案为 "
        f"{old['total_ref']:.6f}，差值为 {report['reduction_vs_grid_percent']:.3f}%。"
        f"每个参考剂量等于 {report['reference']['emitted_cfu']:,.0f} CFU；"
        "CFU 分配按所列跨模型尺度与归一化参考产率解读。",
        "",
        "## 配比与约束",
        "",
        "| 菌株 | 原网格用量（参考剂量） | GA 用量（参考剂量） | GA 占比 |",
        "|---|---:|---:|---:|",
    ]
    for s in STRAINS:
        lines.append(
            f"| Lp-{s} | {old['allocations_ref'][s]:.6f} | "
            f"{best['allocations_ref'][s]:.6f} | {100 * best['fractions'][s]:.3f}% |"
        )
    lines.extend([
        f"| 总量 | {old['total_ref']:.6f} | {best['total_ref']:.6f} | 100% |",
        "",
        "| 完整动态模型输出 | 原网格方案 | GA 最佳可行方案 | 判据 |",
        "|---|---:|---:|---|",
    ])
    specs = [
        ("Gln 消耗比例", "gln_removal", f"≥ {report['constraints']['gln_removal_min']:.3f}"),
        ("氨峰值（参考区室 mM）", "ammonia_peak", f"≤ {report['constraints']['ammonia_peak_max']:.3f}"),
        ("终点氨（参考区室 mM）", "ammonia_end", f"≤ {report['constraints']['ammonia_end_max']:.3f}"),
    ] + [
        (key + "：24 h 响应", key, f"≥ {value:.3f}")
        for key, value in report["reference"]["targets"].items()
    ]
    for label, key, threshold in specs:
        lines.append(
            f"| {label} | {old['metrics'][key]:.8f} | "
            f"{best['metrics'][key]:.8f} | {threshold} |"
        )
    lines.extend([
        "",
        "## 三次独立搜索",
        "",
        "| 随机种子 | 总量（参考剂量） | Lp-Q / 代谢两株总量 | 完整动态全部达标 |",
        "|---|---:|---:|---|",
    ])
    for row in report["runs"]:
        lines.append(
            f"| {row['seed']} | {row['total_ref']:.8f} | "
            f"{100 * row['metabolic_q_fraction']:.3f}% | "
            f"{'是' if row['all_targets_met'] else '否'} |"
        )
    lines.extend([
        "",
        "## 方法与解释",
        "",
        f"四个连续变量分别是四株菌的用量，目标是最小化四者之和。每次搜索采用 "
        f"{POPULATION_SIZE} 个个体、{GENERATIONS} 代，使用实数编码、模拟二进制交叉 "
        "（SBX）、多项式变异、二元锦标赛和父代/子代合并精英保留。"
        "约束采用可行性优先规则：可行个体优先，其次比较总量；"
        "不可行个体比较归一化约束违反量。初始种群独立随机生成，未注入原方案，"
        "NAC/SFN 两个变量也未固定为反解下限。",
        "",
        "本包以 NumPy 独立实现上述算法。算法结构和约束处理术语可参照 "
        "[pymoo 的 GA 文档](https://pymoo.org/algorithms/soo/ga.html)及"
        "[可行性优先规则](https://pymoo.org/constraints/feas_first.html)；"
        "具体 NumPy 实现的交叉和变异参数保存在各次结果的 "
        "`algorithm_settings` 中。",
        "",
        "代谢总量沿用原网格的上下界，Q 在代谢两株中的占比允许在 0–1 连续变化。"
        "每个变量的上界取原可行方案的总量；由于目标是非负用量之和，"
        "任何更优方案都不会被这个上界排除。连续搜索能落在网格点之间，"
        "因此该比较评价连续搜索相对网格基线的总用量变化。",
        "",
        "搜索保留首小时的完整菌量时序和动态药效积分，代谢快速评价采用原有分片积分。"
        f"生物指标加入 {SEARCH_MARGIN:g} 的归一化向内数值余量；"
        "每个种子的最终方案重新运行自适应代谢 ODE（含氨峰驻点）和完整 PK–PD 轨迹，"
        "表中数值来自该最终计算；向内余量用于控制数值评价与最终积分之间的误差。",
        "",
        "NAC 和 SFN 的用量若都接近一个参考剂量，是参考产率按目标需求归一化的结果；"
        "NAC 与 SFN 的参考比产率分别按各自目标需求归一化，分配按该参考产率体系解读。",
        "",
        "本比较采用参考情景；接近约束边界的解对未建模参数误差的余量较小。"
        "不利情景和共同可行方案在相应表格中报告。三次独立搜索评价数值重复性。"
        "将实测比产率和菌株相互作用参数接入后，可重新评价工程分配方案。",
        "",
        "## 文件与复现",
        "",
        "在仓库根目录运行 `python -B run.py ga`，使用与当前模型一致的基线结果；"
        "若模型参数或上游输入改变，先运行 `python -B run.py optimize` 更新基线。",
        "",
        "- `ga_comparison.json`：完整设置、三次搜索结果、最终约束数值。",
        "- `ga_comparison.csv`：原网格与三次 GA 的对照表。",
        "- `ga_convergence.csv`：每代最优目标、约束违反量和可行个体数。",
        "",
        "结果保存在 `modules/integrated_optimization/results/`。"
        "运行 `python -B run.py optimize` 可读取已有 GA 结果并更新 LaTeX 表格与方法片段；"
        "情景分配结果同时保留。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run the metabolic analysis, conditional scenarios, and scientific figures."""
from pathlib import Path
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
# Store runtime files, font caches and temporary files beside the module.
RUNTIME = ROOT / ".runtime"
for directory in (RUNTIME, RUNTIME / "tmp", RUNTIME / "mpl", RUNTIME / "cache"):
    directory.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(RUNTIME / "tmp")
os.environ["MPLCONFIGDIR"] = str(RUNTIME / "mpl")
os.environ["XDG_CACHE_HOME"] = str(RUNTIME / "cache")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
tempfile.tempdir = str(RUNTIME / "tmp")

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from model import (
    read_observations, endpoint_statistics, group_mean, fit_gln, speciation,
    calibrate_ammonia_bcaa, trajectory, design_scan, choose_candidate,
    scenario_sensitivity,
)

COLORS = {"WT": "#7D8791", "Lp-Q": "#167E87", "Lp-NH3": "#8064A2", "1:1 mixture": "#C88239"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "savefig.facecolor": "white",
})


def save_figure(fig, stem, directory):
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def panel(ax, label):
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, weight="bold", fontsize=11)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def observed_figures(obs, stats, gln, figures):
    contracts = [
        ("E06", 0, "Gln degradation: no supplement", "Gln degraded (mM)"),
        ("E06", 15, "Gln degradation: +15 mM", "Gln degraded (mM)"),
        ("E08", 0, "Residual ammonium", "Reported NH$_4^+$ (mM)"),
        ("E07", 0, "BCAA accumulation", "Total BCAA (mM)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0), layout="constrained")
    for label, ax, (exp, added, title, ylabel) in zip("abcd", axes.flat, contracts):
        data = obs[(obs.experiment == exp) & (obs.added_gln_mM == added)]
        strains = ["WT", "Lp-Q" if exp == "E06" else "Lp-NH3"]
        for x, strain in enumerate(strains):
            values = data[data.strain == strain].value.to_numpy()
            ax.scatter(x+np.linspace(-0.07, 0.07, len(values)), values, s=27,
                       color=COLORS[strain], zorder=3, edgecolor="white", linewidth=0.35)
            ax.errorbar(x, values.mean(), yerr=values.std(ddof=1), fmt="_",
                        color="#222222", capsize=4, markersize=15, linewidth=0.9, zorder=4)
            ax.text(x, 0.97, f"n={len(values)}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7, color="#555555")
        ax.set(xlim=(-0.45, 1.45), xticks=[0, 1], xticklabels=strains,
               title=title, ylabel=ylabel)
        ax.set_ylim(0, data.value.max()*1.25)
        panel(ax, label)
    fig.suptitle("Measured metabolic endpoints", fontsize=11, weight="bold")
    save_figure(fig, "01_measured_endpoints", figures)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.0), layout="constrained")
    x = np.linspace(0, 15, 120)
    for strain, key in [("WT", "lambda_g_wt"), ("Lp-Q", "lambda_g_q")]:
        prediction = (gln["gln_basal_mM"]+x)*(-np.expm1(-gln[key]))
        axes[0].plot(x, prediction, color=COLORS[strain], label=strain)
        data = stats[(stats.experiment == "E06") & (stats.strain == strain)]
        axes[0].errorbar(data.added_gln_mM, data["mean"], yerr=data.sd,
                         fmt="o", ms=4, capsize=3, color=COLORS[strain])
    axes[0].set(xlabel="Added Gln (mM)", ylabel="Gln degraded (mM)",
                title="Conditional endpoint fit")
    axes[0].legend()
    a_wt = group_mean(obs, "E08", "WT")
    b_wt = group_mean(obs, "E07", "WT")
    for strain in ("WT", "Lp-NH3"):
        da = stats[(stats.experiment == "E08") & (stats.strain == strain)].iloc[0]
        db = stats[(stats.experiment == "E07") & (stats.strain == strain)].iloc[0]
        x0, y0 = da["mean"]/a_wt, db["mean"]/b_wt
        axes[1].errorbar(x0, y0, xerr=da.sd/a_wt, yerr=db.sd/b_wt, fmt="o",
                        capsize=3, color=COLORS[strain], label=strain)
    axes[1].axhline(1, color="#DDDDDD", lw=0.8)
    axes[1].axvline(1, color="#DDDDDD", lw=0.8)
    axes[1].set(xlabel="Residual ammonium / WT mean", ylabel="BCAA / WT mean",
                title="Joint endpoint phenotype", xlim=(0.55, 1.15), ylim=(0.7, 1.8))
    axes[1].legend(loc="upper right")
    for label, ax in zip("ab", axes):
        panel(ax, label)
    save_figure(fig, "02_endpoint_model_and_joint_phenotype", figures)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), layout="constrained")
    for label, ax, added in zip("ab", axes, (0, 5)):
        for x0, strain in enumerate(("WT", "Lp-Q")):
            values = obs[(obs.experiment == "E05") & (obs.added_gln_mM == added)
                         & (obs.strain == strain)].value.to_numpy()
            ax.scatter(x0+np.linspace(-0.07, 0.07, len(values)), values,
                       color=COLORS[strain], s=23)
            ax.errorbar(x0, values.mean(), yerr=values.std(ddof=1), fmt="_",
                        color="#222222", capsize=4, markersize=14)
        ax.set(xticks=[0, 1], xticklabels=["WT", "Lp-Q"], xlim=(-0.4, 1.4),
               title=f"E05: +{added} mM Gln", ylabel="Reported Gln readout\n(unit unresolved)")
        ax.set_ylim(bottom=0)
        panel(ax, label)
    fig.suptitle("Supporting assay; excluded from concentration calibration", fontsize=10)
    save_figure(fig, "06_supporting_gln_assay", figures)


def scenario_figures(curves, scan, candidate, sensitivity, config, figures):
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.7), layout="constrained")
    ref = config["reference"]
    for label, data in curves.groupby("scenario", sort=False):
        for ax, key in zip(axes, ("Gln_mM", "TAN_mM", "BCAA_mM")):
            ax.plot(data.tau, data[key], label=label, color=COLORS[label], lw=1.5)
    for letter, ax, title in zip("abc", axes, ("Gln", "Total ammonia", "BCAA")):
        ax.set(xlabel="Normalized interval, $\\tau$", ylabel="Concentration (mM)", title=title)
        panel(ax, letter)
    axes[1].axhline(ref["initial_total_ammonia_mM"], color="#AAAAAA", ls=":", lw=0.8)
    axes[0].legend(loc="lower left")
    fig.suptitle("Metabolic dynamics under the reference configuration", fontsize=9)
    save_figure(fig, "03_conditional_dynamics", figures)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.3), layout="constrained")
    fractions = np.sort(scan.q_fraction.unique())
    exposures = np.sort(scan.relative_exposure.unique())
    feasibility = scan.pivot(index="relative_exposure", columns="q_fraction",
                             values="feasible_assumed_targets").to_numpy(dtype=float)
    for ax, key, title, cmap in [
        (axes[0], "gln_removal_fraction", "Gln removal (%)", "YlGnBu"),
        (axes[1], "peak_TAN_to_initial", "Peak TAN / initial TAN", "YlOrBr"),
    ]:
        values = scan.pivot(index="relative_exposure", columns="q_fraction", values=key).to_numpy()
        if key == "gln_removal_fraction":
            values = values*100
        im = ax.pcolormesh(fractions, exposures, values, shading="nearest", cmap=cmap)
        if feasibility.min() != feasibility.max():
            ax.contour(fractions, exposures, feasibility, levels=[0.5],
                       colors=["#333333"], linewidths=0.9)
        if candidate is not None:
            ax.scatter(candidate["q_fraction"], candidate["relative_exposure"],
                       marker="*", s=95, color="#111111", edgecolor="white", lw=0.6)
        ax.set(xlabel="Lp-Q fraction in engineered mixture",
               ylabel="Relative effective exposure", title=title)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    for letter, ax in zip("ab", axes):
        panel(ax, letter)
    fig.suptitle("Mixture–exposure design space", fontsize=9)
    save_figure(fig, "04_conditional_design_space", figures)

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.5), layout="constrained", sharey=True)
    indices = np.arange(len(sensitivity))
    for ax, key, title in [
        (axes[0], "TAN_mM_at_1to1", "Endpoint TAN (mM)"),
        (axes[1], "BCAA_mM_at_1to1", "Endpoint BCAA (mM)"),
    ]:
        reference = sensitivity.iloc[0][key]
        ax.axvline(reference, ls=":", lw=1, color="#999999")
        ax.hlines(indices, reference, sensitivity[key], color="#C9CDD1", lw=1)
        ax.scatter(sensitivity[key], indices, s=26,
                   color=[COLORS["1:1 mixture"]]+["#547A8B"]*(len(indices)-1), zorder=3)
        ax.set(xlabel=title, title=title)
        ax.grid(axis="y", color="#EFEFEF", linewidth=0.5)
    axes[0].set_yticks(indices, sensitivity.scenario)
    axes[0].invert_yaxis()
    for letter, ax in zip("ab", axes):
        panel(ax, letter)
    fig.suptitle("Parameter sensitivity at 1:1 mixture, exposure = 1\n"
                 "One-parameter scenario changes", fontsize=9)
    save_figure(fig, "05_assumption_sensitivity", figures)


def parameter_table(gln, nitrogen, config):
    rows = []
    for key, value in gln.items():
        rows.append({"parameter": key, "value": value,
                     "status": "conditional_fit_E06", "basis": "shared basal Gln and first-order endpoint loss"})
    for key, value in nitrogen.items():
        rows.append({"parameter": key, "value": value,
                     "status": "conditional_calibration_E07_E08",
                     "basis": "assumed initial pools and BCAA nitrogen-source fraction"})
    for key, value in config["reference"].items():
        rows.append({"parameter": key, "value": value, "status": "assumption_or_assay_mapping",
                     "basis": config["parameter_notes"].get(key, "")})
    pka, f_nh3, fraction = speciation(config["reference"])
    for key, value in [("aqueous_pKa", pka), ("NH3_fraction", f_nh3),
                       ("ammonia_observation_fraction", fraction)]:
        rows.append({"parameter": key, "value": value, "status": "literature_equilibrium_at_assumed_conditions",
                     "basis": "R3; dilute zero-salinity water approximation"})
    return pd.DataFrame(rows)


def functional_check(obs, gln, nitrogen, reference, curves):
    """One in-run smoke check: source coverage, endpoint recovery and N balance."""
    assert obs.groupby("experiment").size().to_dict() == {"E05": 12, "E06": 10, "E07": 6, "E08": 6}
    pools = ["Gln_mM", "TAN_mM", "BCAA_mM", "other_organic_N_mM_N", "retained_N_mM_N"]
    assert np.isfinite(curves[pools].to_numpy()).all()
    assert curves[pools].min().min() >= -1e-10
    expected_n = (2*(gln["gln_basal_mM"]+reference["gln_supplement_mM"])
                  +reference["initial_total_ammonia_mM"]+reference["initial_bcaa_mM"]
                  +reference["initial_other_organic_n_mM_N"])
    np.testing.assert_allclose(curves.total_tracked_N_mM_N, expected_n, atol=1e-10, rtol=0)
    # Calibrate the E07/E08 isolated assay with a zero Gln source.
    isolated = dict(reference, gln_supplement_mM=0.0, q_activity_transfer=1.0,
                    n_activity_transfer=1.0)
    no_gln = dict(gln, gln_basal_mM=0.0)
    for theta in (0.0, 0.5, 1.0):
        alternative = dict(isolated, bcaa_new_n_fraction_from_ammonia=theta)
        npar = calibrate_ammonia_bcaa(obs, alternative)
        for strain, wt in [("WT", True), ("Lp-NH3", False)]:
            endpoint = trajectory([1], no_gln, npar, alternative, 0.0, 1.0, wt).iloc[0]
            np.testing.assert_allclose(endpoint.ammonia_assay_equivalent_mM,
                                       group_mean(obs, "E08", strain), atol=1e-10)
            np.testing.assert_allclose(endpoint.BCAA_mM, group_mean(obs, "E07", strain), atol=1e-10)
    zero = trajectory([0, 1], gln, nitrogen, reference, exposure=0)
    np.testing.assert_allclose(zero[pools].iloc[0], zero[pools].iloc[1], atol=1e-10)
    print("PASS: 34 measured values; conditional endpoint recovery; nonnegative pools; nitrogen balance; zero exposure.")


def write_results_report(gln, nitrogen, effects, curves, candidate, sensitivity, config, output):
    end = curves[curves.tau == curves.tau.max()]
    ref = config["reference"]
    design = config["design"]
    lines = [
        "# Metabolic Exchange：计算结果与 Wiki 文案",
        "",
        "本报告由 `run_model.py` 自动生成。实测值、条件拟合和假设情景分别标注。",
        "实验数据为处理后的测定值；重复类型未知。端点摘要报告观测值、均值和样本 SD。",
        "",
        "## 1. 真实数据的结论",
        "",
        "| 实验 | Gln补加/mM | 工程菌 | 相对WT变化 | 含义 |",
        "|---|---:|---|---:|---|",
    ]
    for row in effects.itertuples():
        lines.append(f"| {row.experiment} | {row.added_gln_mM:g} | {row.engineered_strain} "
                     f"| {row.change_percent:+.2f}% | 组间终点差异 |")
    lines.extend([
        "", "E05单位未确认，未纳入浓度动力学拟合。E07/E08同批同条件由用户确认，"
        "R列按非配对观测汇总。氨指标定义为相对WT的残余氨变化，初始氨去除率使用其自身的基准定义。",
        "", "![实测终点](figures/01_measured_endpoints.png)", "",
        "图中点为全部观测值，黑线为均值±样本SD；n表示所列观测值数量，重复类型见实验记录。",
        "",
        "## 2. Gln终点模型",
        "",
        r"$$D_{s,c}=(G_{\mathrm{basal}}+C_c)(1-e^{-\lambda_s})$$",
        "",
        f"- 拟合估计的基础Gln：**{gln['gln_basal_mM']:.4f} mM**，采用共同基础池与一阶消耗模型。",
        f"- WT有效作用系数：{gln['lambda_g_wt']:.5f}；Lp-Q：{gln['lambda_g_q']:.5f}。",
        f"- 对应每个实验终点的消耗比例：WT {100*(-np.expm1(-gln['lambda_g_wt'])):.2f}%，"
        f"Lp-Q {100*(-np.expm1(-gln['lambda_g_q'])):.2f}%。",
        f"- 对10个E06原始数值的残差RMSE：{gln['rmse_mM']:.4f} mM。",
        "",
        "这些条件估计采用共同基础Gln、相同有效作用时长和一阶消耗假设。"
        "模型使用两个补加水平校准有效作用系数，不估计Monod半饱和常数。",
        "",
        "![终点拟合和联合表型](figures/02_endpoint_model_and_joint_phenotype.png)",
        "",
        "左图误差线为SD；曲线是条件模型。右图各轴均除以对应WT均值，"
        "误差线为以WT均值归一化的组内SD，分母按固定参考值处理。",
        "",
        "## 3. 参考情景下的组合动态",
        "",
        f"参考情景：初始总氨{ref['initial_total_ammonia_mM']:g} mM，"
        f"初始BCAA {ref['initial_bcaa_mM']:g} mM，"
        f"额外可用有机氮{ref['initial_other_organic_n_mM_N']:g} mM-N；"
        f"Gln补加{ref['gln_supplement_mM']:g} mM；"
        f"氨释放系数ν={ref['ammonia_release_per_gln']:g}；"
        f"新积累BCAA的氨来源比例θ={ref['bcaa_new_n_fraction_from_ammonia']:g}。"
        "这些起始池、ν、θ和跨实验活性迁移关系均为情景假设。",
        "",
        "τ=1表示一个归一化作用区间，其物理时长待实验记录确定。"
        "参考暴露为1，参考情景按有效菌量设置1:1分配；与喷出CFU的换算需要活性映射。",
        "",
        "| 条件（模拟） | 终点Gln/mM | 终点总氨/mM | 终点BCAA/mM |",
        "|---|---:|---:|---:|",
    ])
    for row in end.itertuples():
        lines.append(f"| {row.scenario} | {row.Gln_mM:.3f} | {row.TAN_mM:.3f} | {row.BCAA_mM:.3f} |")
    lines.extend([
        "", "![条件动态](figures/03_conditional_dynamics.png)",
        "",
        "Gln实验和氨/BCAA实验分别提供参数约束。组合仿真采用共同培养环境、"
        "线性活性混合和WT样背景活性假设；上述数值为这一参考模型的输出。",
        "",
        "## 4. 双菌配比与有效暴露的约束优化",
        "",
        f"示例目标：Gln消耗≥{100*design['target_gln_removal_fraction']:g}%，"
        f"总氨峰值≤初始值的{100*design['max_peak_ammonia_to_initial']:g}%，"
        f"终点总氨≤初始值的{100*design['max_endpoint_ammonia_to_initial']:g}%。"
        f"这些是本次仿真的工程约束。扫描相对暴露"
        f"{design['relative_exposure_min']:g}–{design['relative_exposure_max']:g}及Lp-Q占比0–1。"
        f"共计算{design['q_fraction_points']*design['relative_exposure_points']}个网格方案。"
        "最优规则分为两级：先最小化满足全部约束的有效暴露，"
        "再在相同最小暴露下最大化Gln消耗。最优性限定于该目标、参考条件和搜索网格。",
    ])
    if candidate is None:
        lines.append("\n当前扫描范围内没有同时满足上述假设目标的组合。")
    else:
        lines.extend([
            "",
            f"- 参考情景网格最优方案：Lp-Q占比 **{100*candidate['q_fraction']:.1f}%**，"
            f"Lp-NH3占比 **{100*(1-candidate['q_fraction']):.1f}%**；"
            f"相对有效暴露 **{candidate['relative_exposure']:.2f}**。",
            f"- Gln消耗 **{100*candidate['gln_removal_fraction']:.2f}%**；"
            f"总氨峰值 **{candidate['peak_TAN_mM']:.3f} mM**；"
            f"终点总氨 **{candidate['endpoint_TAN_mM']:.3f} mM**。",
            f"- 同一方案的终点BCAA为 **{candidate['endpoint_BCAA_mM']:.3f} mM**；"
            "BCAA为附加输出，不参与当前优化排序。",
        ])
    lines.extend([
        "", "![设计空间](figures/04_conditional_design_space.png)",
        "",
        "黑色边界标示满足工程约束的区域，星号为上述两级目标下的网格最优方案。",
        "",
        "## 5. 假设敏感性与可辨识性",
        "",
        "![假设敏感性](figures/05_assumption_sensitivity.png)",
        "",
        "单参数敏感性分析在1:1有效菌量和参考暴露下逐项改变参数。",
        "θ=0、0.5、1均能重现同一组E07/E08均值，因此情景分析比较这些氮来源分配设置。"
        "当Gln释放额外氨时，不同分配设置可产生不同的BCAA预测；氮来源测量可进一步约束组合预测。",
        "",
        "| 情景 | 网格最优Lp-Q占比 | 网格最优相对暴露 |",
        "|---|---:|---:|",
    ])
    for row in sensitivity.itertuples():
        fraction = "无可行解" if pd.isna(row.candidate_q_fraction) else f"{100*row.candidate_q_fraction:.1f}%"
        dose = "—" if pd.isna(row.candidate_exposure) else f"{row.candidate_exposure:.2f}"
        lines.append(f"| {row.scenario} | {fraction} | {dose} |")
    lines.extend([
        "",
        "敏感性结果给出最优方案的条件依赖性。参考情景中的最优性以所列模型条件为边界，"
        "不同情景重新校准相应参数并采用相同的两级优化目标。",
        "",
        "## 6. 最优仿真结果与 Wiki 结论",
        "",
        "> 我们利用单菌湿实验数据校准代谢作用，建立双菌耦合模型，"
        "仿真不同有效配比与总有效暴露下的Gln消耗、氨变化和BCAA积累。"
        "通过在Gln消耗与控氨约束下最小化有效暴露，并在该暴露下最大化Gln消耗，"
        "确定模型设定和搜索网格内的最优仿真方案。",
    ])
    if candidate is not None:
        lines.extend([
            "",
            "| 最优仿真方案及输出 | 数值 |",
            "|---|---:|",
            f"| Lp-Q : Lp-NH3 有效比例 | {100*candidate['q_fraction']:.0f} : "
            f"{100*(1-candidate['q_fraction']):.0f} |",
            f"| 相对有效暴露 | {candidate['relative_exposure']:.2f} |",
            f"| Gln消耗比例 | {100*candidate['gln_removal_fraction']:.2f}% |",
            f"| 总氨峰值 | {candidate['peak_TAN_mM']:.3f} mM |",
            f"| 终点总氨 | {candidate['endpoint_TAN_mM']:.3f} mM |",
            f"| 终点BCAA | {candidate['endpoint_BCAA_mM']:.3f} mM |",
            "",
            f"> 在参考情景的{design['q_fraction_points']*design['relative_exposure_points']}个"
            f"网格方案中，最优有效配比为"
            f"Lp-Q:Lp-NH3 = {100*candidate['q_fraction']:.0f}:"
            f"{100*(1-candidate['q_fraction']):.0f}，"
            f"相对有效暴露为{candidate['relative_exposure']:.2f}。"
            f"该方案模拟实现Gln消耗{100*candidate['gln_removal_fraction']:.2f}%，"
            f"总氨峰值{candidate['peak_TAN_mM']:.3f} mM，"
            f"终点总氨{candidate['endpoint_TAN_mM']:.3f} mM，满足本次仿真的代谢约束。",
        ])
    else:
        lines.append("\n> 当前搜索网格中没有满足全部约束的方案，因此本次仿真不报告最优可行解。")
    lines.extend([
        "",
        "机制图建议：保留两株菌、Gln/氨/BCAA三个可测节点；用胞内有机氮池与碳骨架"
        "表示BCAA合成过程。菌株间氨转移和具体酶途径标注为假设连接，简化收支池标注为模型变量。",
        "",
        "模型使用说明与参数定义见上级目录的 `README.md` 和 `config.json`。",
    ])
    (output / "结果解读与Wiki文案_CN.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--check", action="store_true", help="Run one built-in functional smoke check")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output, figures, data = ROOT / "results", ROOT / "results" / "figures", ROOT / "data"
    for directory in (output, figures, data):
        directory.mkdir(parents=True, exist_ok=True)
    obs = read_observations((config_path.parent / config["source_table"]).resolve())
    stats, effects = endpoint_statistics(obs)
    gln, fit_values = fit_gln(obs)
    nitrogen = calibrate_ammonia_bcaa(obs, config["reference"])
    (output / "integrated_parameters.json").write_text(json.dumps({
        "gln": gln,
        "nitrogen": nitrogen,
        "reference": config["reference"],
        "design": config["design"],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    curves = []
    for label, fraction, wt in [
        ("WT", 0.5, True), ("Lp-Q", 1.0, False),
        ("Lp-NH3", 0.0, False), ("1:1 mixture", 0.5, False),
    ]:
        curve = trajectory(np.linspace(0, 1, 201), gln, nitrogen, config["reference"],
                           fraction, 1.0, wt)
        curve["scenario"] = label
        curve["evidence"] = "conditional_simulation"
        curves.append(curve)
    curves = pd.concat(curves, ignore_index=True)
    scan = design_scan(gln, nitrogen, config["reference"], config["design"])
    candidate = choose_candidate(scan)
    sensitivity = scenario_sensitivity(obs, gln, config["reference"], config["design"])
    obs.to_csv(data / "measured_observations.csv", index=False)
    for frame, name in [
        (stats, "endpoint_summary"), (effects, "endpoint_effects"),
        (fit_values, "gln_calibration"), (parameter_table(gln, nitrogen, config), "parameters"),
        (curves, "conditional_timecourses"), (scan, "conditional_design_scan"),
        (sensitivity, "assumption_sensitivity"),
    ]:
        frame.to_csv(output / f"{name}.csv", index=False)
    observed_figures(obs, stats, gln, figures)
    scenario_figures(curves, scan, candidate, sensitivity, config, figures)
    write_results_report(gln, nitrogen, effects, curves, candidate, sensitivity, config, output)
    if args.check:
        functional_check(obs, gln, nitrogen, config["reference"], curves)
    print(json.dumps({
        "measured_values": len(obs), "gln_conditional_fit": gln,
        "reference_candidate": candidate,
        "output_directory": str(output),
        "figure_sets": 6,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

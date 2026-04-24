#!/usr/bin/env python
"""
读取敲除分析结果，提取指定基因敲除后指定反应的流量值

功能：
1. 读取 detailed.csv 文件
2. 根据敲除基因名称筛选数据
3. 输出感兴趣反应的真实流量和预测流量
4. 同时输出三种方法（Eki2vivo, EkiLLm, fba）的结果
"""

import math
import os
from typing import Sequence

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex

# =============================================================================
# 配置参数
# =============================================================================

# 输入文件
INPUT_FILE = "./detailed.csv"

# 感兴趣的敲除基因（注意大小写要与数据中一致）
KNOCKOUT_GENE = "Pgi"

# 感兴趣的反应列表
REACTIONS_OF_INTEREST = [
    "PGI",
    "PFK",
    "FBP",
    "FBA",
    "TPI",
    "G6PDH2r",
    "GND",
    "RPE",
    "RPE_reverse",
    "RPI",
    "RPI_reverse",
    "TKT1",
    "TKT1_reverse",
    "TKT2",
    "TKT2_reverse",
    "TALA",
    "TALA_reverse",
    "EDD",
    "EDA",
    "GAPD",
    "PGK_reverse",
    "PGM_reverse",
    "ENO",
    "PYK",
    "PDH",
    "CS",
    "ACONTa",
    "ACONTb",
    "ICDHyr",
    "AKGDH",
    "SUCOAS",
    "SUCDi",
    "FUM",
    "MDH",
    "PPCK",
    "PPC",
    "ME1",
    "ME2",
    "ICL",
    "MALS"
]

# 要比较的方法
METHODS = ["FluxAnchor", "KinLLM", "fba"]

# 误差计算稳定常数，避免除零
EPSILON = 0.001
COLOR_LOW = "#C1C64E"
COLOR_HIGH = "#9D7EAC"
COLOR_STOPS = [
    "#9D7EAC",
    "#8599B5",
    "#76B8B5",
    "#B0E49D",
    "#C1C64E",
]


def row_normalize(values: Sequence[float]) -> list[float]:
    """对单行数据做L2归一化，强调流量之间的相对比例。"""
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= EPSILON:
        return [0.0 for _ in values]
    return [v / norm for v in values]


def calculate_norm_errors(true_values: Sequence[float], pred_values: Sequence[float]) -> list[float]:
    """基于行归一化结果计算逐反应的标准化误差。"""
    true_normed = row_normalize(true_values)
    pred_normed = row_normalize(pred_values)
    return [abs(t - p) for t, p in zip(true_normed, pred_normed)]


def summarize_norm_error(norm_errors: Sequence[float]) -> float:
    """对一组标准化误差求平均，越小表示整体拟合越好。"""
    if not norm_errors:
        return float("nan")
    return sum(norm_errors) / len(norm_errors)


def compute_bias_ratio(true_value: float, pred_value: float) -> float:
    """abs(预测-真实)/真实，若真实值接近0则使用(|真实值|+1)稳定分母。"""
    # denom = abs(true_value)
    # if denom <= EPSILON:
    #     # 避免除0，保证对真实流量极小的反应仍能展示误差比例
    #     denom = abs(true_value) + 1.0
    # return abs(pred_value - true_value) / denom
    return (pred_value + 1) / (true_value + 1) 


def format_percent(value: float, width: int = 12) -> str:
    """对百分比格式化，若缺失则返回 N/A。"""
    if value is None or not math.isfinite(value):
        return f"{'N/A':>{width}}"
    return f"{value:>{width}.2%}"


def build_prism_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("PrismGradient", COLOR_STOPS)


def color_for_error(error: float, min_error: float, max_error: float, cmap: LinearSegmentedColormap) -> str:
    if not math.isfinite(error):
        return COLOR_HIGH
    if max_error <= min_error:
        return COLOR_LOW
    # Map high error -> purple (0), low error -> yellow (1)
    norm = (max_error - error) / (max_error - min_error)
    norm = min(1.0, max(0.0, norm))
    return to_hex(cmap(norm))

# =============================================================================
# 主程序
# =============================================================================

def main():
    # 读取数据
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, INPUT_FILE)

    print(f"读取文件: {input_path}")
    df = pd.read_csv(input_path)

    print(f"总数据量: {len(df)} 行")
    print(f"可用方法: {df['Method'].unique().tolist()}")
    print(f"可用基因: {sorted(df['Gene'].unique().tolist())[:10]}... (共{df['Gene'].nunique()}个)")

    # 筛选指定基因的数据
    gene_data = df[df['Gene'].str.lower() == KNOCKOUT_GENE.lower()]

    if gene_data.empty:
        print(f"\n错误: 未找到基因 '{KNOCKOUT_GENE}' 的数据")
        print(f"可用基因列表: {sorted(df['Gene'].unique().tolist())}")
        return

    print(f"\n{'='*80}")
    print(f"敲除基因: {KNOCKOUT_GENE}")
    print(f"找到 {len(gene_data)} 条记录（对应不同方法）")
    print(f"{'='*80}")

    # 检查感兴趣的反应是否存在
    available_reactions = []
    missing_reactions = []

    for rxn in REACTIONS_OF_INTEREST:
        if f"{rxn}_true" in df.columns:
            available_reactions.append(rxn)
        else:
            missing_reactions.append(rxn)

    if missing_reactions:
        print(f"\n警告: 以下反应在数据中不存在: {missing_reactions}")

    if not available_reactions:
        print("\n错误: 所有指定的反应都不存在于数据中")
        return

    print(f"\n分析的反应: {available_reactions}")

    # 输出每个方法的结果
    print(f"\n{'='*80}")
    print(f"各方法预测结果对比")
    print(f"{'='*80}")

    # 构建结果表格
    results = []
    method_entries: dict[str, list[dict]] = {}
    entry_lookup: dict[tuple[str, str], dict] = {}
    cmap = build_prism_cmap()

    error_values = []
    for method in METHODS:
        method_data = gene_data[gene_data['Method'] == method]
        if method_data.empty:
            continue
        row = method_data.iloc[0]
        for rxn in available_reactions:
            abs_err = row.get(f"{rxn}_abs_error", 0)
            if pd.notna(abs_err):
                error_values.append(abs(abs_err))
    min_error = min(error_values) if error_values else 0.0
    max_error = max(error_values) if error_values else 1.0

    for method in METHODS:
        method_data = gene_data[gene_data['Method'] == method]
        method_entries[method] = []

        if method_data.empty:
            print(f"\n警告: 方法 '{method}' 没有 {KNOCKOUT_GENE} 的数据")
            continue

        row = method_data.iloc[0]

        print(f"\n【{method}】")
        print(f"  敲除类型: {row.get('Knockout_Type', 'N/A')}")
        print(f"  影响反应数: {row.get('Affected_Reactions', 'N/A')}")
        print(f"  RMSE: {row.get('RMSE', 'N/A'):.4f}" if pd.notna(row.get('RMSE')) else f"  RMSE: N/A")
        print(f"  R²: {row.get('R2', 'N/A'):.4f}" if pd.notna(row.get('R2')) else f"  R²: N/A")
        print()

        print(f"  {'反应':<15} {'真实流量':>12} {'预测流量':>12} {'绝对误差':>12} {'相对误差':>12} {'Norm误差':>12} {'偏差比':>12} {'颜色':>10}")
        print(f"  {'-'*98}")

        true_values = [row.get(f"{rxn}_true", 0) for rxn in available_reactions]
        pred_values = [row.get(f"{rxn}_pred", 0) for rxn in available_reactions]
        norm_errors = calculate_norm_errors(true_values, pred_values)

        for rxn, true_val, pred_val, norm_err in zip(available_reactions, true_values, pred_values, norm_errors):
            abs_err = row.get(f"{rxn}_abs_error", 0)
            rel_err = row.get(f"{rxn}_rel_error", 0)
            bias_ratio = compute_bias_ratio(true_val, pred_val)
            color = color_for_error(abs(abs_err), min_error, max_error, cmap)

            print(
                f"  {rxn:<15} {true_val:>12.4f} {pred_val:>12.4f} "
                f"{abs_err:>12.4f} {rel_err:>12.2%} {norm_err:>12.4f} {format_percent(bias_ratio)} {color:>10}"
            )

            entry = {
                'Method': method,
                'Reaction': rxn,
                'True_Flux': true_val,
                'Pred_Flux': pred_val,
                'Abs_Error': abs_err,
                'Rel_Error': rel_err,
                'Norm_Error': norm_err,
                'Bias_Ratio': bias_ratio,
                'Color': color
            }
            results.append(entry)
            method_entries[method].append(entry)
            entry_lookup[(method, rxn)] = entry

        avg_norm_error = summarize_norm_error(norm_errors)
        if pd.notna(avg_norm_error):
            print(f"\n  平均标准化误差 (行归一化, 越小越好): {avg_norm_error:.4f}")

    # 方法整体误差范围概览
    if any(method_entries.values()):
        print(f"\n{'='*80}")
        print("误差范围概览（帮助快速感知预测与真实流量的偏差区间）")
        print(f"{'='*80}")

        for method in METHODS:
            entries = method_entries.get(method, [])
            if not entries:
                continue

            abs_errors = [abs(item['Abs_Error']) for item in entries]
            rel_errors = [abs(item['Rel_Error']) for item in entries]
            bias_ratios = [item['Bias_Ratio'] for item in entries if math.isfinite(item['Bias_Ratio'])]

            avg_abs_error = sum(abs_errors) / len(abs_errors)
            avg_rel_error = sum(rel_errors) / len(rel_errors)

            min_abs_item = min(entries, key=lambda x: abs(x['Abs_Error']))
            max_abs_item = max(entries, key=lambda x: abs(x['Abs_Error']))
            min_rel_item = min(entries, key=lambda x: abs(x['Rel_Error']))
            max_rel_item = max(entries, key=lambda x: abs(x['Rel_Error']))

            print(f"\n【{method}】")
            print(f"  绝对误差范围: {abs(min_abs_item['Abs_Error']):.4f} ~ {abs(max_abs_item['Abs_Error']):.4f}")
            print(f"  平均绝对误差: {avg_abs_error:.4f}")
            print(f"  相对误差范围: {abs(min_rel_item['Rel_Error']):.2%} ~ {abs(max_rel_item['Rel_Error']):.2%}")
            print(f"  平均相对误差: {avg_rel_error:.2%}")
            if bias_ratios:
                min_bias = min(bias_ratios)
                max_bias = max(bias_ratios)
                avg_bias = sum(bias_ratios) / len(bias_ratios)
                print(f"  偏差比范围: {min_bias:.2%} ~ {max_bias:.2%}")
                print(f"  平均偏差比: {avg_bias:.2%}")
            print(
                f"  最小误差示例: {min_abs_item['Reaction']} "
                f"(真实 {min_abs_item['True_Flux']:.4f}, 预测 {min_abs_item['Pred_Flux']:.4f}, "
                f"差值 {abs(min_abs_item['Abs_Error']):.4f})"
            )
            print(
                f"  最大误差示例: {max_abs_item['Reaction']} "
                f"(真实 {max_abs_item['True_Flux']:.4f}, 预测 {max_abs_item['Pred_Flux']:.4f}, "
                f"差值 {abs(max_abs_item['Abs_Error']):.4f})"
            )

    # 方法间对比（横向对比同一反应）
    print(f"\n{'='*80}")
    print(f"同一反应不同方法对比")
    print(f"{'='*80}")

    for rxn in available_reactions:
        print(f"\n【{rxn}】")
        print(f"  {'方法':<12} {'真实流量':>12} {'预测流量':>12} {'绝对误差':>12} {'相对误差':>12} {'Norm误差':>12} {'偏差比':>12} {'颜色':>10}")
        print(f"  {'-'*98}")

        for method in METHODS:
            method_data = gene_data[gene_data['Method'] == method]
            if not method_data.empty:
                row = method_data.iloc[0]
                true_val = row.get(f"{rxn}_true", 0)
                pred_val = row.get(f"{rxn}_pred", 0)
                abs_err = row.get(f"{rxn}_abs_error", 0)
                rel_err = row.get(f"{rxn}_rel_error", 0)
                norm_err = entry_lookup.get((method, rxn), {}).get('Norm_Error', 0)
                bias_ratio = entry_lookup.get((method, rxn), {}).get('Bias_Ratio')
                color = entry_lookup.get((method, rxn), {}).get('Color')
                print(
                    f"  {method:<12} {true_val:>12.4f} {pred_val:>12.4f} "
                    f"{abs_err:>12.4f} {rel_err:>12.2%} {norm_err:>12.4f} {format_percent(bias_ratio)} {color:>10}"
                )

    # 按标准化误差排序展示差距
    if results:
        print(f"\n{'='*80}")
        print("按Norm标准化误差从小到大展示")
        print(f"{'='*80}")

        sorted_results = sorted(results, key=lambda x: x['Norm_Error'])
        print(f"  {'方法':<12} {'反应':<12} {'真实流量':>12} {'预测流量':>12} {'Norm误差':>12} {'颜色':>10}")
        print(f"  {'-'*76}")
        for item in sorted_results:
            print(
                f"  {item['Method']:<12} {item['Reaction']:<12} "
                f"{item['True_Flux']:>12.4f} {item['Pred_Flux']:>12.4f} "
                f"{item['Norm_Error']:>12.4f} {item['Color']:>10}"
            )

    # 保存结果到CSV
    if results:
        results_df = pd.DataFrame(results)
        output_file = os.path.join(script_dir, f"reaction_flux_{KNOCKOUT_GENE}.csv")
        results_df.to_csv(output_file, index=False)
        print(f"\n结果已保存到: {output_file}")


def plot_prism_colorbar():
    """Custom GraphPad Prism-style colorbar (high error -> low error)."""
    import matplotlib.pyplot as plt

    prism_cmap = build_prism_cmap()
    data = np.linspace(0, 1, 100).reshape(10, 10)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap=prism_cmap)
    cbar = fig.colorbar(im)
    cbar.set_label('Flux error (high → low)', rotation=270, labelpad=15)
    ax.set_title("Flux error colormap: #9D7EAC → #C1C64E")
    plt.show()


if __name__ == "__main__":
    main()

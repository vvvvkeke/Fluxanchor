#!/usr/bin/env python
"""
可视化Flux Ratio误差对比
参考 biological_analysis_supplement.py 的 2_pathway_error_comparison.pdf 风格
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================================
# 配置参数
# ============================================================================

# 设置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'

# 方法和配色（与biological_analysis_supplement.py保持一致）
methods = ['Eki2vivo', 'EkiLLm', 'fba']
methods_color = ['#96CCEA', '#B2A3DD', '#ED949A']

# Pathway配色（根据flux ratio数据调整）
# pathway_colors_full = {
#     'Glycolysis': '#e74c3c',
#     'ED pathway': '#3498db',
#     'PP pathway': '#2ecc71',
#     'Gluconeogenesis': '#feca57',
#     'TCA': '#f39c12',
#     'Other': '#95a5a6'
# }
pathway_colors_full = {
    'Glycolysis': "#d3eac5", 'ED': "#CD61D7", 'PP': "#db349b",  'TCA': '#2ecc71',
    'Gluconeogenesis': '#feca57', 'Amino acid': '#f39c12', 
    'Fatty acid': '#9b59b6', 
    'Other': '#95a5a6',
    # 'Nucleotide': '#ee5a6f', 'Fermentation': '#48dbfb', 'Respiration': "#7D1B88",
    # 'Exchange': '#1abc9c', 'Energy': '#e67e22'
}


# 输入输出路径
results_dir = './results/flux_ratio_analysis'
output_dir = './analysis/flux_ratio_visualizations'
os.makedirs(output_dir, exist_ok=True)

print("="*100)
print("Flux Ratio误差可视化")
print("="*100)

# ============================================================================
# 读取数据
# ============================================================================
print("\n正在读取pathway汇总数据...")
pathway_summary = pd.read_csv(f'{results_dir}/pathway_summary.csv')

# 简化pathway名称
pathway_name_map = {
    'Glycolysis(Serine through glycolysis)': 'Glycolysis',
    'PP pathway(PEP through PP pathway, upper bound)  ': 'PP',
    'ED pathway(PYR through ED) ': 'ED',
    'Gluco-neogenesis(PEP from oxaloacetate)': 'Gluconeo-PEP',
    'Gluco-neogenesis(PYR from MAL, upper bound)': 'Gluconeo-PYR(U)',
    'Gluco-neogenesis(PYR from MAL, lower bound)': 'Gluconeo-PYR(L)',
    'TCA cycle(Oxaloacetate through TCA cycle)': 'TCA'
}

pathway_summary['Pathway_Short'] = pathway_summary['Pathway'].map(pathway_name_map)

# 定义pathway顺序和颜色
pathway_order = ['Glycolysis', 'PP', 'ED', 
                'Gluconeo-PEP', 'Gluconeo-PYR(U)', 'Gluconeo-PYR(L)',
                'TCA']

pathway_colors = {
    'Glycolysis': "#d3eac5",
    'ED': "#CD61D7",
    'PP': "#db349b",
    'Gluconeo-PEP': '#feca57',
    'Gluconeo-PYR(U)': '#f39c12',
    'Gluconeo-PYR(L)': '#f39c12',
    'TCA': '#2ecc71',
}

print(f"加载数据: {len(pathway_summary)} 条记录")
print(f"Methods: {methods}")
print(f"Pathways: {pathway_order}")

# ============================================================================
# 图1: Pathway级别的平均绝对误差对比（主图）
# ============================================================================
print("\n生成图1: Pathway级别的平均绝对误差对比...")

fig, ax = plt.subplots(figsize=(16, 8))

# 准备绘图数据
x = np.arange(len(pathway_order))
width = 0.25

for i, method in enumerate(methods):
    method_data = pathway_summary[pathway_summary['Method'] == method]

    means = []
    stds = []
    for pathway in pathway_order:
        pathway_row = method_data[method_data['Pathway_Short'] == pathway]
        if len(pathway_row) > 0:
            means.append(pathway_row['Avg_Abs_Error'].values[0])
            stds.append(pathway_row['Std_Abs_Error'].values[0])
        else:
            means.append(0)
            stds.append(0)

    bars = ax.bar(x + i*width, means, width, label=method, alpha=0.8,
           yerr=stds, color=methods_color[i],
           error_kw={'ecolor': methods_color[i],
                     'capsize': 3,
                     'capthick': 1.2,
                     'elinewidth': 1.2})

    # 在每个柱子上方添加数值标签
    # for j, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
    #     height = bar.get_height()
    #     if height > 0:  # 只标注非零值
    #         # 标签位置：柱子高度 + 标准差 + 小偏移
    #         label_y = height + std_val + 0.01
    #         ax.text(bar.get_x() + bar.get_width()/2., label_y,
    #                f'{mean_val:.3f}',
    #                ha='center', va='bottom',
    #                fontsize=12, fontweight='bold',
    #                color=methods_color[i])

ax.tick_params(axis='both', direction='in', width=4)

# 图例
legend = ax.legend(fontsize=25)
for i, text in enumerate(legend.get_texts()):
    if i < len(methods_color):
        text.set_color(methods_color[i])

# 坐标轴设置
ax.tick_params(axis='y', labelsize=20)
ax.set_ylabel('MAE', fontsize=25, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(pathway_order, fontsize=20, rotation=45, ha='right')

# 为每个标签设置对应的pathway颜色
for tick_label, pathway in zip(ax.get_xticklabels(), pathway_order):
    tick_label.set_color(pathway_colors.get(pathway, '#000000'))

plt.tight_layout()
output_path = os.path.join(output_dir, 'pathway_error_comparison.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 已保存: {output_path}")

output_path_png = os.path.join(output_dir, 'pathway_error_comparison.png')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
print(f"✅ 已保存: {output_path_png}")
plt.close()
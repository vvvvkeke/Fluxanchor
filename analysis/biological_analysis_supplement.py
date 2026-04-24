"""
补充生物学分析图 - 基于13C_log2fc_all_reaction_cluster2.py的数据
提供更直观的定量分析，验证和补充Grok的解读
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'          # 全部文字
plt.rcParams['axes.labelweight'] = 'bold'     # 坐标轴标签
# ============================================================================
# 配置参数
# ============================================================================
ORGANISM = "E. coli"  # 可选: "E. coli" 或 "B. subtilis"
# ORGANISM = "B. subtilis"  # 可选: "E. coli" 或 "B. subtilis"
COLORMAP = 'RdBu_r'
FLUX_THRESHOLD = 0

# 根据物种配置文件路径
if ORGANISM == "E. coli":
    # wildtype_csv_path = '/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/13C_analysis_data_threshold_0.01.csv'
    # detailed_csv_path = '/home/zhangyangyu/kcat_km_predict/results_threshold_0.01/iECDH1ME8569_1439/detailed.csv'

    wildtype_csv_path = '/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/13C_analysis_data_extend_reactions.csv'
    detailed_csv_path = '/home/zhangyangyu/kcat_km_predict/results_extend_reactions/iECDH1ME8569_1439/detailed.csv'
    output_dir = '/home/zhangyangyu/kcat_km_predict/analysis/draw7_biological'
elif ORGANISM == "B. subtilis":
    wildtype_csv_path = '/home/zhangyangyu/kcat_km_predict/predict/Bacillus subtilis/analysis/13C_analysis_data_threshold_0.01.csv'
    detailed_csv_path = '/home/zhangyangyu/kcat_km_predict/results_threshold_0.01/Bacillus_subtilis/detailed.csv'
    output_dir = '/home/zhangyangyu/kcat_km_predict/analysis/draw_bacillus_biological'
else:
    raise ValueError(f"不支持的物种: {ORGANISM}")

os.makedirs(output_dir, exist_ok=True)

print("=" * 100)
print("补充生物学分析 - 定量验证和可视化")
print("=" * 100)

# ============================================================================
# 读取数据
# ============================================================================
print("\n正在读取数据...")
wildtype_df = pd.read_csv(wildtype_csv_path)
detailed_df = pd.read_csv(detailed_csv_path)
reaction_names = wildtype_df['Reaction'].tolist()

# 定义代谢路径注释
pathway_annotations = {
    'Glycolysis': [
        "ENO", 
        "FBA", 
        "GAPD", 
        "PFK", 
        "PGI", 
        "PGK", 
        "PYK", 
        "TPI",
        "PGM",
        # 关键的反向反应（代谢回补和 overflow 指标）
        "PGK_reverse",  # PGK 逆流：磷酸化底物合成
        "ENO_reverse",  # ENO 逆流
        "FBA_reverse",  # FBA 逆流
        "GAPD_reverse", # GAPD 逆流
        "TPI_reverse",  # TPI 逆流（磷酸二羟丙酮/甘油醛-3-磷酸互变）
        "PGI_reverse",  # PGI 逆流（葡萄糖-6-磷酸/果糖-6-磷酸互变）
        "PGM_reverse"  # PGM 逆流（磷酸甘油酸变位酶）
    ],
    'PPP': [
        "G6PDH2r", 
        "GND", 
        "TALA", 
        "TKT1", 
        "TKT2",
        "RPI",
        "RPE",
        # 戊糖磷酸途径的可逆反应
        "G6PDH2r_reverse",  # 葡萄糖-6-磷酸脱氢酶逆流
        "TALA_reverse",  # 可逆转醛醇酶反应
        "TKT1_reverse",  # 可逆转酮醇酶反应
        "TKT2_reverse",  # 可逆转酮醇酶反应
        # "RPE_reverse",   # 核酮糖-5-磷酸-3-差向异构酶
        # "RPI_reverse"   # 核糖-5-磷酸异构酶
    ],
    'TCA': [
            "CS", 
            "FUM", 
            "ICDHyr", 
            "MDH", 
            "PDH", 
            "SUCDi"
            "FUM_reverse",   # 延胡索酸酶逆流
            "MDH_reverse",   # 苹果酸脱氢酶逆流
            "ICDHyr_reverse", # 异柠檬酸脱氢酶逆流
            "SUCOAS_reverse", # 琥珀酰辅酶A合成酶逆流
            "ACONTa_reverse", # 顺乌头酸酶a逆流
            "ACONTb_reverse" # 顺乌头酸酶b逆流
            ],
    'Gluconeogenesis': ['FBP', 'PPCK', 'PPS', 'PEPC'],
    'Amino acid': [
        # 芳香族氨基酸途径
        "CHORS", "DHQS", "DHQTi",
        # 支链氨基酸途径
        "DHAD1", "IPMD", "IPPS",
        # 丝氨酸/甘氨酸途径
        "PSERT", "PSP_L", "SERAT",
        # 赖氨酸途径
        "DAPDC", "DAPE", "DHDPS", "DHDPRy",
    ],
    'Fatty acid': [
        'ACCOAC', 
    'ACACT1r', 'ACACT2r', 'ACACT3r', 'ACACT4r', 'ACACT5r', 'ACACT6r', 'ACACT7r', 'ACACT8r',
    'ACACT1r_reverse', 'ACACT2r_reverse', 'ACACT3r_reverse', 'ACACT4r_reverse', 'ACACT5r_reverse', 'ACACT6r_reverse', 'ACACT7r_reverse', 'ACACT8r_reverse',
    'ECOAH1', 'ECOAH2', 'ECOAH3', 'ECOAH4', 'ECOAH5', 'ECOAH6', 'ECOAH7', 'ECOAH8',
    'ECOAH1_reverse', 'ECOAH2_reverse', 'ECOAH3_reverse', 'ECOAH4_reverse', 'ECOAH5_reverse', 'ECOAH6_reverse', 'ECOAH7_reverse', 'ECOAH8_reverse',
    'HACD1', 'HACD2', 'HACD3', 'HACD4', 'HACD5', 'HACD6', 'HACD7', 'HACD8',
    'HACD1_reverse', 'HACD2_reverse', 'HACD3_reverse', 'HACD4_reverse', 'HACD5_reverse', 'HACD6_reverse', 'HACD7_reverse', 'HACD8_reverse',
    ],
    # 'Nucleotide': ['ADE', 'GUA', 'CYT', 'URA', 'THY', 'PRPP', 'PUR', 'PYR', 'RNDR', 'NTD', 'NTP', 'dNTP'],
    # 'Fermentation': ['LDH', 'ALCD', 'ACK', 'PTA', 'LACT', 'ETOH', 'FORM', 'FOR'],
    # 'Respiration': ['NADH', 'FADH', 'CYT', 'COX', 'UQO', 'NADH16pp', 'CYTBD', 'CYTBO'],
    # 'Exchange': ['EX_glc', 'EX_o2', 'EX_co2', 'EX_ac', 'EX_lac', 'EX_succ', 'EX_', 'IEX', 'tex'],
    # 'Energy': ['ATPS4rpp', 'ATPS', 'ATP', 'ADP', 'AMP'],
}

pathway_order = ['Glycolysis', 'PPP', 'TCA', 'Gluconeogenesis',
                 'Amino acid', 'Fatty acid', 'Other'
                #  'Fatty_acid', 'Nucleotide', 'Fermentation',
                #  'Respiration', 'Exchange', 'Energy'
                 ]
                 

pathway_colors = {
    'Glycolysis': '#e74c3c', 'PPP': '#3498db', 'TCA': '#2ecc71',
    'Gluconeogenesis': '#feca57', 'Amino acid': '#f39c12', 'Fatty acid': '#9b59b6', 'Other': '#95a5a6'
    # 'Fatty_acid': '#9b59b6',
    # 'Nucleotide': '#ee5a6f', 'Fermentation': '#48dbfb', 'Respiration': "#7D1B88",
    # 'Exchange': '#1abc9c', 'Energy': '#e67e22'
}

# 为每个反应分配路径标签
reaction_pathway_map = {}
for reaction in reaction_names:
    assigned = False
    for pathway, keywords in pathway_annotations.items():
        for keyword in keywords:
            if keyword in reaction:
                reaction_pathway_map[reaction] = pathway
                assigned = True
                break
        if assigned:
            break
    if not assigned:
        reaction_pathway_map[reaction] = 'Other'

# 构建log2fc数据
def build_log2fc_matrix(wildtype_df, detailed_df, reaction_names, method_name,
                        method_wildtype_col, flux_threshold=FLUX_THRESHOLD):
    """构建log2 fold change矩阵"""
    epsilon = 1e-10

    # 处理wildtype
    wildtype_log2fc = {}
    for idx, row in wildtype_df.iterrows():
        reaction = row['Reaction']
        true_val = row['Value']
        pred_val = row[method_wildtype_col]

        if pd.notna(true_val) and pd.notna(pred_val):
            if abs(true_val) < flux_threshold:
                wildtype_log2fc[reaction] = np.nan
            else:
                log2_fc = np.log2((pred_val + epsilon) / (true_val + epsilon))
                wildtype_log2fc[reaction] = log2_fc
        else:
            wildtype_log2fc[reaction] = np.nan

    # 处理突变型
    method_df = detailed_df[detailed_df['Method'] == method_name].copy()
    log2fc_data = {'wildtype': wildtype_log2fc}

    for idx, row in method_df.iterrows():
        gene = row['Gene']
        gene_data = {}

        for reaction in reaction_names:
            true_col = f"{reaction}_true"
            pred_col = f"{reaction}_pred"

            if true_col in method_df.columns and pred_col in method_df.columns:
                true_val = row[true_col]
                pred_val = row[pred_col]

                if pd.notna(true_val) and pd.notna(pred_val):
                    if abs(true_val) <= flux_threshold:
                        gene_data[reaction] = np.nan
                    else:
                        log2_fc = np.log2((pred_val + epsilon) / (true_val + epsilon))
                        gene_data[reaction] = log2_fc
                else:
                    gene_data[reaction] = np.nan

        log2fc_data[gene] = gene_data

    log2fc_df = pd.DataFrame(log2fc_data).T
    return log2fc_df

# 构建三种方法的数据
methods = ['Eki2vivo', 'EkiLLm', 'fba']
methods_wildtype = ['Eki2vivo', 'EkiLLM', 'FBA']
methods_color = ['#96CCEA', '#B2A3DD', '#ED949A']
method_log2fc_data = {}

print("\n正在构建log2 fold change数据...")
for method, method_wildtype in zip(methods, methods_wildtype):
    log2fc_df = build_log2fc_matrix(wildtype_df, detailed_df, reaction_names,
                                     method, method_wildtype)
    method_log2fc_data[method] = log2fc_df
    print(f"  {method}: {log2fc_df.shape}")

# 找到共同反应
common_reactions = set(reaction_names)
for method in methods:
    common_reactions = common_reactions.intersection(set(method_log2fc_data[method].columns))
common_reactions = sorted(list(common_reactions))

for method in methods:
    method_log2fc_data[method] = method_log2fc_data[method][common_reactions]

print(f"\n共同反应数: {len(common_reactions)}")

# ============================================================================
# 分析1: 整体误差分布比较 (Violin Plot)
# ============================================================================
print("\n" + "="*100)
print("分析1: 三种方法的误差分布比较 (验证Eki2vivo是否最准确)")
print("="*100)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 准备数据
all_data = []
for method in methods:
    df = method_log2fc_data[method]
    values = df.values.flatten()
    values = values[~np.isnan(values)]
    all_data.extend([{'Method': method, 'Log2FC': v} for v in values])

plot_df = pd.DataFrame(all_data)

# 子图1: Violin plot
ax = axes[0]
sns.violinplot(data=plot_df, x='Method', y='Log2FC', ax=ax, palette=methods_color, inner='box')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect prediction')
ax.axhline(y=1, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='2-fold error')
ax.axhline(y=-1, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)
ax.set_title('Prediction Error Distribution Across Methods', fontsize=14, fontweight='bold')
ax.set_ylabel('Log2(Predicted/Measured)', fontsize=12, fontweight='bold')
ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 子图2: 绝对误差的Box plot
plot_df['AbsLog2FC'] = plot_df['Log2FC'].abs()
ax = axes[1]
sns.boxplot(data=plot_df, x='Method', y='AbsLog2FC', ax=ax, palette=methods_color, showfliers=False)
ax.set_title('Absolute Prediction Error (|Log2FC|)', fontsize=14, fontweight='bold')
ax.set_ylabel('|Log2(Predicted/Measured)|', fontsize=12, fontweight='bold')
ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# 添加统计信息
for i, method in enumerate(methods):
    method_data = plot_df[plot_df['Method'] == method]['AbsLog2FC']
    median_val = method_data.median()
    ax.text(i, median_val, f'{median_val:.3f}', ha='center', va='bottom',
            fontweight='bold', fontsize=10)

plt.tight_layout()
output_path = os.path.join(output_dir, '1_error_distribution_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 已保存: {output_path}")
plt.close()

# 打印统计结果
print("\n统计摘要:")
for method in methods:
    method_data = plot_df[plot_df['Method'] == method]
    print(f"\n{method}:")
    print(f"  中位数 |Log2FC|: {method_data['AbsLog2FC'].median():.3f}")
    print(f"  平均值 |Log2FC|: {method_data['AbsLog2FC'].mean():.3f}")
    print(f"  在±0.3范围内的比例: {(method_data['AbsLog2FC'] < 0.3).sum() / len(method_data) * 100:.1f}%")
    print(f"  在±1范围内的比例: {(method_data['AbsLog2FC'] < 1).sum() / len(method_data) * 100:.1f}%")

# ============================================================================
# 分析2: 按通路分组的误差统计 (验证哪些通路预测得好/差)
# ============================================================================
print("\n" + "="*100)
print("分析2: 按通路分组的误差统计")
print("="*100)

pathway_errors = []
for method in methods:
    df = method_log2fc_data[method]
    for pathway in pathway_order:
        # 获取该通路的所有反应
        pathway_reactions = [r for r in df.columns if reaction_pathway_map.get(r, 'Other') == pathway]
        if len(pathway_reactions) > 0:
            values = df[pathway_reactions].values.flatten()
            values = values[~np.isnan(values)]
            if len(values) > 0:
                abs_values = np.abs(values)
                pathway_errors.append({
                    'Method': method,
                    'Pathway': pathway,
                    'Mean_AbsLog2FC': abs_values.mean(),
                    'Median_AbsLog2FC': np.median(abs_values),
                    'Std_AbsLog2FC': abs_values.std(),
                    'N': len(values)
                })

pathway_error_df = pd.DataFrame(pathway_errors)

# 创建分组条形图
fig, ax = plt.subplots(figsize=(16, 8))

# 准备绘图数据
pathway_list = pathway_order
x = np.arange(len(pathway_list))
width = 0.25

for i, method in enumerate(methods):
    method_data = pathway_error_df[pathway_error_df['Method'] == method]
    means = [method_data[method_data['Pathway'] == p]['Mean_AbsLog2FC'].values[0]
             if len(method_data[method_data['Pathway'] == p]) > 0 else 0
             for p in pathway_list]
    stds = [method_data[method_data['Pathway'] == p]['Std_AbsLog2FC'].values[0]
            if len(method_data[method_data['Pathway'] == p]) > 0 else 0
            for p in pathway_list]

    ax.bar(x + i*width, means, width, label=method, alpha=0.8, yerr=stds, color=methods_color[i],
           # 误差棒样式全写进 error_kw
           error_kw={'ecolor': methods_color[i],   # 误差棒颜色
                     'capsize': 3,
                     'capthick': 1.2,
                     'elinewidth': 1.2})
ax.tick_params(axis='both',          # 同时改 x、y 轴
               direction='in', width=4)       # 关键：in 表示向内
# 图例（loc 可自选，fontsize 是关键）
legend = ax.legend(fontsize=25)
# 为图例文字设置对应的methods颜色
for i, text in enumerate(legend.get_texts()):
    if i < len(methods_color):
        text.set_color(methods_color[i])
# 如果想连带 y 轴刻度一起加大
ax.tick_params(axis='y', labelsize=20)
# ax.set_xlabel('Pathway', fontsize=12, fontweight='bold')
# ax.set_ylabel('Mean |Log2FC|', fontsize=12, fontweight='bold')
# ax.set_title('Prediction Error by Metabolic Pathway',
#              fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(pathway_list, fontsize=20)
# 为每个标签设置对应的pathway颜色
for tick_label, pathway in zip(ax.get_xticklabels(), pathway_list):
    tick_label.set_color(pathway_colors.get(pathway, '#000000'))

# ax.legend()
# ax.grid(axis='y', alpha=0.3)
# ax.axhline(y=0.3, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Good (<0.3)')
# ax.axhline(y=1.0, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Acceptable (<1.0)')

plt.tight_layout()
output_path = os.path.join(output_dir, '2_pathway_error_comparison.png')
output_path = os.path.join(output_dir, '2_pathway_error_comparison.pdf')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 已保存: {output_path}")
plt.close()

# 打印通路统计
print("\n按通路的误差统计 (Mean |Log2FC|):")
for pathway in pathway_list:
    print(f"\n{pathway}:")
    for method in methods:
        method_data = pathway_error_df[(pathway_error_df['Method'] == method) &
                                       (pathway_error_df['Pathway'] == pathway)]
        if len(method_data) > 0:
            print(f"  {method}: {method_data['Mean_AbsLog2FC'].values[0]:.3f} ± {method_data['Std_AbsLog2FC'].values[0]:.3f}")

# ============================================================================
# 分析3: 通路水平的过高估计vs过低估计分析
# ============================================================================
print("\n" + "="*100)
print("="*100)

fig, axes = plt.subplots(3, 1, figsize=(14, 16))

for idx, method in enumerate(methods):
    ax = axes[idx]
    df = method_log2fc_data[method]

    pathway_bias = []
    for pathway in pathway_list:
        pathway_reactions = [r for r in df.columns if reaction_pathway_map.get(r, 'Other') == pathway]
        if len(pathway_reactions) > 0:
            values = df[pathway_reactions].values.flatten()
            values = values[~np.isnan(values)]
            if len(values) > 0:
                overestimation = (values > 1).sum()  # log2FC > 1 表示预测是实测的2倍以上
                underestimation = (values < -1).sum()  # log2FC < -1 表示预测不到实测的一半
                accurate = ((values >= -1) & (values <= 1)).sum()
                pathway_bias.append({
                    'Pathway': pathway,
                    'Overestimation': overestimation,
                    'Underestimation': -underestimation,  # 负数方便堆叠
                    'Accurate': accurate
                })

    bias_df = pd.DataFrame(pathway_bias)

    # 创建堆叠条形图
    x_pos = np.arange(len(bias_df))
    p1 = ax.bar(x_pos, bias_df['Overestimation'], color='#e74c3c', alpha=0.8, label='Overestimation (>2x)')
    p2 = ax.bar(x_pos, bias_df['Underestimation'], color='#3498db', alpha=0.8, label='Underestimation (<0.5x)')
    p3 = ax.bar(x_pos, bias_df['Accurate'], bottom=bias_df['Underestimation'],
                color='#95a5a6', alpha=0.5, label='Accurate (0.5-2x)')

    ax.set_ylabel('Number of Flux Measurements', fontsize=11, fontweight='bold')
    ax.set_title(f'{method} - Systematic Bias per Pathway', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bias_df['Pathway'], rotation=45, ha='right')
    # 为每个标签设置对应的pathway颜色
    for tick_label, pathway in zip(ax.get_xticklabels(), pathway_list):
        tick_label.set_color(pathway_colors.get(pathway, '#000000'))
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.legend(loc='upper right')
    ax.tick_params(axis='both',          # 同时改 x、y 轴
               direction='in', width=2)       # 关键：in 表示向内
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
output_path = os.path.join(output_dir, '3_pathway_systematic_bias.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 已保存: {output_path}")
plt.close()

# ============================================================================
# 分析4: 特定敲除株的通路响应模式 (雷达图)
# ============================================================================
print("\n" + "="*100)
print("分析4: 关键敲除株的通路响应模式")
print("="*100)

key_knockouts = ['pgi', 'zwf', 'gnd', 'tkt', 'eda', 'pfkA', 'pyk', 'ptsG', 'wildtype']
available_knockouts = []
for ko in key_knockouts:
    for strain in method_log2fc_data[methods[0]].index:
        if ko.lower() in strain.lower():
            available_knockouts.append(strain)
            break

if len(available_knockouts) > 0:
    print(f"找到的关键敲除株: {available_knockouts}")

    # 为每个方法创建雷达图
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw=dict(projection='polar'))

    for method_idx, method in enumerate(methods):
        ax = axes[method_idx]
        df = method_log2fc_data[method]

        # 计算每个敲除株在各通路的平均log2FC
        angles = np.linspace(0, 2 * np.pi, len(pathway_list), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        for strain in available_knockouts[:5]:  # 最多画5个
            if strain in df.index:
                values = []
                for pathway in pathway_list:
                    pathway_reactions = [r for r in df.columns if reaction_pathway_map.get(r, 'Other') == pathway]
                    if len(pathway_reactions) > 0:
                        strain_values = df.loc[strain, pathway_reactions].values
                        strain_values = strain_values[~np.isnan(strain_values)]
                        if len(strain_values) > 0:
                            values.append(np.mean(strain_values))
                        else:
                            values.append(0)
                    else:
                        values.append(0)

                values += values[:1]  # 闭合
                ax.plot(angles, values, 'o-', linewidth=2, label=strain, alpha=0.7)
                ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(pathway_list, size=9)
        # 为每个标签设置对应的pathway颜色
        for tick_label, pathway in zip(ax.get_xticklabels(), pathway_list):
            tick_label.set_color(pathway_colors.get(pathway, '#000000'))

        ax.set_ylim(-2, 2)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax.set_title(f'{method}\nPathway Response Pattern', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, '4_knockout_pathway_radar.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存: {output_path}")
    plt.close()
else:
    print("未找到关键敲除株")

# ============================================================================
# 分析5: 方法间相关性分析
# ============================================================================
print("\n" + "="*100)
print("分析5: 三种方法的预测相关性分析")
print("="*100)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

method_pairs = [
    ('Eki2vivo', 'EkiLLm'),
    ('Eki2vivo', 'fba'),
    ('EkiLLm', 'fba')
]

for idx, (method1, method2) in enumerate(method_pairs):
    ax = axes[idx]

    df1 = method_log2fc_data[method1]
    df2 = method_log2fc_data[method2]

    # 获取共同的样本和反应
    common_strains = list(set(df1.index) & set(df2.index))
    common_rxns = list(set(df1.columns) & set(df2.columns))

    values1 = []
    values2 = []
    for strain in common_strains:
        for rxn in common_rxns:
            v1 = df1.loc[strain, rxn]
            v2 = df2.loc[strain, rxn]
            if pd.notna(v1) and pd.notna(v2):
                values1.append(v1)
                values2.append(v2)

    # 散点图
    ax.scatter(values1, values2, alpha=0.3, s=10)
    ax.plot([-5, 5], [-5, 5], 'r--', linewidth=2, label='Perfect agreement')

    # 计算相关系数
    corr = np.corrcoef(values1, values2)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=12, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(f'{method1} Log2FC', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{method2} Log2FC', fontsize=11, fontweight='bold')
    ax.set_title(f'{method1} vs {method2}', fontsize=12, fontweight='bold')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    ax.legend()

plt.tight_layout()
output_path = os.path.join(output_dir, '5_method_correlation.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 已保存: {output_path}")
plt.close()

# ============================================================================
# 分析6: 最难预测的敲除株排名
# ============================================================================
print("\n" + "="*100)
print("分析6: 最难预测的敲除株排名 (验证TCA相关敲除株是否最难预测)")
print("="*100)

fig, axes = plt.subplots(3, 1, figsize=(14, 16))

for idx, method in enumerate(methods):
    ax = axes[idx]
    df = method_log2fc_data[method]

    # 计算每个株的平均绝对误差
    strain_errors = []
    for strain in df.index:
        values = df.loc[strain].values
        values = values[~np.isnan(values)]
        if len(values) > 0:
            mean_abs_error = np.mean(np.abs(values))
            strain_errors.append({'Strain': strain, 'Mean_AbsLog2FC': mean_abs_error})

    strain_error_df = pd.DataFrame(strain_errors)
    strain_error_df = strain_error_df.sort_values('Mean_AbsLog2FC', ascending=False)

    # 只显示前20个
    top20 = strain_error_df.head(20)

    # 为TCA相关的敲除株着色
    tca_genes = ['sdh', 'cyt', 'ndh', 'fum', 'mdh', 'citA', 'icd', 'suc']
    colors = ['#e74c3c' if any(gene in strain.lower() for gene in tca_genes)
              else '#3498db' for strain in top20['Strain']]

    ax.barh(range(len(top20)), top20['Mean_AbsLog2FC'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20['Strain'], fontsize=9)
    ax.set_xlabel('Mean |Log2FC|', fontsize=11, fontweight='bold')
    ax.set_title(f'{method} - Top 20 Hardest-to-Predict Strains\n(red=TCA relative, bluc=other)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.axvline(x=1, color='orange', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
output_path = os.path.join(output_dir, '6_hardest_strains.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 已保存: {output_path}")
plt.close()

# 打印TCA相关株的统计
print("\nTCA相关敲除株的误差统计:")
for method in methods:
    df = method_log2fc_data[method]
    tca_strains = [s for s in df.index if any(gene in s.lower() for gene in tca_genes)]
    if len(tca_strains) > 0:
        tca_values = []
        for strain in tca_strains:
            values = df.loc[strain].values
            values = values[~np.isnan(values)]
            if len(values) > 0:
                tca_values.extend(np.abs(values))
        print(f"  {method}: 平均 |Log2FC| = {np.mean(tca_values):.3f}")

# ============================================================================
# ============================================================================
print("\n" + "="*100)
print("分析7: 热力图颜色分布的定量统计")
print("="*100)

for method in methods:
    print(f"\n{method}:")
    df = method_log2fc_data[method]

    # Z-score标准化（列标准化 - 与热力图一致）
    df_zscore = df.copy()
    for col in df_zscore.columns:
        values = df_zscore[col].values
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() > 1:
            valid_values = values[valid_mask]
            z_scores = stats.zscore(valid_values)
            df_zscore.loc[valid_mask, col] = z_scores

    all_zscores = df_zscore.values.flatten()
    all_zscores = all_zscores[~np.isnan(all_zscores)]

    # 统计颜色区域
    white_region = ((all_zscores >= -1) & (all_zscores <= 1)).sum()
    red_region = (all_zscores > 1).sum()
    blue_region = (all_zscores < -1).sum()

    total = len(all_zscores)
    print(f"  白色区域 (±1 Z-score): {white_region} ({white_region/total*100:.1f}%)")
    print(f"  红色区域 (>1 Z-score): {red_region} ({red_region/total*100:.1f}%)")
    print(f"  蓝色区域 (<-1 Z-score): {blue_region} ({blue_region/total*100:.1f}%)")

# ============================================================================
# 分析7: 热力图颜色分布的定量统计 – 竖排三色饼图
# ============================================================================
print("\n" + "="*100)
print("分析7: 热力图颜色分布的定量统计 – 竖排三色饼图")
print("="*100)

# 预计算三种方法的颜色区域占比
method_color_stats = {}
for method in methods:
    df = method_log2fc_data[method]

    # 按列Z-score标准化（与热力图一致）
    df_zscore = df.copy()
    for col in df_zscore.columns:
        values = df_zscore[col].values
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() > 1:
            valid_values = values[valid_mask]
            z_scores = stats.zscore(valid_values)
            df_zscore.loc[valid_mask, col] = z_scores

    all_zscores = df_zscore.values.flatten()
    all_zscores = all_zscores[~np.isnan(all_zscores)]

    white = ((all_zscores >= -1) & (all_zscores <= 1)).sum()
    red   = (all_zscores > 1).sum()
    blue  = (all_zscores < -1).sum()
    total = len(all_zscores)

    method_color_stats[method] = {
        'white': white,
        'red'  : red,
        'blue' : blue,
        'total': total
    }

# 绘图参数
colors = {'white': 'white', 'red': '#ED949A', 'blue': '#96CCEA'}
labels = {'white': 'White\n(±1 Z-score)',
          'red'  : 'Red\n(>1 Z-score)',
          'blue' : 'Blue\n(<-1 Z-score)'}

# 竖排布局：3行1列
fig, axes = plt.subplots(3, 1, figsize=(5, 12))
axes = axes.flatten()

for ax, method in zip(axes, methods):
    stats = method_color_stats[method]
    sizes = [stats['white'], stats['red'], stats['blue']]
    total = stats['total']

    # 计算百分比
    pct = {k: stats[k] / total * 100 for k in ['white', 'red', 'blue']}

    # 统一 wedge 边框：细黑边，保证白色区域可见
    wedgeprops = {'linewidth': 1.2, 'edgecolor': 'black'}

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,
        colors=[colors['white'], colors['red'], colors['blue']],
        wedgeprops=wedgeprops,
        autopct=lambda p: f'{p:.1f}%',
        textprops={'color': 'black', 'weight': 'bold', 'fontsize': 11}
    )

    # 下方加粗方法名
    ax.text(0, -1.4, method, ha='center', va='center',
            fontsize=14, weight='bold', transform=ax.transAxes)

    # 手动图例
    legend_labels = [labels[k] for k in ['white', 'red', 'blue']]
    ax.legend(wedges, legend_labels,
              title="Region",
              loc='center left',
              bbox_to_anchor=(1.05, 0.5),
              frameon=False,
              fontsize=10,
              title_fontsize=11)

plt.tight_layout()
output_path = os.path.join(output_dir, '7_heatmap_color_pies_vertical.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ 已保存: {output_path}")
plt.close()
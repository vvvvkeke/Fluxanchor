import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm
from matplotlib.patches import Patch, FancyBboxPatch
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配色
colors = ["#96CCEA", "white", "#ED949A"]
COLORMAP = LinearSegmentedColormap.from_list("custom", colors)
FLUX_THRESHOLD = 0

# ============================================================================
# 物种配置
# ============================================================================
ORGANISM = "E. coli"

if ORGANISM == "E. coli":
    wildtype_csv_path = '/home/huangjiesheng/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/13C_analysis_data_extend_reactions.csv'
    detailed_csv_path = '/home/huangjiesheng/kcat_km_predict/results_extend_reactions/iECDH1ME8569_1439/detailed.csv'
    output_dir = '/home/huangjiesheng/kcat_km_predict/analysis/draw7'
elif ORGANISM == "B. subtilis":
    wildtype_csv_path = '/home/huangjiesheng/kcat_km_predict/predict/Bacillus subtilis/analysis/13C_analysis_data_threshold_0.01.csv'
    detailed_csv_path = '/home/huangjiesheng/kcat_km_predict/results_threshold_0.01/Bacillus_subtilis/detailed.csv'
    output_dir = '/home/huangjiesheng/kcat_km_predict/analysis/draw_bacillus'

os.makedirs(output_dir, exist_ok=True)

print("=" * 100)
print("无Log变换的真实Fold Change热力图")
print("=" * 100)

# ============================================================================
# 读取数据
# ============================================================================
print("\n正在读取数据...")
wildtype_df = pd.read_csv(wildtype_csv_path)
detailed_df = pd.read_csv(detailed_csv_path)
print(f"Wildtype数据形状: {wildtype_df.shape}")
print(f"详细数据形状: {detailed_df.shape}")

reaction_names = wildtype_df['Reaction'].tolist()
print(f"反应总数: {len(reaction_names)}")

# ============================================================================
# 构建真实Fold Change矩阵（不做log变换）
# ============================================================================
def build_fold_change_matrix(wildtype_df, detailed_df, reaction_names, method_name, method_wildtype_col, flux_threshold=FLUX_THRESHOLD):
    """构建真实fold change矩阵（pred/true），不做log变换"""
    epsilon = 1e-6

    # 1. 处理wildtype数据
    print(f"  处理wildtype数据...")
    wildtype_fc = {}
    for idx, row in wildtype_df.iterrows():
        reaction = row['Reaction']
        true_val = row['Value']
        pred_val = row[method_wildtype_col]

        if pd.notna(true_val) and pd.notna(pred_val):
            if abs(true_val) < flux_threshold:
                wildtype_fc[reaction] = 1.0  # 无变化
            else:
                # 直接计算fold change，不做log
                fc = (pred_val + epsilon) / (true_val + epsilon)
                wildtype_fc[reaction] = fc
        else:
            wildtype_fc[reaction] = 1.0

    # 2. 处理突变型数据
    method_df = detailed_df[detailed_df['Method'] == method_name].copy()
    print(f"  {method_name}: 找到 {len(method_df)} 个突变株")

    fc_data = {'wildtype': wildtype_fc}

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
                        gene_data[reaction] = 1.0
                    else:
                        fc = (pred_val + epsilon) / (true_val + epsilon)
                        gene_data[reaction] = fc
                else:
                    gene_data[reaction] = 1.0

        fc_data[gene] = gene_data

    fc_df = pd.DataFrame(fc_data).T
    return fc_df

# 构建数据
print("\n正在构建Fold Change数据（无log变换）...")
methods = ['FluxAnchor', 'KinLLM', 'fba']
methods_wildtype = ['FluxAnchor', 'KinLLM', 'FBA']
method_fc_data = {}

for method, method_wildtype in zip(methods, methods_wildtype):
    fc_df = build_fold_change_matrix(wildtype_df, detailed_df, reaction_names, method, method_wildtype)
    method_fc_data[method] = fc_df
    print(f"    {method}: {fc_df.shape}")

# 找到共同反应
common_reactions = set(reaction_names)
for method in methods:
    common_reactions = common_reactions.intersection(set(method_fc_data[method].columns))
common_reactions = sorted(list(common_reactions))
print(f"\n共同反应数: {len(common_reactions)}")

for method in methods:
    method_fc_data[method] = method_fc_data[method][common_reactions]

# ============================================================================
# 路径注释（与原代码相同）
# ============================================================================
pathway_annotations = {
    'Glycolysis': [
        "PGI", "PFK", "FBA", "TPI", "GAPD", "PGK", "PGM", "ENO", "GLCptspp", "PYK",
        "FBA_reverse", "PGK_reverse", "ENO_reverse", "GAPD_reverse", "TPI_reverse",
        "PGI_reverse", "PGM_reverse",
    ],
    'PP': [
        "G6PDH2r", "PGL", "GND", "RPE", "RPI", "TKT1", "TKT2", "TALA",
        "G6PDH2r_reverse", "RPE_reverse", "RPI_reverse", "TKT1_reverse",
        "TKT2_reverse", "TALA_reverse",
    ],
    'ED': ["EDA", "EDD"],
    'TCA': [
        "PDH", "CS", "ACONTa", "ACONTb", "ICDHyr", "AKGDH", "SUCOAS", "SUCDi",
        "FUM", "MDH", "ACONTa_reverse", "ACONTb_reverse", "ICDHyr_reverse",
        "SUCOAS_reverse", "FUM_reverse", "MDH_reverse",
    ],
    'Gluconeogenesis': ["FBP", "PPCK", "ME1", "ME2", "PPC"],
    'Amino acid': [
        "CHORS", "DHQS", "DHQTi", "DHAD1", "IPMD", "IPPS", "PSERT", "PSP_L",
        "SERAT", "SERAT_reverse", "DAPDC", "DAPE", "DHDPS", "DHDPRy", "DAPE_reverse",
    ],
    'Fatty acid': [
        'ACCOAC',
        'ACACT1r', 'ACACT2r', 'ACACT3r', 'ACACT4r', 'ACACT5r', 'ACACT6r', 'ACACT7r', 'ACACT8r',
        'ACACT1r_reverse', 'ACACT2r_reverse', 'ACACT3r_reverse', 'ACACT4r_reverse',
        'ACACT5r_reverse', 'ACACT6r_reverse', 'ACACT7r_reverse', 'ACACT8r_reverse',
        'ECOAH1', 'ECOAH2', 'ECOAH3', 'ECOAH4', 'ECOAH5', 'ECOAH6', 'ECOAH7', 'ECOAH8',
        'ECOAH1_reverse', 'ECOAH2_reverse', 'ECOAH3_reverse', 'ECOAH4_reverse',
        'ECOAH5_reverse', 'ECOAH6_reverse', 'ECOAH7_reverse', 'ECOAH8_reverse',
        'HACD1', 'HACD2', 'HACD3', 'HACD4', 'HACD5', 'HACD6', 'HACD7', 'HACD8',
        'HACD1_reverse', 'HACD2_reverse', 'HACD3_reverse', 'HACD4_reverse',
        'HACD5_reverse', 'HACD6_reverse', 'HACD7_reverse', 'HACD8_reverse',
    ],
}

pathway_order = ['Glycolysis', 'PP', 'ED', 'TCA', 'Gluconeogenesis', 'Amino acid', 'Fatty acid', 'Other']

pathway_colors = {
    'Glycolysis': "#2E9914", 'PP': "#db349b", 'ED': "#F2B342", 'TCA': '#925A44',
    'Gluconeogenesis': '#CD61D7', 'Amino acid': '#f39c12', 'Fatty acid': '#9b59b6', 'Other': '#95a5a6',
}

# 为每个反应分配路径
reaction_pathway_map = {}
for reaction in common_reactions:
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

def get_fixed_reaction_order(pathway_annotations, pathway_order, available_reactions):
    sorted_columns = []
    used_reactions = set()
    for pathway in pathway_order:
        if pathway in pathway_annotations:
            for reaction in pathway_annotations[pathway]:
                if reaction in available_reactions and reaction not in used_reactions:
                    sorted_columns.append(reaction)
                    used_reactions.add(reaction)
    for reaction in available_reactions:
        if reaction not in used_reactions:
            sorted_columns.append(reaction)
    return sorted_columns

# ============================================================================
# 统计数据分布
# ============================================================================
print("\n" + "=" * 80)
print("数据分布统计（真实Fold Change，pred/true）")
print("=" * 80)

for method in methods:
    df = method_fc_data[method]
    all_values = df.values.flatten()
    all_values = all_values[~np.isnan(all_values)]

    print(f"\n{method}:")
    print(f"  样本数: {len(df)}, 反应数: {len(df.columns)}")
    print(f"  最小值: {np.min(all_values):.4f}")
    print(f"  最大值: {np.max(all_values):.4f}")
    print(f"  中位数: {np.median(all_values):.4f}")
    print(f"  均值: {np.mean(all_values):.4f}")
    print(f"  标准差: {np.std(all_values):.4f}")
    print(f"  四分位数: Q1={np.percentile(all_values, 25):.4f}, Q3={np.percentile(all_values, 75):.4f}")
    print(f"  百分位数: 5%={np.percentile(all_values, 5):.4f}, 95%={np.percentile(all_values, 95):.4f}")

    # 统计误差分布
    accurate = np.sum((all_values >= 0.5) & (all_values <= 2.0))  # 0.5-2倍算准确
    underpredict = np.sum(all_values < 0.5)  # 预测偏低
    overpredict = np.sum(all_values > 2.0)  # 预测偏高
    total = len(all_values)

    print(f"  误差分布:")
    print(f"    准确 (0.5-2倍): {accurate} ({accurate/total*100:.1f}%)")
    print(f"    预测偏低 (<0.5倍): {underpredict} ({underpredict/total*100:.1f}%)")
    print(f"    预测偏高 (>2倍): {overpredict} ({overpredict/total*100:.1f}%)")

# ============================================================================
# 绘制无log热力图
# ============================================================================
radius = 0.5

def make_heatmap_rounded_squares(ax, data_df, cmap, vmin, vmax, radius=0.15, linewidth=0.5, edgecolor='white', use_log_norm=False):
    """将heatmap的每个小方块替换为圆角正方形"""
    import matplotlib.colors as mcolors

    for c in ax.collections:
        c.set_visible(False)

    ny, nx = data_df.shape

    if use_log_norm:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(ny):
        for j in range(nx):
            val = data_df.iloc[i, j]
            if pd.isna(val):
                color = 'white'
            else:
                # 对于log scale，需要clip到合理范围
                if use_log_norm:
                    val_clipped = np.clip(val, vmin, vmax)
                else:
                    val_clipped = val
                color = cmap(norm(val_clipped))

            rect = FancyBboxPatch(
                (j, i), 1, 1,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                clip_on=False
            )
            ax.add_patch(rect)

def create_no_log_heatmap(method, fc_df, reaction_pathway_map, pathway_colors, pathway_order, output_dir):
    """创建无log变换的热力图，显示真实fold change"""
    print(f"\n{'='*80}")
    print(f"生成 {method} 的无log热力图")
    print(f"{'='*80}")

    df = fc_df.copy()
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # 按固定顺序排列
    sorted_columns = get_fixed_reaction_order(pathway_annotations, pathway_order, df.columns.tolist())
    df = df[sorted_columns]
    df_filled = df.fillna(1.0)

    col_colors = [pathway_colors.get(reaction_pathway_map.get(r, 'Other'), '#95a5a6')
                  for r in df_filled.columns]

    n_rows, n_cols = df_filled.shape
    cell = 0.2
    fig_w = n_cols * cell + 2.0
    fig_h = n_rows * cell + 1.5

    # ========== 方案1: 线性刻度，截断极端值 ==========
    # 将数据截断到合理范围 [0.1, 10]
    df_clipped = df_filled.clip(lower=0.1, upper=10)

    g = sns.clustermap(df_clipped,
                       row_cluster=False,
                       col_cluster=False,
                       cmap=COLORMAP,
                       center=1.0,  # 中心是1（无变化）
                       vmin=0.1, vmax=10,
                       cbar_pos=None,
                       figsize=(fig_w, fig_h),
                       col_colors=col_colors,
                       xticklabels=False,
                       yticklabels=True,
                       linewidths=0,
                       dendrogram_ratio=0.0)

    g.ax_heatmap.set_aspect('equal')

    # 圆角方块
    make_heatmap_rounded_squares(g.ax_heatmap, df_clipped,
                                 cmap=COLORMAP, vmin=0.1, vmax=10,
                                 radius=radius, linewidth=0, edgecolor='white')

    for label in g.ax_heatmap.get_yticklabels():
        label.set_weight('bold')

    pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])

    heat_pos = g.ax_heatmap.get_position()
    col_pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([heat_pos.x0, col_pos.y0, heat_pos.width, col_pos.height])

    # 图例
    legend_elements = [Patch(facecolor=pathway_colors[p], label=p)
                       for p in pathway_order if p in reaction_pathway_map.values()]
    legend = g.ax_heatmap.legend(handles=legend_elements,
                                loc='upper left', bbox_to_anchor=(1.02, 1),
                                frameon=True, title='Pathway', fontsize=9)
    for txt in legend.get_texts():
        txt.set_weight('bold')
        txt.set_color(pathway_colors.get(txt.get_text(), '#000'))
    legend.get_title().set_weight('bold')

    # Colorbar
    legend_bbox = legend.get_window_extent(g.fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(g.fig.transFigure.inverted())
    cbar_ax = g.fig.add_axes([legend_bbox.x0, legend_bbox.y0 - 0.40,
                              legend_bbox.width, 0.02])
    norm = mpl.colors.Normalize(vmin=0.1, vmax=10)
    sm = mpl.cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    sm.set_array([])
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Fold Change (pred/true)', fontsize=10, weight='bold')
    cbar.set_ticks([0.1, 0.5, 1, 2, 5, 10])
    cbar.set_ticklabels(['0.1x', '0.5x', '1x', '2x', '5x', '10x'])
    for l in cbar.ax.get_xticklabels():
        l.set_weight('bold')

    out_path = os.path.join(output_dir, f'{method}_heatmap_no_log_linear.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"已保存（线性刻度，截断到0.1-10倍）: {out_path}")
    plt.close()

    # ========== 方案2: 对称log刻度 ==========
    # 使用SymLogNorm来处理围绕1的对称数据
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # 将数据转换为以1为中心的对数刻度
    # 使用 log2(fc) 但显示原始倍数
    df_for_plot = np.log2(df_filled.clip(lower=0.01, upper=100))

    g2 = sns.clustermap(df_for_plot,
                        row_cluster=False,
                        col_cluster=False,
                        cmap=COLORMAP,
                        center=0,  # log2(1) = 0
                        vmin=-4, vmax=4,  # log2(0.0625) to log2(16)
                        cbar_pos=None,
                        figsize=(fig_w, fig_h),
                        col_colors=col_colors,
                        xticklabels=False,
                        yticklabels=True,
                        linewidths=0,
                        dendrogram_ratio=0.0)

    g2.ax_heatmap.set_aspect('equal')

    make_heatmap_rounded_squares(g2.ax_heatmap, df_for_plot,
                                 cmap=COLORMAP, vmin=-4, vmax=4,
                                 radius=radius, linewidth=0, edgecolor='white')

    for label in g2.ax_heatmap.get_yticklabels():
        label.set_weight('bold')

    pos = g2.ax_col_colors.get_position()
    g2.ax_col_colors.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])

    heat_pos = g2.ax_heatmap.get_position()
    col_pos = g2.ax_col_colors.get_position()
    g2.ax_col_colors.set_position([heat_pos.x0, col_pos.y0, heat_pos.width, col_pos.height])

    legend = g2.ax_heatmap.legend(handles=legend_elements,
                                 loc='upper left', bbox_to_anchor=(1.02, 1),
                                 frameon=True, title='Pathway', fontsize=9)
    for txt in legend.get_texts():
        txt.set_weight('bold')
        txt.set_color(pathway_colors.get(txt.get_text(), '#000'))
    legend.get_title().set_weight('bold')

    legend_bbox = legend.get_window_extent(g2.fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(g2.fig.transFigure.inverted())
    cbar_ax = g2.fig.add_axes([legend_bbox.x0, legend_bbox.y0 - 0.40,
                               legend_bbox.width, 0.02])
    norm = mpl.colors.Normalize(vmin=-4, vmax=4)
    sm = mpl.cm.ScalarMappable(cmap=COLORMAP, norm=norm)
    sm.set_array([])
    cbar = g2.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Fold Change (pred/true)', fontsize=10, weight='bold')
    # 显示真实倍数
    cbar.set_ticks([-4, -2, -1, 0, 1, 2, 4])
    cbar.set_ticklabels(['1/16x', '1/4x', '1/2x', '1x', '2x', '4x', '16x'])
    for l in cbar.ax.get_xticklabels():
        l.set_weight('bold')

    out_path2 = os.path.join(output_dir, f'{method}_heatmap_no_log_symlog.pdf')
    plt.savefig(out_path2, dpi=300, bbox_inches='tight')
    print(f"已保存（对称log刻度，显示倍数）: {out_path2}")
    plt.close()

    # 保存原始fold change数据
    csv_path = os.path.join(output_dir, f'{method}_fold_change_no_log.csv')
    df.to_csv(csv_path)
    print(f"已保存原始fold change数据: {csv_path}")

    return df

# 为每个方法生成热力图
print("\n" + "=" * 100)
print("开始生成热力图...")
print("=" * 100)

for method in methods:
    create_no_log_heatmap(method, method_fc_data[method], reaction_pathway_map,
                          pathway_colors, pathway_order, output_dir)

# ============================================================================
# 绘制误差分布直方图
# ============================================================================
print("\n" + "=" * 100)
print("绘制误差分布直方图...")
print("=" * 100)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, method in enumerate(methods):
    ax = axes[idx]
    df = method_fc_data[method]
    all_values = df.values.flatten()
    all_values = all_values[~np.isnan(all_values)]

    # 使用log刻度的直方图
    log_values = np.log2(np.clip(all_values, 0.001, 1000))

    ax.hist(log_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect (1x)')
    ax.axvline(x=1, color='orange', linestyle='--', linewidth=1.5, label='2x error')
    ax.axvline(x=-1, color='orange', linestyle='--', linewidth=1.5, label='0.5x error')

    ax.set_xlabel('Log2(Fold Change)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title(f'{method}\nFold Change Distribution', fontsize=14, fontweight='bold')

    # 添加真实倍数的次坐标轴
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    tick_positions = [-4, -2, -1, 0, 1, 2, 4]
    tick_labels = ['1/16x', '1/4x', '1/2x', '1x', '2x', '4x', '16x']
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=9)
    ax2.set_xlabel('Fold Change', fontsize=10, fontweight='bold')

    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(output_dir, 'fold_change_distribution.pdf')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"已保存误差分布图: {out_path}")
plt.close()

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 100)
print("完成！")
print("=" * 100)
print(f"\n输出目录: {output_dir}")
print("\n生成的文件:")
print("  1. *_heatmap_no_log_linear.pdf  - 线性刻度热力图 (截断到0.1-10倍)")
print("  2. *_heatmap_no_log_symlog.pdf  - 对称log刻度热力图 (显示真实倍数)")
print("  3. *_fold_change_no_log.csv     - 原始fold change数据")
print("  4. fold_change_distribution.pdf - 误差分布直方图")
print("\n刻度说明:")
print("  - 1x = 预测准确")
print("  - 2x = 预测是真实值的2倍")
print("  - 0.5x = 预测是真实值的一半")
print("  - 蓝色 = 预测偏低，白色 = 准确，红色 = 预测偏高")

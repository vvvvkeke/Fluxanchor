"""
基于 13C_combined_metric_heatmap2.py 改写
指标说明：
  方案2：flux-weighted log2FC (归一化到 [0, 1])
  方案3：signed SMAPE (取绝对值，范围 [0, 1])

优化点：
  1. 颜色映射：深蓝色(#003399)表示误差最小(0)，白色表示误差最大(1)。
  2. 字体大小：显著增大了左侧基因标注的字体，方便论文排版。
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, FancyBboxPatch
import matplotlib as mpl

warnings.filterwarnings('ignore')

# ============================================================================
# 全局样式配置
# ============================================================================
radius = 0.5
# 核心修改：深蓝色代表 0 (准确)，白色代表 1 (误差大)
colors = ["#003399", "white"] 
COLORMAP = LinearSegmentedColormap.from_list("custom", colors)
FLUX_THRESHOLD = 0

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 路径配置
# ============================================================================
wildtype_csv_path = '/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/13C_analysis_data_extend_reactions.csv'
detailed_csv_path = '/home/zhangyangyu/kcat_km_predict/results_extend_reactions/iECDH1ME8569_1439/detailed.csv'
output_dir = '/home/zhangyangyu/kcat_km_predict/analysis/combined_metric_output'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 圆角方块绘制函数
# ============================================================================
def make_heatmap_rounded_squares(ax, data_df, cmap, vmin=0, vmax=1,
                                  radius=0.15, linewidth=0.5, edgecolor='white'):
    import matplotlib.colors as mcolors
    for c in ax.collections:
        c.set_visible(False)
    ny, nx = data_df.shape
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(ny):
        for j in range(nx):
            val = data_df.iloc[i, j]
            # 如果是 NaN 则画白色，否则根据 cmap 映射
            color = 'white' if pd.isna(val) else cmap(norm(val))
            rect = FancyBboxPatch(
                (j, i), 1, 1,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=color, edgecolor=edgecolor,
                linewidth=linewidth, clip_on=False
            )
            ax.add_patch(rect)

# ============================================================================
# 数据处理逻辑 (计算 TAU)
# ============================================================================
def compute_tau(wildtype_df, detailed_df, reaction_names):
    all_true_vals = []
    for v in wildtype_df['Value']:
        if pd.notna(v) and abs(v) > FLUX_THRESHOLD: all_true_vals.append(abs(v))
    for rxn in reaction_names:
        col = f"{rxn}_true"
        if col in detailed_df.columns:
            for v in detailed_df[col]:
                if pd.notna(v) and abs(v) > FLUX_THRESHOLD: all_true_vals.append(abs(v))
    return np.median(all_true_vals) if all_true_vals else 1.0

# ============================================================================
# 矩阵构建函数
# ============================================================================
def build_weighted_log2fc_matrix(wildtype_df, detailed_df, reaction_names, method_name, method_wildtype_col, tau):
    epsilon = 1e-5
    def weighted_score(pred_val, true_val):
        log2fc = np.log2((pred_val + epsilon) / (true_val + epsilon))
        weight = abs(true_val) / (abs(true_val) + tau)
        return log2fc * weight

    wildtype_row = {row['Reaction']: (0.0 if abs(row['Value']) <= FLUX_THRESHOLD else weighted_score(row[method_wildtype_col], row['Value'])) 
                    for _, row in wildtype_df.iterrows()}
    method_df = detailed_df[detailed_df['Method'] == method_name]
    rows = {'wildtype': wildtype_row}
    for _, row in method_df.iterrows():
        gene_row = {rxn: (weighted_score(row[f"{rxn}_pred"], row[f"{rxn}_true"]) if f"{rxn}_true" in method_df.columns and abs(row[f"{rxn}_true"]) > FLUX_THRESHOLD else 0.0) 
                    for rxn in reaction_names}
        rows[row['Gene']] = gene_row
    return pd.DataFrame(rows).T

def build_smape_matrix(wildtype_df, detailed_df, reaction_names, method_name, method_wildtype_col):
    epsilon = 1e-5
    def signed_smape(pred_val, true_val):
        num, den = abs(pred_val - true_val), (abs(pred_val) + abs(true_val)) / 2.0 + epsilon
        return np.sign(pred_val - true_val) * np.clip(num / den, 0, 2)

    wildtype_row = {row['Reaction']: (0.0 if abs(row['Value']) <= FLUX_THRESHOLD else signed_smape(row[method_wildtype_col], row['Value'])) 
                    for _, row in wildtype_df.iterrows()}
    method_df = detailed_df[detailed_df['Method'] == method_name]
    rows = {'wildtype': wildtype_row}
    for _, row in method_df.iterrows():
        gene_row = {rxn: (signed_smape(row[f"{rxn}_pred"], row[f"{rxn}_true"]) if f"{rxn}_true" in method_df.columns and abs(row[f"{rxn}_true"]) > FLUX_THRESHOLD else 0.0) 
                    for rxn in reaction_names}
        rows[row['Gene']] = gene_row
    return pd.DataFrame(rows).T

# ============================================================================
# 通路配置
# ============================================================================
pathway_annotations = {
    'Glycolysis': ["PGI", "PFK", "FBA", "TPI", "GAPD", "PGK", "PGM", "ENO", "GLCptspp", "PYK", "FBA_reverse", "PGK_reverse", "ENO_reverse", "GAPD_reverse", "TPI_reverse", "PGI_reverse", "PGM_reverse"],
    'PP': ["G6PDH2r", "PGL", "GND", "RPE", "RPI", "TKT1", "TKT2", "TALA", "G6PDH2r_reverse", "RPE_reverse", "RPI_reverse", "TKT1_reverse", "TKT2_reverse", "TALA_reverse"],
    'ED': ["EDA", "EDD"],
    'TCA': ["PDH", "CS", "ACONTa", "ACONTb", "ICDHyr", "AKGDH", "SUCOAS", "SUCDi", "FUM", "MDH", "ACONTa_reverse", "ACONTb_reverse", "ICDHyr_reverse", "SUCOAS_reverse", "FUM_reverse", "MDH_reverse", "ICL", "MALS"],
    'Gluconeogenesis': ["FBP", "PPCK", "ME1", "ME2", "PPC"],
    'Amino acid': ["CHORS", "DHQS", "DHQTi", "DHAD1", "IPMD", "IPPS", "PSERT", "PSP_L", "SERAT", "SERAT_reverse", "DAPDC", "DAPE", "DHDPS", "DHDPRy", "DAPE_reverse"],
    'Fatty acid': ['ACCOAC', 'ACACT1r', 'ACACT2r', 'ACACT3r', 'ACACT4r', 'ACACT5r', 'ACACT6r', 'ACACT7r', 'ACACT8r', 'ACACT1r_reverse', 'ACACT2r_reverse', 'ACACT3r_reverse', 'ACACT4r_reverse', 'ACACT5r_reverse', 'ACACT6r_reverse', 'ACACT7r_reverse', 'ACACT8r_reverse', 
                   'ECOAH1', 'ECOAH2', 'ECOAH3', 'ECOAH4', 'ECOAH5', 'ECOAH6', 'ECOAH7', 'ECOAH8', 'ECOAH1_reverse', 'ECOAH2_reverse', 'ECOAH3_reverse', 'ECOAH4_reverse', 'ECOAH5_reverse', 'ECOAH6_reverse', 'ECOAH7_reverse', 'ECOAH8_reverse', 
                   'HACD1', 'HACD2', 'HACD3', 'HACD4', 'HACD5', 'HACD6', 'HACD7', 'HACD8', 'HACD1_reverse', 'HACD2_reverse', 'HACD3_reverse', 'HACD4_reverse', 'HACD5_reverse', 'HACD6_reverse', 'HACD7_reverse', 'HACD8_reverse'],
}
pathway_order = ['Glycolysis', 'PP', 'ED', 'TCA', 'Gluconeogenesis', 'Amino acid', 'Fatty acid', 'Other']
pathway_colors = {'Glycolysis': "#2E9914", 'PP': "#db349b", 'ED': "#F2B342", 'TCA': '#925A44', 'Gluconeogenesis': '#CD61D7', 'Amino acid': '#f39c12', 'Fatty acid': '#9b59b6', 'Other': '#95a5a6'}

def get_fixed_reaction_order(available_reactions):
    used, order = set(), []
    for pathway in pathway_order:
        for rxn in pathway_annotations.get(pathway, []):
            if rxn in available_reactions and rxn not in used:
                order.append(rxn); used.add(rxn)
    for rxn in available_reactions:
        if rxn not in used: order.append(rxn)
    return order

def get_reaction_pathway(reaction):
    for pathway, keywords in pathway_annotations.items():
        if any(k in reaction for k in keywords): return pathway
    return 'Other'

# ============================================================================
# 热力图绘制主函数
# ============================================================================
def create_metric_heatmap(method, score_df, reaction_pathway_map,
                           pathway_colors, pathway_order, output_dir,
                           metric_name, cbar_label, vmin=0, vmax=1):
    df = score_df.copy().abs()
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    sorted_columns = get_fixed_reaction_order(df.columns.tolist())
    df = df[sorted_columns].fillna(0)

    # 布局参数：增大 cell 保证清晰度
    cell = 0.1 
    fig_w = df.shape[1] * cell
    fig_h = df.shape[0] * cell 

    col_colors = [pathway_colors.get(get_reaction_pathway(r), '#95a5a6') for r in df.columns]

    g = sns.clustermap(df, row_cluster=False, col_cluster=False, cmap=COLORMAP,
                       vmin=vmin, vmax=vmax, cbar_pos=None, figsize=(fig_w, fig_h),
                       col_colors=col_colors, xticklabels=False, yticklabels=True,
                       linewidths=0, dendrogram_ratio=0.0)

    g.ax_heatmap.set_aspect('equal')
    make_heatmap_rounded_squares(g.ax_heatmap, df, cmap=COLORMAP, vmin=vmin, vmax=vmax,
                                  radius=radius, linewidth=0, edgecolor='white')

    # 修改：显著增大左侧基因标注字体
    for label in g.ax_heatmap.get_yticklabels():
        label.set_weight('bold')
        label.set_fontsize(8) 

    # 调整位置
    pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])
    heat_pos = g.ax_heatmap.get_position()
    g.ax_col_colors.set_position([heat_pos.x0, g.ax_col_colors.get_position().y0, heat_pos.width, g.ax_col_colors.get_position().height])

    # 图例与 Colorbar
    present_pathways = set(get_reaction_pathway(r) for r in df.columns)
    legend_elements = [Patch(facecolor=pathway_colors[p], label=p) for p in pathway_order if p in present_pathways]
    legend = g.ax_heatmap.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                                  frameon=True, title='Pathway', fontsize=11)
    for txt in legend.get_texts():
        txt.set_weight('bold')
        txt.set_color(pathway_colors.get(txt.get_text(), '#000'))
    legend.get_title().set_weight('bold')

    legend_bbox = legend.get_window_extent(g.fig.canvas.get_renderer()).transformed(g.fig.transFigure.inverted())
    cbar_ax = g.fig.add_axes([legend_bbox.x0, legend_bbox.y0 - 0.35, legend_bbox.width, 0.02])
    sm = mpl.cm.ScalarMappable(cmap=COLORMAP, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(cbar_label, fontsize=12, weight='bold')
    for l in cbar.ax.get_xticklabels(): l.set_weight('bold')

    plt.savefig(os.path.join(output_dir, f'{method}_{metric_name}.pdf'), dpi=300, bbox_inches='tight')
    plt.close('all')

# ============================================================================
# 主程序
# ============================================================================
print("读取数据中...")
wildtype_df = pd.read_csv(wildtype_csv_path)
detailed_df = pd.read_csv(detailed_csv_path)
reaction_names = wildtype_df['Reaction'].tolist()
TAU = compute_tau(wildtype_df, detailed_df, reaction_names)

methods, methods_wt = ['FluxAnchor', 'KinLLM', 'fba'], ['FluxAnchor', 'KinLLM', 'FBA']

# 方案2：Flux-weighted Log2FC
weighted_data = {m: build_weighted_log2fc_matrix(wildtype_df, detailed_df, reaction_names, m, wt, TAU) 
                 for m, wt in zip(methods, methods_wt)}
common_rxns = sorted(list(set(reaction_names).intersection(*(set(weighted_data[m].columns) for m in methods))))
wl_99 = max(np.nanpercentile(np.abs(weighted_data[m].values), 99) for m in methods)

for m in methods:
    data = (weighted_data[m][common_rxns] / wl_99).clip(upper=1)
    create_metric_heatmap(m, data, None, pathway_colors, pathway_order, output_dir, 'weighted_log2FC', 'Normalized Error')

# 方案3：SMAPE
smape_data = {m: build_smape_matrix(wildtype_df, detailed_df, reaction_names, m, wt) 
              for m, wt in zip(methods, methods_wt)}
for m in methods:
    # SMAPE 天然范围 0-2，这里取绝对值并 clip 到 1 观察
    data = smape_data[m][common_rxns].abs().clip(upper=1)
    create_metric_heatmap(m, data, None, pathway_colors, pathway_order, output_dir, 'signed_SMAPE', 'Error Scale')

print(f"\n任务完成！输出目录: {output_dir}")
print("颜色说明：深蓝色代表误差为0(准确)，白色代表误差大。")
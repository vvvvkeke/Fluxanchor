"""
简单的 abs(pred - true) 热力图
不做任何 log 变换，直接看绝对差值
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
from matplotlib.patches import Patch, FancyBboxPatch
import matplotlib as mpl

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

radius = 0.5

def make_heatmap_rounded_squares(ax, data_df, cmap, vmin=0, vmax=3,
                                  radius=0.15, linewidth=0.5, edgecolor='white'):
    import matplotlib.colors as mcolors
    for c in ax.collections:
        c.set_visible(False)
    ny, nx = data_df.shape
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(ny):
        for j in range(nx):
            val = data_df.iloc[i, j]
            color = 'white' if pd.isna(val) else cmap(norm(val))
            rect = FancyBboxPatch(
                (j, i), 1, 1,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=color, edgecolor=edgecolor,
                linewidth=linewidth, clip_on=False
            )
            ax.add_patch(rect)

# ============================================================================
# 路径配置（和原脚本一样）
# ============================================================================
wildtype_csv_path = '/home/huangjiesheng/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/13C_analysis_data_extend_reactions.csv'
detailed_csv_path = '/home/huangjiesheng/kcat_km_predict/results_extend_reactions/iECDH1ME8569_1439/detailed.csv'
output_dir = '/home/huangjiesheng/kcat_km_predict/analysis/abs_diff_output'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# 读取数据
# ============================================================================
print("读取数据...")
wildtype_df = pd.read_csv(wildtype_csv_path)
detailed_df = pd.read_csv(detailed_csv_path)
reaction_names = wildtype_df['Reaction'].tolist()
print(f"反应数: {len(reaction_names)}, 突变株数: {len(detailed_df)}")

# ============================================================================
# 构建 abs(pred - true) 矩阵
# ============================================================================
def build_abs_diff_matrix(wildtype_df, detailed_df, reaction_names, method_name, method_wildtype_col, flux_threshold=0):
    """每个格子 = abs(pred - true)，flux 接近 0 的格子填 NaN"""

    # wildtype 行
    wildtype_row = {}
    for _, row in wildtype_df.iterrows():
        reaction = row['Reaction']
        true_val = row['Value']
        pred_val = row[method_wildtype_col]
        if pd.notna(true_val) and pd.notna(pred_val) and abs(true_val) >= flux_threshold:
            wildtype_row[reaction] = abs(pred_val - true_val)
        else:
            wildtype_row[reaction] = np.nan

    # 突变株各行
    method_df = detailed_df[detailed_df['Method'] == method_name].copy()
    print(f"  {method_name}: {len(method_df)} 个突变株")

    rows = {'wildtype': wildtype_row}
    for _, row in method_df.iterrows():
        gene = row['Gene']
        gene_row = {}
        for reaction in reaction_names:
            true_col = f"{reaction}_true"
            pred_col = f"{reaction}_pred"
            if true_col in method_df.columns and pred_col in method_df.columns:
                true_val = row[true_col]
                pred_val = row[pred_col]
                if pd.notna(true_val) and pd.notna(pred_val) and abs(true_val) >= flux_threshold:
                    gene_row[reaction] = abs(pred_val - true_val)
                else:
                    gene_row[reaction] = np.nan
            else:
                gene_row[reaction] = np.nan
        rows[gene] = gene_row

    return pd.DataFrame(rows).T  # shape: (samples, reactions)

# ============================================================================
# 路径注释（和原脚本一样，用于列颜色）
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

def get_reaction_pathway(reaction):
    for pathway, keywords in pathway_annotations.items():
        if any(k in reaction for k in keywords):
            return pathway
    return 'Other'

def get_fixed_reaction_order(available_reactions):
    used = set()
    order = []
    for pathway in pathway_order:
        for rxn in pathway_annotations.get(pathway, []):
            if rxn in available_reactions and rxn not in used:
                order.append(rxn)
                used.add(rxn)
    for rxn in available_reactions:
        if rxn not in used:
            order.append(rxn)
    return order

# ============================================================================
# 构建数据
# ============================================================================
methods = ['FluxAnchor', 'KinLLM', 'fba']
methods_wildtype_col = ['FluxAnchor', 'KinLLM', 'FBA']

all_data = {}
for method, wt_col in zip(methods, methods_wildtype_col):
    all_data[method] = build_abs_diff_matrix(wildtype_df, detailed_df, reaction_names, method, wt_col)

# 找共同反应列
common_reactions = set(reaction_names)
for method in methods:
    common_reactions &= set(all_data[method].columns)
common_reactions = sorted(list(common_reactions))
print(f"共同反应数: {len(common_reactions)}")

for method in methods:
    all_data[method] = all_data[method][common_reactions]

# ============================================================================
# 画图：每个 method 一张图
# ============================================================================
from matplotlib.patches import Patch

for method in methods:
    df = all_data[method].copy()
    df = df.dropna(axis=0, how='all')

    # 排列列顺序
    sorted_cols = get_fixed_reaction_order(df.columns.tolist())
    df = df[sorted_cols]

    # 列颜色（pathway）
    col_colors = [pathway_colors.get(get_reaction_pathway(r), '#95a5a6') for r in df.columns]

    # 动态大小
    n_rows, n_cols = df.shape
    cell = 0.2
    fig_w = n_cols * cell + 1.5
    fig_h = n_rows * cell + 1.0

    # 上限用 95th percentile 截断，防止极端值把颜色压死
    vmax = np.nanpercentile(df.values, 95)
    vmax = max(vmax, 1e-6)

    print(f"\n{method}: shape={df.shape}, vmax(95th)={vmax:.4f}")

    g = sns.clustermap(
        df.fillna(0),
        row_cluster=False,
        col_cluster=False,
        cmap='Reds',          # 0 = 白色, 大 = 红色，直观
        vmin=0,
        vmax=vmax,
        cbar_pos=None,
        figsize=(fig_w, fig_h),
        col_colors=col_colors,
        xticklabels=False,
        yticklabels=True,
        linewidths=0,
        dendrogram_ratio=0.0,
    )

    g.ax_heatmap.set_aspect('equal')

    make_heatmap_rounded_squares(g.ax_heatmap, df.fillna(0),
                                  cmap=plt.cm.Reds, vmin=0, vmax=vmax,
                                  radius=radius, linewidth=0, edgecolor='white')

    for label in g.ax_heatmap.get_yticklabels():
        label.set_weight('bold')

    # 对齐列颜色条和热力图（先上移，再对齐宽度）
    pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])

    heat_pos = g.ax_heatmap.get_position()
    col_pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([heat_pos.x0, col_pos.y0, heat_pos.width, col_pos.height])

    # Pathway 图例
    present_pathways = sorted(set(get_reaction_pathway(r) for r in df.columns))
    legend_elements = [Patch(facecolor=pathway_colors[p], label=p) for p in pathway_order if p in present_pathways]
    legend = g.ax_heatmap.legend(
        handles=legend_elements,
        loc='upper left', bbox_to_anchor=(1.02, 1),
        frameon=True, title='Pathway', fontsize=9,
    )
    for txt in legend.get_texts():
        txt.set_weight('bold')
        txt.set_color(pathway_colors.get(txt.get_text(), '#000'))
    legend.get_title().set_weight('bold')

    # Colorbar
    legend_bbox = legend.get_window_extent(g.fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(g.fig.transFigure.inverted())
    cbar_ax = g.fig.add_axes([legend_bbox.x0, legend_bbox.y0 - 0.35, legend_bbox.width, 0.02])
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap='Reds', norm=norm)
    sm.set_array([])
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('|pred - true|  (mmol/gDW/h)', fontsize=10, weight='bold')
    for l in cbar.ax.get_xticklabels():
        l.set_weight('bold')

    out_path = os.path.join(output_dir, f'{method}_abs_diff_heatmap.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"保存: {out_path}")
    plt.close('all')

# ============================================================================
# 三方法并排对比图（用各自 95th percentile 归一化后同尺度）
# ============================================================================
print("\n绘制三方法对比图...")

# 统一 vmax（所有方法的 95th percentile 取最大）
global_vmax = max(
    np.nanpercentile(all_data[m].values, 95) for m in methods
)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, method in zip(axes, methods):
    df = all_data[method].copy()
    df = df.dropna(axis=0, how='all')
    sorted_cols = get_fixed_reaction_order(df.columns.tolist())
    df = df[sorted_cols]

    all_vals = df.values.flatten()
    all_vals = all_vals[~np.isnan(all_vals)]
    print(f"{method}: mean={np.mean(all_vals):.4f}, median={np.median(all_vals):.4f}, max={np.max(all_vals):.4f}")

    im = ax.imshow(df.fillna(0).values, aspect='auto', cmap='Reds', vmin=0, vmax=global_vmax)
    ax.set_title(f'{method}', fontsize=13, weight='bold')
    ax.set_xlabel('Reactions', fontsize=11)
    ax.set_ylabel('Samples', fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label='|pred - true|')

plt.suptitle('abs(pred - true)  flux error heatmap  (same color scale)', fontsize=13, weight='bold')
plt.tight_layout()
out_path = os.path.join(output_dir, 'all_methods_abs_diff_compare.pdf')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"保存对比图: {out_path}")
plt.close('all')

print("\n完成！")

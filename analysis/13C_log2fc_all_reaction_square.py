import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')
from matplotlib.patches import Patch   
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as patches
import matplotlib as mpl


radius = 0.5 # 圆角半径
def make_heatmap_rounded_squares(ax, data_df, cmap, vmin=-3, vmax=3, radius=0.15, linewidth=0.5, edgecolor='white'):
    """
    将 heatmap 的每个小方块替换为圆角正方形，使用自定义 colormap
    """
    import matplotlib.colors as mcolors

    # 清除原始热力图的小方块
    for c in ax.collections:
        c.set_visible(False)

    ny, nx = data_df.shape
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(ny):
        for j in range(nx):
            val = data_df.iloc[i, j]
            if pd.isna(val):
                color = 'white'
            else:
                color = cmap(norm(val))

            rect = FancyBboxPatch(
                (j , i ), 1, 1,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                facecolor=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                clip_on=False
            )
            ax.add_patch(rect)

# 设置中文字体（如果需要）
# plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# viridis 丑到家了
# RdBu_r 勉强一看
# BrBG 次了点
# GnBu 配色还挺好看，但是在这里好像不太适合，单边数据比较适合
# 配置  RdBu_r  viridis
# RdYlBu_r  # 不是很明显
colors = ["#96CCEA", "white", "#ED949A"]
# colors = ["#73a8d5", "white", "#e283a6"]

COLORMAP = LinearSegmentedColormap.from_list("custom", colors)
# COLORMAP = 'RdBu_r'  # 蓝色=预测偏低，白色=准确，红色=预测偏高
FLUX_THRESHOLD = 0   #设置np.nan的阈值

# ============================================================================
# 物种选择（超参数）
# ============================================================================
# 可选值: "E. coli" 或 "B. subtilis"
# ORGANISM = "B. subtilis"  # 修改此参数来切换物种
ORGANISM = "E. coli"  # 修改此参数来切换物种
# ============================================================================

# 根据物种配置文件路径
if ORGANISM == "E. coli":
    # wildtype_csv_path = '/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/13C_analysis_data.csv'
    # detailed_csv_path = '/home/zhangyangyu/kcat_km_predict/results/iECDH1ME8569_1439/detailed.csv'
    wildtype_csv_path = '/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/13C_analysis_data_extend_reactions.csv'
    detailed_csv_path = '/home/zhangyangyu/kcat_km_predict/results_extend_reactions/iECDH1ME8569_1439/detailed.csv'
    # wildtype_csv_path = '/home/zhangyangyu/kcat_km_predict/predict/iECDH1ME8569_1439/analysis/13C_analysis_data_threshold_0.01.csv'
    # detailed_csv_path = '/home/zhangyangyu/kcat_km_predict/results_threshold_0.01/iECDH1ME8569_1439/detailed.csv'
    output_dir = '/home/zhangyangyu/kcat_km_predict/analysis/draw7'
elif ORGANISM == "B. subtilis":
    wildtype_csv_path = '/home/zhangyangyu/kcat_km_predict/predict/Bacillus subtilis/analysis/13C_analysis_data_threshold_0.01.csv'
    detailed_csv_path = '/home/zhangyangyu/kcat_km_predict/results_threshold_0.01/Bacillus_subtilis/detailed.csv'
    output_dir = '/home/zhangyangyu/kcat_km_predict/analysis/draw_bacillus'
else:
    raise ValueError(f"不支持的物种: {ORGANISM}。请选择 'E. coli' 或 'B. subtilis'")

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

print("=" * 100)
print("单细胞风格代谢热力图分析")
print("=" * 100)

# ============================================================================
# Step 1: 读取数据并构建log2 fold change矩阵
# ============================================================================
print("\n正在读取数据...")
wildtype_df = pd.read_csv(wildtype_csv_path)
detailed_df = pd.read_csv(detailed_csv_path)
print(f"Wildtype数据形状: {wildtype_df.shape}")
print(f"详细数据形状: {detailed_df.shape}")

# 从wildtype数据中获取反应名称
reaction_names = wildtype_df['Reaction'].tolist()
print(f"从wildtype数据提取的反应总数: {len(reaction_names)}")

def build_log2fc_matrix_with_wildtype(wildtype_df, detailed_df, reaction_names, method_name, method_wildtype_col, flux_threshold=FLUX_THRESHOLD):
    """构建log2 fold change矩阵，包含wildtype数据"""
    epsilon = 1e-5

    # 1. 处理wildtype数据
    print(f"  处理wildtype数据...")
    wildtype_log2fc = {}
    for idx, row in wildtype_df.iterrows():
        reaction = row['Reaction']
        true_val = row['Value']
        pred_val = row[method_wildtype_col]

        if pd.notna(true_val) and pd.notna(pred_val):
            if abs(true_val) < flux_threshold:
                wildtype_log2fc[reaction] = 0
            else:
                log2_fc = np.log2((pred_val + epsilon) / (true_val + epsilon))
                wildtype_log2fc[reaction] = log2_fc
        else:
            wildtype_log2fc[reaction] = 0

    # 2. 处理突变型数据
    method_df = detailed_df[detailed_df['Method'] == method_name].copy()
    print(f"  {method_name}: 找到 {len(method_df)} 个突变株")

    log2fc_data = {'wildtype': wildtype_log2fc}  # 先添加wildtype

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
                        gene_data[reaction] = 0
                    else:
                        log2_fc = np.log2((pred_val + epsilon) / (true_val + epsilon))
                        gene_data[reaction] = log2_fc
                else:
                    gene_data[reaction] = 0

        log2fc_data[gene] = gene_data

    log2fc_df = pd.DataFrame(log2fc_data).T
    return log2fc_df

# 为三种方法构建数据（包含wildtype）
print("\n正在构建log2 fold change数据（包含wildtype）...")
methods = ['Eki2vivo', 'EkiLLm', 'fba']
methods_wildtype = ['Eki2vivo', 'EkiLLM', 'FBA']  # wildtype CSV中的列名
method_log2fc_data = {}

for method, method_wildtype in zip(methods, methods_wildtype):
    log2fc_df = build_log2fc_matrix_with_wildtype(wildtype_df, detailed_df, reaction_names, method, method_wildtype)
    method_log2fc_data[method] = log2fc_df
    print(f"    {method}: {log2fc_df.shape} (包含wildtype)")

# 找到所有方法的共同反应
common_reactions = set(reaction_names)
for method in methods:
    common_reactions = common_reactions.intersection(set(method_log2fc_data[method].columns))

common_reactions = sorted(list(common_reactions))
print(f"\n所有方法的共同反应数: {len(common_reactions)}")

# 过滤到共同反应
for method in methods:
    method_log2fc_data[method] = method_log2fc_data[method][common_reactions]
    print(f"  {method}: {method_log2fc_data[method].shape}")

# ============================================================================
# Step 2: 定义代谢路径注释（基于反应名称）
# ============================================================================
print("\n正在定义代谢路径注释...")

# 定义关键代谢路径的反应
pathway_annotations = {
    'Glycolysis': [
        "PGI", # 磷酸葡萄糖异构酶 pehtose-6-phosphate isomerase
        "PFK", # 磷酸果糖激酶 phosphofructokinase
        "FBA", # 果糖二磷酸醛缩酶 fructose-bisphosphate aldolase
        "TPI", # 磷酸丙糖异构酶 triose phosphate isomerase
        "GAPD", # 甘油醛-3-磷酸脱氢酶 glyceraldehyde-3-phosphate dehydrogenase
        "PGK",
        "PGM", # 磷酸甘油酸变位酶 phosphoglycerate mutase
        "ENO", 
        "GLCptspp",  
        "PYK",
        "FBA_reverse",  # FBA 逆流
        "PGK_reverse",  # PGK 逆流：磷酸化底物合成
        "ENO_reverse",  # ENO 逆流
        "GAPD_reverse", # GAPD 逆流
        "TPI_reverse",  # TPI 逆流（磷酸二羟丙酮/甘油醛-3-磷酸互变）
        "PGI_reverse",  # PGI 逆流（葡萄糖-6-磷酸/果糖-6-磷酸互变）
        "PGM_reverse",  # PGM 逆流（磷酸甘油酸变位酶）
    ],
    'PP': [
        "G6PDH2r", "PGL", "GND", "RPE", "RPI", "TKT1", "TKT2", "TALA",
        "G6PDH2r_reverse",  # 葡萄糖-6-磷酸脱氢酶逆流
        "RPE_reverse",   # 核酮糖-5-磷酸-3-差向异构酶
        "RPI_reverse",   # 核糖-5-磷酸异构酶
        "TKT1_reverse",  # 可逆转酮醇酶反应
        "TKT2_reverse",  # 可逆转酮醇酶反应
        "TALA_reverse",  # 可逆转醛醇酶反应
    ],
    'ED':[
        "EDA", # 2-脱氢-3-脱氧-磷酸葡萄糖酸醛缩酶
        "EDD", # 6-磷酸葡萄糖酸脱水酶
    ],
    'TCA': [
        "PDH", # 丙酮酸脱氢酶 pyruvate dehydrogenase
        "CS", # 柠檬酸合成酶 citrate synthase
        "ACONTa", # 顺乌头酸酶a aconitase a
        "ACONTb",  # 顺乌头酸酶b aconitase b
        "ICDHyr", # 异柠檬酸脱氢酶 isocitrate dehydrogenase
        "AKGDH", # α-酮戊二酸脱氢酶 alpha-ketoglutarate dehydrogenase
        "SUCOAS", # 琥珀酰辅酶A合成酶 succinyl-CoA synthetase
        "SUCDi", # 琥珀酸脱氢酶 succinate dehydrogenase
        "FUM", # 延胡索酸酶 fumarase
        "MDH", # 苹果酸脱氢酶同工酶
        "ACONTa_reverse", # 顺乌头酸酶a逆流
        "ACONTb_reverse", # 顺乌头酸酶b逆流
        "ICDHyr_reverse", # 异柠檬酸脱氢酶逆流
        "SUCOAS_reverse", # 琥珀酰辅酶A合成酶逆流
        "FUM_reverse",   # 延胡索酸酶逆流
        "MDH_reverse",   # 苹果酸脱氢酶逆流
    ],
    'Gluconeogenesis': ["FBP", "PPCK", "ME1", "ME2", "PPC"],
    'Amino acid': [
        # 芳香族氨基酸途径
        "CHORS", "DHQS", "DHQTi",
        # 支链氨基酸途径
        "DHAD1", "IPMD", "IPPS",
        # 丝氨酸/甘氨酸途径
        "PSERT", # phosphoserine transaminase
        "PSP_L", # phosphoserine phosphatase
        "SERAT", # serine acetyltransferase
        "SERAT_reverse", # serine acetyltransferase
        # 赖氨酸途径
        "DAPDC", # diaminopimelate decarboxylase
        "DAPE", # diaminopimelate epimerase
        "DHDPS", # dihydrodipicolinate synthase
        "DHDPRy", # dihydrodipicolinate reductase
        "DAPE_reverse",
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

pathway_order = ['Glycolysis', 'PP', 'ED', 'TCA', 'Gluconeogenesis',
                 'Amino acid', 'Fatty acid', 'Other',
                # 'Nucleotide', 'Fermentation',
                #  'Respiration', 'Exchange', 'Energy'
                 ]
                 

pathway_colors = {
    'Glycolysis': "#2E9914", 'PP': "#db349b", 'ED': "#F2B342", 'TCA': '#925A44',
    'Gluconeogenesis': '#CD61D7', 'Amino acid': '#f39c12', 
    'Fatty acid': '#9b59b6', 
    'Other': '#95a5a6',
    # 'Nucleotide': '#ee5a6f', 'Fermentation': '#48dbfb', 'Respiration': "#7D1B88",
    # 'Exchange': '#1abc9c', 'Energy': '#e67e22'
}

# 为每个反应分配路径标签
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

# 统计每个路径的反应数
pathway_counts = {}
for pathway in reaction_pathway_map.values():
    pathway_counts[pathway] = pathway_counts.get(pathway, 0) + 1

print("\n路径注释统计:")
for pathway, count in sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {pathway}: {count} 个反应")

# 创建路径颜色映射
# pathway_colors = {
#     'Glycolysis': '#e74c3c',                              # 红色
#     'PPP': '#3498db',                                      # 蓝色
#     'TCA': '#2ecc71',                                      # 绿色
#     'ED': '#ff6b6b',                                       # 浅红色
#     'Gluconeogenesis': '#feca57',                          # 黄色
#     'Amino_acid': '#f39c12',                               # 橙色
#     'Fatty acid': '#9b59b6',                               # 紫色
#     'Nucleotide Metabolism': '#ee5a6f',                    # 玫红色
#     'Fermentation & Anaerobic Metabolism': '#48dbfb',      # 浅蓝色
#     'Respiration & Electron Transport': "#7D1B88",         # 深蓝色
#     'Exchange': '#1abc9c',                                 # 青色
#     'Energy': '#e67e22',                                   # 橘黄色
#     'Other': '#95a5a6'                                     # 灰色
# }

# ============================================================================
# Step 3: 为每个方法创建单细胞风格的热力图
# ============================================================================

def get_fixed_reaction_order(pathway_annotations, pathway_order, available_reactions):
    """
    根据pathway_annotations中定义的固定顺序获取反应列表
    只返回available_reactions中存在的反应
    """
    sorted_columns = []
    used_reactions = set()

    for pathway in pathway_order:
        if pathway in pathway_annotations:
            for reaction in pathway_annotations[pathway]:
                if reaction in available_reactions and reaction not in used_reactions:
                    sorted_columns.append(reaction)
                    used_reactions.add(reaction)

    # 将未分配的反应（Other类别）添加到最后
    for reaction in available_reactions:
        if reaction not in used_reactions:
            sorted_columns.append(reaction)

    return sorted_columns

def create_single_cell_style_heatmap(method, log2fc_df, reaction_pathway_map,
                                     pathway_colors, pathway_order, output_dir, colormap=COLORMAP):
    """
    列标准化 + 圆角方格（无聚类，按固定顺序）
    """
    print(f"\n{'='*80}")
    print(f"生成 {method} 的单细胞风格热力图（列标准化+圆角，无聚类）")
    print(f"{'='*80}")

    # ---------- 1. 数据准备 ----------
    df = log2fc_df.copy()
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    print(f"过滤后形状: {df.shape}")

    # ---------- 2. 列 Z-score ----------
    from scipy import stats
    df_zscore = df.copy()
    for col in df_zscore.columns:
        vals = df_zscore[col].values
        mask = ~np.isnan(vals)
        if mask.sum() > 1:
            df_zscore.loc[mask, col] = stats.zscore(vals[mask])
    df_filled = df_zscore.fillna(0)

    # ---------- 3. 按固定顺序排列反应（不排序） ----------
    sorted_columns = get_fixed_reaction_order(pathway_annotations, pathway_order, df_filled.columns.tolist())
    df_filled = df_filled[sorted_columns]

    # ---------- 4. 不进行行聚类，保持原始顺序 ----------
    # row_linkage = linkage(df_filled.values, method='ward', metric='euclidean')

    # ---------- 5. clustermap 布局（无聚类） ----------
    col_colors = [pathway_colors.get(reaction_pathway_map.get(r, 'Other'), '#95a5a6')
                  for r in df_filled.columns]

    n_rows, n_cols = df_filled.shape
    cell = 0.2
    fig_w = n_cols * cell + 1.5
    fig_h = n_rows * cell + 1.0

    g = sns.clustermap(df_filled,
                       row_cluster=False,  # 不进行行聚类
                       col_cluster=False,  # 不进行列聚类
                       cmap=colormap,
                       center=0, vmin=-3, vmax=3,
                       cbar_pos=None,
                       figsize=(fig_w, fig_h),
                       col_colors=col_colors,
                       xticklabels=False,
                       yticklabels=True,
                       linewidths=0,
                       dendrogram_ratio=0.0)  # 不显示树状图

    g.ax_heatmap.set_aspect('equal')

    # ---------- 6. 圆角方格 ----------
    make_heatmap_rounded_squares(g.ax_heatmap, df_filled,
                                 cmap=colormap, vmin=-3, vmax=3,
                                 radius=radius, linewidth=0, edgecolor='white')

    # ---------- 7. 标签/颜色条对齐 ----------
    # 加粗 yticklabels
    for label in g.ax_heatmap.get_yticklabels():
        label.set_weight('bold')

    pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])

    heat_pos = g.ax_heatmap.get_position()
    col_pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([heat_pos.x0, col_pos.y0,
                                  heat_pos.width, col_pos.height])

    # ---------- 8. 标题 ----------
    # g.fig.suptitle(f'{method} - Heatmap (Column Z-score, Rounded Squares, No Clustering)',
    #                fontsize=18, fontweight='bold', y=0.98)

    # ---------- 9. 图例 & colorbar ----------
    legend_elements = [Patch(facecolor=pathway_colors[p], label=p)
                       for p in pathway_order if p in reaction_pathway_map.values()]
    legend = g.ax_heatmap.legend(handles=legend_elements,
                                loc='upper left', bbox_to_anchor=(1.02, 1),
                                frameon=True, title='Pathway', fontsize=9)
    for txt in legend.get_texts():
        txt.set_weight('bold')
        txt.set_color(pathway_colors.get(txt.get_text(), '#000'))
    legend.get_title().set_weight('bold')

    # colorbar 放在图例下方
    legend_bbox = legend.get_window_extent(g.fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(g.fig.transFigure.inverted())
    cbar_ax = g.fig.add_axes([legend_bbox.x0, legend_bbox.y0 - 0.35,
                              legend_bbox.width, 0.02])
    norm = mpl.colors.Normalize(vmin=-3, vmax=3)
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Z-score (Log2 FC)', fontsize=10, weight='bold')
    for l in cbar.ax.get_xticklabels():
        l.set_weight('bold')

    # ---------- 10. 保存 ----------
    out_path = os.path.join(output_dir,
                            f'{method}_heatmap_col_zscore_rounded.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"已保存（列标准化+圆角，无聚类）: {out_path}")
    plt.close()

    return df_zscore, df, sorted_columns

def create_row_normalized_heatmap(method, log2fc_df, sorted_columns,
                                   reaction_pathway_map, pathway_colors, pathway_order,
                                   output_dir, colormap=COLORMAP):
    """
    创建行标准化的热力图（无聚类，按固定顺序）

    步骤：
    1. 使用已排序的列顺序
    2. Z-score标准化（行标准化 - 每个样本）
    3. 不进行聚类，按固定顺序显示
    4. 添加路径注释条
    """
    print(f"\n{'='*80}")
    print(f"生成 {method} 的行标准化热力图（无聚类）")
    print(f"{'='*80}")

    # 使用传入的排序好的列
    df = log2fc_df[sorted_columns].copy()

    print(f"数据形状: {df.shape}")

    # Step 1: Z-score标准化（对行 - 每个样本）
    print("\nStep 1: Z-score标准化（对行标准化 - 每个样本）...")
    from scipy import stats

    df_zscore_row = df.copy()
    for idx in df_zscore_row.index:
        values = df_zscore_row.loc[idx].values
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() > 1:  # 至少需要2个有效值才能标准化
            valid_values = values[valid_mask]
            z_scores = stats.zscore(valid_values)
            df_zscore_row.loc[idx, valid_mask] = z_scores
        else:
            df_zscore_row.loc[idx] = values

    print(f"Z-score后的范围: [{df_zscore_row.min().min():.2f}, {df_zscore_row.max().max():.2f}]")

    # Step 2: 用0填充NaN
    df_filled_row = df_zscore_row.fillna(0)

    # Step 3: 创建热力图（无聚类）
    print("\nStep 2: 创建热力图（无聚类）...")

    # 准备路径注释条
    col_colors = [pathway_colors.get(reaction_pathway_map.get(r, 'Other'), '#95a5a6')
                  for r in df_zscore_row.columns]

    n_rows = len(df_filled_row)
    n_cols = len(df_filled_row.columns)
    # 每个格子的大小（单位：英寸）
    cell_size = 0.2  # 你可以调这个值，控制整体大小

    figwidth = n_cols * cell_size
    figheight = n_rows * cell_size

    # 额外加上空间
    figheight += 1.0
    figwidth += 1.5  # 右侧 colorbar + legend
    # 创建clustermap（无聚类）
    g = sns.clustermap(df_filled_row,
                       row_cluster=False,  # 不对行进行聚类
                       col_cluster=False,  # 不对列进行聚类
                       cmap=colormap,
                       center=0,
                       vmin=-3, vmax=3,
                       cbar_pos=None,  # 隐藏默认colorbar，稍后自定义位置
                       figsize=(figwidth, figheight),  # ✅ 关键：按比例设置
                       col_colors=col_colors,
                       xticklabels=False,
                       yticklabels=True,  # 先关掉，后面手动加粗
                       linewidths=0, # 方格之间的线宽
                       dendrogram_ratio=0.0)  # 不显示树状图

    # 强制正方形格子
    g.ax_heatmap.set_aspect('equal')

    # 使用你定义好的 COLORMAP 覆盖圆角方块
    make_heatmap_rounded_squares(
        g.ax_heatmap,
        df_filled_row,
        cmap=COLORMAP,  # ✅ 传入你自定义的 colormap
        vmin=-3,
        vmax=3,
        radius=radius,
        linewidth=0,
        edgecolor='white'
    )

    # === 加粗右边突变类型标签 ===
    for label in g.ax_heatmap.get_yticklabels():
        label.set_weight('bold')

    # === 强制 col_colors 宽度与热力图一致 ===
    heat_ax = g.ax_heatmap
    col_color_ax = g.ax_col_colors
    heat_pos = heat_ax.get_position()
    col_color_pos = col_color_ax.get_position()

    # 匹配宽度
    col_color_ax.set_position([
        heat_pos.x0,
        col_color_pos.y0,
        heat_pos.width,
        col_color_pos.height
    ])

    # === 绘制完圆角方块后刷新一次画布 ===
    g.fig.canvas.draw()

    # === 上移 col_colors 条，避免与热力图重叠 ===
    pos = col_color_ax.get_position()
    col_color_ax.set_position([
        pos.x0,
        pos.y0 + 0.01,  # 上移一点（可改 0.005~0.02 调整）
        pos.width,
        pos.height
    ])

    # === 匹配 col_colors 宽度与热力图一致 ===
    heat_pos = heat_ax.get_position()
    col_color_pos = col_color_ax.get_position()
    col_color_ax.set_position([
        heat_pos.x0,
        col_color_pos.y0,
        heat_pos.width,
        col_color_pos.height
    ])

    # 设置标题
    # g.fig.suptitle(f'{method} - Heatmap (Row Z-score normalized, No Clustering)',
    #                fontsize=18, fontweight='bold', y=0.98)

    # 添加图例（路径颜色）- 按照pathway_order排序
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=pathway_colors[pathway], label=pathway)
                      for pathway in pathway_order
                      if pathway in set(reaction_pathway_map.values())]

    legend = g.ax_heatmap.legend(handles=legend_elements,
                                loc='upper left',
                                bbox_to_anchor=(1.02, 1),
                                frameon=True,
                                title='Pathway',
                                fontsize=9)

    # 加粗图例中的标签文字
    for text in legend.get_texts():
        text.set_weight('bold')

    # 为图例文字设置对应的pathway颜色
    for text in legend.get_texts():
        pathway = text.get_text()
        text.set_color(pathway_colors.get(pathway, '#000000'))

    # 可选：加粗图例标题
    # legend.get_title().set_weight('bold')
    # 在Pathway图例下方添加colorbar
    # 获取图例的位置
    legend_bbox = legend.get_window_extent(g.fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(g.fig.transFigure.inverted())

    # 在图例下方创建colorbar的位置
    cbar_ax = g.fig.add_axes([legend_bbox.x0, legend_bbox.y0 - 0.35,
                              legend_bbox.width, 0.02])

    # 创建colorbar
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=-3, vmax=3)
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Z-score (Log2 FC)', fontsize=10, weight="bold")
    # ---------- 关键：把刻度数字加粗 ----------
    for l in cbar.ax.get_xticklabels():
        l.set_weight('bold')
    # 保存
    output_path = os.path.join(output_dir, f'{method}_heatmap_row_zscore.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存（行标准化，无聚类）: {output_path}")
    plt.close()

    return df_zscore_row

def create_no_normalization_heatmap(method, log2fc_df, sorted_columns,
                                    reaction_pathway_map, pathway_colors, pathway_order,
                                    output_dir, colormap=COLORMAP):
    """
    不归一化 + 圆角方格（无聚类，按固定顺序）
    """
    print(f"\n{'='*80}")
    print(f"生成 {method} 的不归一化热力图（圆角，无聚类）")
    print(f"{'='*80}")

    df = log2fc_df[sorted_columns].copy()
    df_filled = df.fillna(0)                     # 只用于绘图

    col_colors = [pathway_colors.get(reaction_pathway_map.get(r, 'Other'), '#95a5a6')
                  for r in df_filled.columns]

    n_rows, n_cols = df_filled.shape
    cell = 0.2
    fig_w = n_cols * cell + 1.5
    fig_h = n_rows * cell + 1.0

    g = sns.clustermap(df_filled,
                       row_cluster=False,  # 不进行行聚类
                       col_cluster=False,  # 不进行列聚类
                       cmap=colormap,
                       center=0, vmin=-5, vmax=5,
                       cbar_pos=None,
                       figsize=(fig_w, fig_h),
                       col_colors=col_colors,
                       xticklabels=False,
                       yticklabels=True,
                       linewidths=0,
                       dendrogram_ratio=0.0)  # 不显示树状图

    g.ax_heatmap.set_aspect('equal')

    # ---------- 圆角 ----------
    make_heatmap_rounded_squares(g.ax_heatmap, df_filled,
                                 cmap=colormap, vmin=-5, vmax=5,
                                 radius=radius, linewidth=0, edgecolor='white')

    # ---------- 对齐 ----------
    for label in g.ax_heatmap.get_yticklabels():
        label.set_weight('bold')

    pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])

    heat_pos = g.ax_heatmap.get_position()
    col_pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([heat_pos.x0, col_pos.y0,
                                  heat_pos.width, col_pos.height])

    # ---------- 标题 ----------
    # g.fig.suptitle(f'{method} - Heatmap (No Normalization, Rounded Squares, No Clustering)',
    #                fontsize=18, fontweight='bold', y=0.98)

    # ---------- 图例 & colorbar ----------
    legend_elements = [Patch(facecolor=pathway_colors[p], label=p)
                       for p in pathway_order if p in reaction_pathway_map.values()]
    legend = g.ax_heatmap.legend(handles=legend_elements,
                                loc='upper left', bbox_to_anchor=(1.02, 1),
                                frameon=True, title='Pathway', fontsize=9)
    for txt in legend.get_texts():
        txt.set_weight('bold')
        txt.set_color(pathway_colors.get(txt.get_text(), '#000'))
    legend.get_title().set_weight('bold')

    legend_bbox = legend.get_window_extent(g.fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(g.fig.transFigure.inverted())
    cbar_ax = g.fig.add_axes([legend_bbox.x0, legend_bbox.y0 - 0.35,
                              legend_bbox.width, 0.02])
    norm = mpl.colors.Normalize(vmin=-5, vmax=5)
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Log2 FC', fontsize=10, weight='bold')
    for l in cbar.ax.get_xticklabels():
        l.set_weight('bold')

    # ---------- 保存 ----------
    out_path = os.path.join(output_dir,
                            f'{method}_heatmap_no_norm_rounded.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"已保存（不归一化+圆角，无聚类）: {out_path}")
    plt.close()

    return df

# 为每个方法生成单细胞风格热力图（列标准化和行标准化）
method_zscore_data = {}
method_row_zscore_data = {}
for method in methods:
    # 生成列标准化热力图（无聚类）
    df_z, df_orig, sorted_cols = create_single_cell_style_heatmap(
        method, method_log2fc_data[method], reaction_pathway_map,
        pathway_colors, pathway_order, output_dir
    )
    method_zscore_data[method] = df_z

    # 保存列Z-score数据
    csv_path = os.path.join(output_dir, f'{method}_zscore_col_data.csv')
    df_z.to_csv(csv_path)
    print(f"已保存列Z-score数据: {csv_path}")

    # 生成行标准化热力图（无聚类，使用相同的列排序）
    df_z_row = create_row_normalized_heatmap(
        method, df_orig, sorted_cols,
        reaction_pathway_map, pathway_colors, pathway_order, output_dir
    )
    method_row_zscore_data[method] = df_z_row

    # 保存行Z-score数据
    csv_path_row = os.path.join(output_dir, f'{method}_zscore_row_data.csv')
    df_z_row.to_csv(csv_path_row)
    print(f"已保存行Z-score数据: {csv_path_row}")

    # 生成不归一化热力图（无聚类，使用相同的列排序）
    df_no_norm = create_no_normalization_heatmap(
        method, df_orig, sorted_cols,
        reaction_pathway_map, pathway_colors, pathway_order, output_dir
    )

    # 保存不归一化数据
    csv_path_no_norm = os.path.join(output_dir, f'{method}_no_normalization_data.csv')
    df_no_norm.to_csv(csv_path_no_norm)
    print(f"已保存不归一化数据: {csv_path_no_norm}")

# ============================================================================
# Step 3.5: 整合三个方法的数据，生成整合热力图
# ============================================================================
print("\n" + "="*100)
print("Step 3.5: 整合三个方法的数据")
print("="*100)

def create_integrated_heatmap(method_log2fc_data, methods, reaction_pathway_map,
                               pathway_colors, pathway_order, output_dir, colormap=COLORMAP):
    """
    整合三个方法的数据，生成行标准化和列标准化的热力图（无聚类，按固定顺序）
    """
    print("\n正在整合三个方法的数据...")

    # 合并三个方法的数据
    integrated_data = []
    for method in methods:
        df = method_log2fc_data[method].copy()
        # 为每个样本添加方法标识
        df.index = [f"{method}_{idx}" for idx in df.index]
        integrated_data.append(df)

    # 纵向合并
    df_integrated = pd.concat(integrated_data, axis=0)
    print(f"整合后数据形状: {df_integrated.shape}")
    print(f"  共 {len(df_integrated)} 个样本 ({len(methods)} 个方法 × {len(df_integrated)//len(methods)} 个样本/方法)")
    print(f"  共 {len(df_integrated.columns)} 个反应")

    # 移除全是NaN的行和列
    df_integrated = df_integrated.dropna(axis=0, how='all')
    df_integrated = df_integrated.dropna(axis=1, how='all')
    print(f"过滤后数据形状: {df_integrated.shape}")

    # ========== 列标准化 ==========
    print("\n" + "="*80)
    print("生成整合数据的列标准化热力图（无聚类）")
    print("="*80)

    # Z-score标准化（对列 - 每个反应）
    print("\nStep 1: Z-score标准化（对列标准化 - 每个反应）...")
    from scipy import stats

    df_zscore_col = df_integrated.copy()
    for col in df_zscore_col.columns:
        values = df_zscore_col[col].values
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() > 1:
            valid_values = values[valid_mask]
            z_scores = stats.zscore(valid_values)
            df_zscore_col.loc[valid_mask, col] = z_scores
        else:
            df_zscore_col[col] = values

    print(f"列Z-score后的范围: [{df_zscore_col.min().min():.2f}, {df_zscore_col.max().max():.2f}]")

    # 按固定顺序排列反应（不排序）
    print("\nStep 2: 按固定顺序排列反应...")
    sorted_columns = get_fixed_reaction_order(pathway_annotations, pathway_order, df_zscore_col.columns.tolist())

    df_zscore_col = df_zscore_col[sorted_columns]
    print(f"列排序完成，共 {len(sorted_columns)} 列")

    # 用0填充NaN
    df_filled_col = df_zscore_col.fillna(0)

    # 创建列标准化热力图（无聚类）
    print("\nStep 3: 创建整合热力图（列标准化，无聚类）...")
    col_colors = [pathway_colors.get(reaction_pathway_map.get(r, 'Other'), '#95a5a6')
                  for r in df_zscore_col.columns]

    g = sns.clustermap(df_filled_col,
                       row_cluster=False,  # 不进行行聚类
                       col_cluster=False,  # 不进行列聚类
                       cmap=colormap,
                       center=0,
                       vmin=-3, vmax=3,
                       cbar_pos=None,
                       figsize=(32, 16),
                       col_colors=col_colors,
                       xticklabels=False,
                       yticklabels=True,
                       linewidths=0,
                       dendrogram_ratio=0.0)  # 不显示树状图

    # g.fig.suptitle('Integrated Heatmap (Column Z-score normalized)\n(所有方法整合，按固定顺序，无聚类)',
    #                fontsize=20, fontweight='bold', y=0.99)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=pathway_colors[pathway], label=pathway)
                      for pathway in pathway_order
                      if pathway in set(reaction_pathway_map.values())]

    legend = g.ax_heatmap.legend(handles=legend_elements,
                                loc='upper left',
                                bbox_to_anchor=(1.02, 1),
                                frameon=True,
                                title='Pathway',
                                fontsize=9)

    # 在Pathway图例下方添加colorbar
    legend_bbox = legend.get_window_extent(g.fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(g.fig.transFigure.inverted())

    cbar_ax = g.fig.add_axes([legend_bbox.x0, legend_bbox.y0 - 0.25,
                              legend_bbox.width, 0.015])

    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=-3, vmax=3)
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Z-score (Log2 FC)', fontsize=10)

    output_path = os.path.join(output_dir, 'Integrated_heatmap_col_zscore.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存整合热力图（列标准化，无聚类）: {output_path}")
    plt.close()

    # 保存列Z-score数据
    csv_path = os.path.join(output_dir, 'Integrated_zscore_col_data.csv')
    df_zscore_col.to_csv(csv_path)
    print(f"已保存整合列Z-score数据: {csv_path}")

    # ========== 行标准化 ==========
    print("\n" + "="*80)
    print("生成整合数据的行标准化热力图（无聚类）")
    print("="*80)

    # 使用相同的列排序
    df_integrated_sorted = df_integrated[sorted_columns].copy()

    # Z-score标准化（对行 - 每个样本）
    print("\nStep 1: Z-score标准化（对行标准化 - 每个样本）...")
    df_zscore_row = df_integrated_sorted.copy()
    for idx in df_zscore_row.index:
        values = df_zscore_row.loc[idx].values
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() > 1:
            valid_values = values[valid_mask]
            z_scores = stats.zscore(valid_values)
            df_zscore_row.loc[idx, valid_mask] = z_scores
        else:
            df_zscore_row.loc[idx] = values

    print(f"行Z-score后的范围: [{df_zscore_row.min().min():.2f}, {df_zscore_row.max().max():.2f}]")

    # 用0填充NaN
    df_filled_row = df_zscore_row.fillna(0)

    # 创建行标准化热力图（无聚类）
    print("\nStep 2: 创建整合热力图（行标准化，无聚类）...")

    g = sns.clustermap(df_filled_row,
                       row_cluster=False,  # 不进行行聚类
                       col_cluster=False,  # 不进行列聚类
                       cmap=colormap,
                       center=0,
                       vmin=-3, vmax=3,
                       cbar_pos=None,
                       figsize=(32, 16),
                       col_colors=col_colors,
                       xticklabels=False,
                       yticklabels=True,
                       linewidths=0,
                       dendrogram_ratio=0.0)  # 不显示树状图

    # g.fig.suptitle('Integrated Heatmap (Row Z-score normalized)\n(所有方法整合，按固定顺序，无聚类，行标准化)',
    #                fontsize=20, fontweight='bold', y=0.99)

    # 添加图例
    legend = g.ax_heatmap.legend(handles=legend_elements,
                                loc='upper left',
                                bbox_to_anchor=(1.02, 1),
                                frameon=True,
                                title='Pathway',
                                fontsize=9)

    # 在Pathway图例下方添加colorbar
    legend_bbox = legend.get_window_extent(g.fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(g.fig.transFigure.inverted())

    cbar_ax = g.fig.add_axes([legend_bbox.x0, legend_bbox.y0 - 0.25,
                              legend_bbox.width, 0.015])

    norm = mpl.colors.Normalize(vmin=-3, vmax=3)
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Z-score (Log2 FC)', fontsize=10)

    output_path = os.path.join(output_dir, 'Integrated_heatmap_row_zscore.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已保存整合热力图（行标准化，无聚类）: {output_path}")
    plt.close()

    # 保存行Z-score数据
    csv_path = os.path.join(output_dir, 'Integrated_zscore_row_data.csv')
    df_zscore_row.to_csv(csv_path)
    print(f"已保存整合行Z-score数据: {csv_path}")

# 调用整合函数
create_integrated_heatmap(method_log2fc_data, methods, reaction_pathway_map,
                          pathway_colors, pathway_order, output_dir)



# ============================================================================
# Step 4: 代谢细胞图谱（Pseudotime - t-SNE）
# ============================================================================
print("\n" + "="*100)
print("Step 4: 代谢细胞图谱（Pseudotime Trajectory）")
print("="*100)

def create_metabolic_trajectory(method_zscore_data, methods, output_dir):
    """
    创建代谢轨迹可视化（类似单细胞的pseudotime）
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for idx, method in enumerate(methods):
        ax = axes[idx]
        df_z = method_zscore_data[method]

        print(f"\n{method}:")
        print(f"  数据形状: {df_z.shape}")

        # 填充NaN
        df_filled = df_z.fillna(0)

        # t-SNE降维
        print("  运行t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_filled)-1))
        embed = tsne.fit_transform(df_filled)

        # 使用第一主成分作为伪时间
        pseudotime = embed[:, 0]

        # 绘制散点图
        scatter = ax.scatter(embed[:, 0], embed[:, 1],
                           c=pseudotime,
                           cmap='viridis',
                           s=100,
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)

        # 添加标签（只标注一些关键的）
        for i, label in enumerate(df_filled.index):
            if i % 3 == 0:  # 每隔3个标注一个
                ax.annotate(label, (embed[i, 0], embed[i, 1]),
                          fontsize=7, alpha=0.6)

        ax.set_title(f'{method}\nMetabolic Pseudotime Trajectory',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pseudotime', fontsize=10)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'metabolic_pseudotime_trajectory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 已保存代谢轨迹图: {output_path}")
    plt.close()

create_metabolic_trajectory(method_zscore_data, methods, output_dir)

# ============================================================================
# Step 5: PCA分析（主成分分析）
# ============================================================================
print("\n" + "="*100)
print("Step 5: PCA主成分分析")
print("="*100)

from sklearn.decomposition import PCA

def create_pca_analysis(method_zscore_data, methods, output_dir):
    """PCA分析"""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for idx, method in enumerate(methods):
        ax = axes[idx]
        df_z = method_zscore_data[method]
        df_filled = df_z.fillna(0)

        print(f"\n{method}:")

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_filled)

        print(f"  PC1解释方差: {pca.explained_variance_ratio_[0]*100:.2f}%")
        print(f"  PC2解释方差: {pca.explained_variance_ratio_[1]*100:.2f}%")

        # 绘制
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                           c=range(len(df_filled)),
                           cmap='Spectral',
                           s=100,
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)

        # 添加标签
        for i, label in enumerate(df_filled.index):
            if i % 3 == 0:
                ax.annotate(label, (pca_result[i, 0], pca_result[i, 1]),
                          fontsize=7, alpha=0.6)

        ax.set_title(f'{method}\nPCA Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'pca_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 已保存PCA分析图: {output_path}")
    plt.close()

create_pca_analysis(method_zscore_data, methods, output_dir)

# ============================================================================
# Step 6: 路径水平的热力图（按路径聚合）
# ============================================================================
print("\n" + "="*100)
print("Step 6: 路径水平的热力图")
print("="*100)

def create_pathway_level_heatmap(method_log2fc_data, reaction_pathway_map,
                                 pathway_colors, pathway_order, methods, output_dir):
    """
    创建路径水平的聚合热力图
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    for idx, method in enumerate(methods):
        ax = axes[idx]
        df = method_log2fc_data[method]

        # 按路径聚合（取中位数）
        pathway_data = {}
        for strain in df.index:
            pathway_values = {}
            for pathway in set(reaction_pathway_map.values()):
                reactions = [r for r in df.columns if reaction_pathway_map.get(r, 'Other') == pathway]
                if reactions:
                    values = df.loc[strain, reactions].values
                    # 取中位数（忽略NaN）
                    pathway_values[pathway] = np.nanmedian(values)
                else:
                    pathway_values[pathway] = np.nan
            pathway_data[strain] = pathway_values

        pathway_df = pd.DataFrame(pathway_data).T

        # 按照pathway_order排序路径列
        existing_pathways = [p for p in pathway_order if p in pathway_df.columns]
        pathway_df = pathway_df[existing_pathways]

        # 绘制热力图
        sns.heatmap(pathway_df,
                   cmap=COLORMAP,
                   center=0,
                   vmin=-5, vmax=5,
                   cbar_kws={'label': 'Median Log2 FC', 'shrink': 0.8},
                   ax=ax,
                   linewidths=0.5,
                   linecolor='white',
                   xticklabels=True,
                   yticklabels=True)

        ax.set_title(f'{method}\nPathway-level Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Pathway', fontsize=12)
        ax.set_ylabel('Strains', fontsize=12)

        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'pathway_level_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 已保存路径水平热力图: {output_path}")
    plt.close()

create_pathway_level_heatmap(method_log2fc_data, reaction_pathway_map,
                            pathway_colors, pathway_order, methods, output_dir)

# ============================================================================
# Step 7: 相关性网络图（反应之间的相关性）
# ============================================================================
print("\n" + "="*100)
print("Step 7: 反应相关性网络分析")
print("="*100)

def create_correlation_network(method, log2fc_df, reaction_pathway_map,
                               pathway_colors, output_dir, top_n=50):
    """
    创建反应之间的相关性网络图
    只显示相关性最强的top_n个反应
    """
    print(f"\n{method}:")

    df = log2fc_df.copy()

    # 填充NaN值
    df_filled = df.fillna(0)

    # 选择方差最大的top_n个反应（最有变化的反应）
    reaction_var = df_filled.var(axis=0)
    # 过滤掉方差为0或NaN的反应
    reaction_var = reaction_var[reaction_var > 0]
    top_reactions = reaction_var.nlargest(min(top_n, len(reaction_var))).index.tolist()
    df_top = df_filled[top_reactions]

    print(f"  选择方差最大的 {len(top_reactions)} 个反应")

    # 计算相关性矩阵
    corr_matrix = df_top.corr()

    # 用0填充NaN值
    corr_matrix = corr_matrix.fillna(0)

    # 只保留强相关性（绝对值 > 0.6）
    corr_matrix_filtered = corr_matrix.copy()
    corr_matrix_filtered[abs(corr_matrix_filtered) < 0.6] = 0

    # 绘制热力图（不使用聚类，直接显示）
    fig, ax = plt.subplots(figsize=(16, 14))

    # 准备颜色注释
    col_colors = [pathway_colors.get(reaction_pathway_map.get(r, 'Other'), '#95a5a6')
                  for r in corr_matrix_filtered.columns]

    # 使用简单的热力图，不进行聚类
    sns.heatmap(corr_matrix_filtered,
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation'},
                ax=ax,
                xticklabels=True,
                yticklabels=True,
                linewidths=0.5,
                square=True)

    ax.set_title(f'{method} - Reaction Correlation Network (Top {len(top_reactions)})',
                 fontsize=16, fontweight='bold', pad=20)

    # 旋转标签
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{method}_correlation_network.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ 已保存: {output_path}")
    plt.close()

# 为每个方法生成相关性网络
for method in methods:
    create_correlation_network(method, method_log2fc_data[method],
                              reaction_pathway_map, pathway_colors, output_dir)

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*100)
print("分析完成！")
print("="*100)
print(f"\n所有结果保存在: {output_dir}")
print("\n生成的文件:")
print("  1. 热力图（列标准化）: *_heatmap_col_zscore_rounded.pdf")
print("  2. 热力图（行标准化）: *_heatmap_row_zscore.pdf")
print("  3. 热力图（不归一化）: *_heatmap_no_norm_rounded.pdf")
print("  4. 整合热力图（列标准化）: Integrated_heatmap_col_zscore.pdf")
print("  5. 整合热力图（行标准化）: Integrated_heatmap_row_zscore.pdf")
print("  6. 代谢轨迹图: metabolic_pseudotime_trajectory.png")
print("  7. PCA分析图: pca_analysis.png")
print("  8. 路径水平热力图: pathway_level_heatmap.png")
print("  9. 相关性网络图: *_correlation_network.png")
print(" 10. 列Z-score数据: *_zscore_col_data.csv + Integrated_zscore_col_data.csv")
print(" 11. 行Z-score数据: *_zscore_row_data.csv + Integrated_zscore_row_data.csv")
print(" 12. 不归一化数据: *_no_normalization_data.csv")
print("\n关键特点:")
print("  ✓ 包含wildtype数据")
print("  ✓ 无聚类，按固定顺序排列")
print("  ✓ Z-score标准化（列和行两种方式）")
print("  ✓ 不归一化原始数据热图")
print("  ✓ 三个方法的数据整合分析")
print("  ✓ 按pathway_annotations中定义的固定顺序排列反应")
print("  ✓ colorbar位置在Pathway图例下方")
print("  ✓ 路径注释条")
print("  ✓ 伪时间轨迹分析")
print("  ✓ PCA主成分分析")
print("  ✓ 路径水平聚合")
print("  ✓ 反应相关性网络")

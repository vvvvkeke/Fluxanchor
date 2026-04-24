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


radius = 0.5 # Rounded-corner radius
def make_heatmap_rounded_squares(ax, data_df, cmap, vmin=-3, vmax=3, radius=0.15, linewidth=0.5, edgecolor='white'):
    """
    Replace each heatmap cell with a rounded square using a custom colormap.
    """
    import matplotlib.colors as mcolors

    # Remove the original heatmap patches.
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

# Configure fonts (if needed)
# plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

# Commentary on various colormaps:
# viridis is deemed unattractive, RdBu_r is passable, BrBG slightly better,
# GnBu looks nice but suits one-sided data better, etc.
colors = ["#96CCEA", "white", "#ED949A"]
# colors = ["#73a8d5", "white", "#e283a6"]

COLORMAP = LinearSegmentedColormap.from_list("custom", colors)
# COLORMAP = 'RdBu_r'  # blue=underprediction, white=accurate, red=overprediction
FLUX_THRESHOLD = 0   # threshold for setting values to np.nan

# ============================================================================
# Species selection (hyperparameter)
# ============================================================================
# Options: "E. coli" or "B. subtilis"
# ORGANISM = "B. subtilis"  # Change this line to switch organism.
ORGANISM = "E. coli"  # Default organism selection.
# ============================================================================

# Configure file paths based on the organism.
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
    raise ValueError(f"Unsupported organism: {ORGANISM}. Please choose 'E. coli' or 'B. subtilis'.")

# Ensure the output directory exists.
os.makedirs(output_dir, exist_ok=True)

print("=" * 100)
print("Single-cell style metabolic heatmap analysis")
print("=" * 100)

# ============================================================================
# Step 1: Read data and construct log2 fold-change matrices.
# ============================================================================
print("\nLoading data...")
wildtype_df = pd.read_csv(wildtype_csv_path)
detailed_df = pd.read_csv(detailed_csv_path)
print(f"Wildtype data shape: {wildtype_df.shape}")
print(f"Detailed data shape: {detailed_df.shape}")

# Reaction names come directly from the wild-type data.
reaction_names = wildtype_df['Reaction'].tolist()
print(f"Number of reactions extracted from wildtype: {len(reaction_names)}")

def build_log2fc_matrix_with_wildtype(wildtype_df, detailed_df, reaction_names, method_name, method_wildtype_col, flux_threshold=FLUX_THRESHOLD):
    """Construct a log2 fold-change matrix including wild-type data."""
    epsilon = 1e-5

    # 1. Process wild-type data.
    print(f"  Processing wild-type data...")
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

    # 2. Process mutant data.
    method_df = detailed_df[detailed_df['Method'] == method_name].copy()
    print(f"  {method_name}: found {len(method_df)} mutant strains")

    log2fc_data = {'wildtype': wildtype_log2fc}  # include wild-type first

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

# Build log2 fold changes for each method (including wild type).
print("\nConstructing log2 fold-change matrices (including wild type)...")
methods = ['Eki2vivo', 'EkiLLm', 'fba']
methods_wildtype = ['Eki2vivo', 'EkiLLM', 'FBA']  # column names in wildtype CSV
method_log2fc_data = {}

for method, method_wildtype in zip(methods, methods_wildtype):
    log2fc_df = build_log2fc_matrix_with_wildtype(wildtype_df, detailed_df, reaction_names, method, method_wildtype)
    method_log2fc_data[method] = log2fc_df
    print(f"    {method}: {log2fc_df.shape} (includes wildtype)")

# Determine reactions common to all methods.
common_reactions = set(reaction_names)
for method in methods:
    common_reactions = common_reactions.intersection(set(method_log2fc_data[method].columns))

common_reactions = sorted(list(common_reactions))
print(f"\nNumber of reactions shared across methods: {len(common_reactions)}")

# Restrict each method to the shared set.
for method in methods:
    method_log2fc_data[method] = method_log2fc_data[method][common_reactions]
    print(f"  {method}: {method_log2fc_data[method].shape}")

# ============================================================================
# Step 2: define metabolic pathway annotations (based on reaction names)
# ============================================================================
print("\nDefining metabolic pathway annotations...")

# Define key reactions by pathway
pathway_annotations = {
    'Glycolysis': [
        "PGI", # Phosphoglucose isomerase
        "PFK", # Phosphofructokinase
        "FBP", # Fructose-1,6-bisphosphatase
        "FBA", # Fructose-bisphosphate aldolase
        "TPI", # Triose phosphate isomerase
        "GAPD", # Glyceraldehyde-3-phosphate dehydrogenase
        "PGK",
        "PGM", # Phosphoglycerate mutase
        "ENO", 
        "GLCptspp",  "PYK",
        "FBA_reverse",
        "PGK_reverse",  # reverse flux forming phosphorylated products
        "ENO_reverse",
        "GAPD_reverse",
        "TPI_reverse",  # DHAP/G3P interconversion
        "PGI_reverse",  # Glucose-6-P / Fructose-6-P interconversion
        "PGM_reverse",
    ],
    'PP': [
        "G6PDH2r", "PGL", "GND", "RPE", "RPI", "TKT1", "TKT2", "TALA",
        "G6PDH2r_reverse",  # Reverse G6P dehydrogenase
        "RPE_reverse",   # Ribulose-5-P epimerase
        "RPI_reverse",   # Ribose-5-P isomerase
        "TKT1_reverse",  # reversible transketolase
        "TKT2_reverse",
        "TALA_reverse",  # reversible transaldolase
    ],
    'ED':[
        "EDA", # 2-Keto-3-deoxy-phosphogluconate aldolase
        "EDD", # 6-phosphogluconate dehydratase
    ],
    'TCA': [
        "PDH", # Pyruvate dehydrogenase
        "CS", # Citrate synthase
        "ACONTa", # Aconitase (step a)
        "ACONTb",
        "ICDHyr", # Isocitrate dehydrogenase
        "AKGDH", # Alpha-ketoglutarate dehydrogenase
        "SUCOAS", # Succinyl-CoA synthetase
        "SUCDi", # Succinate dehydrogenase
        "FUM", # Fumarase
        "MDH", # Malate dehydrogenase
        "ACONTa_reverse",
        "ACONTb_reverse",
        "ICDHyr_reverse",
        "SUCOAS_reverse",
        "FUM_reverse",
        "MDH_reverse",
    ],
    'Gluconeogenesis': ['PPCK', 'PPS', "ME1", "ME2", "PPC"],
    'Amino acid': [
        # Aromatic amino acid pathway
        "CHORS", "DHQS", "DHQTi",
        # Branched-chain amino acid pathway
        "DHAD1", "IPMD", "IPPS",
        # Serine/glycine pathway
        "PSERT", # phosphoserine transaminase
        "PSP_L", # phosphoserine phosphatase
        "SERAT", # serine acetyltransferase
        "SERAT_reverse"
        # Lysine pathway
        "DAPDC", # diaminopimelate decarboxylase
        "DAPE",
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
    'Glycolysis': "#d3eac5", 'PP': "#db349b", 'ED': "#CD61D7", 'TCA': '#2ecc71',
    'Gluconeogenesis': '#feca57', 'Amino acid': '#f39c12', 
    'Fatty acid': '#9b59b6', 
    'Other': '#95a5a6',
    # 'Nucleotide': '#ee5a6f', 'Fermentation': '#48dbfb', 'Respiration': "#7D1B88",
    # 'Exchange': '#1abc9c', 'Energy': '#e67e22'
}

# Assign pathway labels to reactions
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

# Count the number of reactions in each pathway
pathway_counts = {}
for pathway in reaction_pathway_map.values():
    pathway_counts[pathway] = pathway_counts.get(pathway, 0) + 1

print("\nPathway annotation counts:")
for pathway, count in sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {pathway}: {count} reactions")

# Create pathway color mapping
# pathway_colors = {
#     'Glycolysis': '#e74c3c',                              # red
#     'PPP': '#3498db',                                      # blue
#     'TCA': '#2ecc71',                                      # green
#     'ED': '#ff6b6b',                                       # light red
#     'Gluconeogenesis': '#feca57',                          # yellow
#     'Amino_acid': '#f39c12',                               # orange
#     'Fatty acid': '#9b59b6',                               # purple
#     'Nucleotide Metabolism': '#ee5a6f',                    # magenta
#     'Fermentation & Anaerobic Metabolism': '#48dbfb',      # light blue
#     'Respiration & Electron Transport': "#7D1B88",         # dark blue
#     'Exchange': '#1abc9c',                                 # cyan
#     'Energy': '#e67e22',                                   # orange-yellow
#     'Other': '#95a5a6'                                     # gray
# }

# ============================================================================
# Step 3: create single-cell style heatmaps for each method
# ============================================================================

def create_single_cell_style_heatmap(method, log2fc_df, reaction_pathway_map,
                                     pathway_colors, pathway_order, output_dir, colormap=COLORMAP):
    """
    Column normalization + rounded squares + dendrogram/colorbar/legend recreation.
    """
    print(f"\n{'='*80}")
    print(f"Generating single-cell style heatmap for {method} (column normalization + rounded cells)")
    print(f"{'='*80}")

    # ---------- 1. Data preparation ----------
    df = log2fc_df.copy()
    df = df.dropna(axis=0, how='all').dropna(axis=1, how='all')
    print(f"Filtered shape: {df.shape}")

    # ---------- 2. Column Z-score ----------
    from scipy import stats
    df_zscore = df.copy()
    for col in df_zscore.columns:
        vals = df_zscore[col].values
        mask = ~np.isnan(vals)
        if mask.sum() > 1:
            df_zscore.loc[mask, col] = stats.zscore(vals[mask])
    df_filled = df_zscore.fillna(0)

    # ---------- 3. Sort reactions within each pathway ----------
    pathway_reactions = {p: [] for p in pathway_order}
    for r in df_zscore.columns:
        pathway_reactions[reaction_pathway_map.get(r, 'Other')].append(r)

    sorted_columns = []
    for pathway in pathway_order:
        reactions = pathway_reactions[pathway]
        if not reactions: continue
        intensity = {r: np.mean(np.abs(df_zscore[r].values[~np.isnan(df_zscore[r])]))
                     for r in reactions}
        sorted_columns.extend([r for r, _ in sorted(intensity.items(),
                                                   key=lambda x: x[1], reverse=True)])
    df_filled = df_filled[sorted_columns]

    # ---------- 4. Row clustering ----------
    row_linkage = linkage(df_filled.values, method='ward', metric='euclidean')

    # ---------- 5. Clustermap layout ----------
    col_colors = [pathway_colors.get(reaction_pathway_map.get(r, 'Other'), '#95a5a6')
                  for r in df_filled.columns]

    n_rows, n_cols = df_filled.shape
    cell = 0.2
    fig_w = n_cols * cell + 1.5
    fig_h = n_rows * cell + 1.0

    g = sns.clustermap(df_filled,
                       row_linkage=row_linkage,
                       col_cluster=False,
                       cmap=colormap,
                       center=0, vmin=-3, vmax=3,
                       cbar_pos=None,
                       figsize=(fig_w, fig_h),
                       col_colors=col_colors,
                       xticklabels=False,
                       yticklabels=True,
                       linewidths=0,
                       dendrogram_ratio=0.05)

    g.ax_heatmap.set_aspect('equal')

    # ---------- 6. Rounded cells ----------
    make_heatmap_rounded_squares(g.ax_heatmap, df_filled,
                                 cmap=colormap, vmin=-3, vmax=3,
                                 radius=radius, linewidth=0, edgecolor='white')

    # ---------- 7. Align dendrogram/labels/colorbar ----------
    # thicken the dendrogram lines
    for _, spine in g.ax_row_dendrogram.spines.items():
        spine.set_linewidth(2.5)
    for line in g.ax_row_dendrogram.collections:
        line.set_linewidth(2.5)

    # bold yticklabels
    for label in g.ax_heatmap.get_yticklabels():
        label.set_weight('bold')

    # shift col_colors and tree upward
    ymin, ymax = g.ax_heatmap.get_ylim()
    shift = (ymax - ymin) / n_rows * 0.5
    dendro_ymin, dendro_ymax = g.ax_row_dendrogram.get_ylim()
    g.ax_row_dendrogram.set_ylim(dendro_ymin - shift, dendro_ymax - shift)

    pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])

    heat_pos = g.ax_heatmap.get_position()
    col_pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([heat_pos.x0, col_pos.y0,
                                  heat_pos.width, col_pos.height])

    # ---------- 8. Title ----------
    g.fig.suptitle(f'{method} - Clustered Heatmap (Column Z-score, Rounded Squares)',
                   fontsize=18, fontweight='bold', y=0.98)

    # ---------- 9. Legend & colorbar ----------
    legend_elements = [Patch(facecolor=pathway_colors[p], label=p)
                       for p in pathway_order if p in reaction_pathway_map.values()]
    legend = g.ax_heatmap.legend(handles=legend_elements,
                                loc='upper left', bbox_to_anchor=(1.02, 1),
                                frameon=True, title='Pathway', fontsize=9)
    for txt in legend.get_texts():
        txt.set_weight('bold')
        txt.set_color(pathway_colors.get(txt.get_text(), '#000'))
    legend.get_title().set_weight('bold')

    # Place colorbar beneath the legend
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

    # ---------- 10. Save figure ----------
    out_path = os.path.join(output_dir,
                            f'{method}_clustered_heatmap_col_zscore_rounded.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved (column normalization + rounded corners): {out_path}")
    plt.close()

    return df_zscore, row_linkage, df, sorted_columns

def create_row_normalized_heatmap(method, log2fc_df, row_linkage, sorted_columns,
                                   reaction_pathway_map, pathway_colors, pathway_order,
                                   output_dir, colormap=COLORMAP):
    """
    Create a row-normalized clustered heatmap using the given column order.

    Steps:
    1. Use the provided column order.
    2. Apply row-wise Z-score normalization.
    3. Reuse existing row linkage.
    4. Add dendrogram, pathway annotation bars, etc.
    """
    print(f"\n{'='*80}")
    print(f"Generating row-normalized heatmap for {method}")
    print(f"{'='*80}")

    # Use the provided sorted columns.
    df = log2fc_df[sorted_columns].copy()

    print(f"Data shape: {df.shape}")

    # Step 1: Row-wise Z-score normalization.
    print("\nStep 1: Z-score normalization (row-wise for each sample)...")
    from scipy import stats

    df_zscore_row = df.copy()
    for idx in df_zscore_row.index:
        values = df_zscore_row.loc[idx].values
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() > 1:
            valid_values = values[valid_mask]
            z_scores = stats.zscore(valid_values)
            df_zscore_row.loc[idx, valid_mask] = z_scores
        else:
            df_zscore_row.loc[idx] = values

    print(f"Range after Z-score: [{df_zscore_row.min().min():.2f}, {df_zscore_row.max().max():.2f}]")

    # Step 2: Fill NaNs with zeros.
    df_filled_row = df_zscore_row.fillna(0)

    # Step 3: Create the clustered heatmap (reuse row linkage).
    print("\nStep 2: Create heatmap (reuse existing row linkage)...")

    # Build pathway annotation colors.
    col_colors = [pathway_colors.get(reaction_pathway_map.get(r, 'Other'), '#95a5a6')
                  for r in df_zscore_row.columns]
    
    n_rows = len(df_filled_row)
    n_cols = len(df_filled_row.columns)
    cell_size = 0.2  # adjust overall figure size

    figwidth = n_cols * cell_size
    figheight = n_rows * cell_size

    figheight += 1.0
    figwidth += 1.5  # extra space for colorbar + legend

    g = sns.clustermap(df_filled_row,
                       row_linkage=row_linkage,
                       col_cluster=False,
                       cmap=colormap,
                       center=0,
                       vmin=-3, vmax=3,
                       cbar_pos=None,
                       figsize=(figwidth, figheight),
                       col_colors=col_colors,
                       xticklabels=False,
                       yticklabels=True,
                       linewidths=0,
                       dendrogram_ratio=0.05)

    
    # Force each heatmap cell to be square
    g.ax_heatmap.set_aspect('equal')

    # Apply rounded squares with the custom colormap
    make_heatmap_rounded_squares(
        g.ax_heatmap,
        df_filled_row,
        cmap=COLORMAP,  # pass the custom colormap
        vmin=-3,
        vmax=3,
        radius=radius,
        linewidth=0,
        edgecolor='white'
    )
    
    # Change 1: thicken the left dendrogram lines
    for _, spine in g.ax_row_dendrogram.spines.items():
        spine.set_linewidth(2.5)  # bold dendrogram lines
    for line in g.ax_row_dendrogram.collections:
        line.set_linewidth(2.5)   # bold dendrogram branches

    # Change 2: bold mutation labels on the right
    for label in g.ax_heatmap.get_yticklabels():
        label.set_weight('bold')
    # Fix 2: shift col_colors upward to avoid overlap with heatmap
    for patch in g.ax_col_colors.collections:
        # each patch is a QuadMesh; move the y offsets
        offsets = patch.get_offsets()
        offsets[:, 1] += 1  # shift upward by roughly 0.2 units (tweak if needed)
        patch.set_offsets(offsets)

    # Adjustment 1: shift dendrogram upward
    dendro_ax = g.ax_row_dendrogram
    heat_ax = g.ax_heatmap

    # Get the heatmap y-range
    ymin, ymax = heat_ax.get_ylim()

    # Compute dendrogram shift amount
    dendro_ymin, dendro_ymax = dendro_ax.get_ylim()
    shift = (ymax - ymin) / n_rows * 0.5  # move up half a cell
    dendro_ax.set_ylim(dendro_ymin - shift, dendro_ymax - shift)

    # Adjustment 3: match col_colors width to the heatmap
    heat_ax = g.ax_heatmap
    col_color_ax = g.ax_col_colors
    heat_pos = heat_ax.get_position()
    col_color_pos = col_color_ax.get_position()

    # Match widths
    col_color_ax.set_position([
        heat_pos.x0,
        col_color_pos.y0,
        heat_pos.width,
        col_color_pos.height
    ])


     # Refresh canvas after drawing rounded squares
    g.fig.canvas.draw()

    # Geometric tweaks: align dendrogram and col_colors
    dendro_ax = g.ax_row_dendrogram
    heat_ax = g.ax_heatmap
    col_color_ax = g.ax_col_colors

    # 1) Raise the dendrogram so the tips align with the cell centers
    ymin, ymax = heat_ax.get_ylim()
    dendro_ymin, dendro_ymax = dendro_ax.get_ylim()
    shift = (ymax - ymin) / len(df_filled_row) * 0.5
    dendro_ax.set_ylim(dendro_ymin - shift, dendro_ymax - shift)

    # 2) Raise the col_colors bar to avoid overlapping the heatmap
    pos = col_color_ax.get_position()
    col_color_ax.set_position([
        pos.x0,
        pos.y0 + 0.01,  # nudge upward slightly (change 0.005-0.02 as needed)
        pos.width,
        pos.height
    ])

    # 3) Match the width of col_colors and the heatmap
    heat_pos = heat_ax.get_position()
    col_color_pos = col_color_ax.get_position()
    col_color_ax.set_position([
        heat_pos.x0,
        col_color_pos.y0,
        heat_pos.width,
        col_color_pos.height
    ])

    # ax = g.ax_heatmap
    # yticks = ax.get_yticks()
    # ytick_labels = [df_filled_row.index[i] for i in yticks if i < len(df_filled_row.index)]
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(ytick_labels, fontsize=10, fontweight='bold')  # bold labels

    # Set title
    g.fig.suptitle(f'{method} - Clustered Heatmap (Row Z-score normalized)',
                   fontsize=18, fontweight='bold', y=0.98)

    # Add legend (pathway colors) following pathway_order
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
    
    # Bold legend labels
    for text in legend.get_texts():
        text.set_weight('bold')

    # Color legend labels with their pathway color
    for text in legend.get_texts():
        pathway = text.get_text()
        text.set_color(pathway_colors.get(pathway, '#000000'))

    # Optionally bold the legend title
    legend.get_title().set_weight('bold')
    # Add a colorbar below the pathway legend
    # Get legend bounding box
    legend_bbox = legend.get_window_extent(g.fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(g.fig.transFigure.inverted())

    # Reserve colorbar space under the legend
    cbar_ax = g.fig.add_axes([legend_bbox.x0, legend_bbox.y0 - 0.35,
                              legend_bbox.width, 0.02])

    # Build the colorbar
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=-3, vmax=3)
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Z-score (Log2 FC)', fontsize=10, weight="bold")
    # Bold the colorbar tick labels
    for l in cbar.ax.get_xticklabels():
        l.set_weight('bold')
    # Save
    output_path = os.path.join(output_dir, f'{method}_clustered_heatmap_row_zscore.png')
    output_path = os.path.join(output_dir, f'{method}_clustered_heatmap_row_zscore.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved (row normalization): {output_path}")
    plt.close()

    return df_zscore_row

def create_no_normalization_heatmap(method, log2fc_df, row_linkage, sorted_columns,
                                    reaction_pathway_map, pathway_colors, pathway_order,
                                    output_dir, colormap=COLORMAP):
    """
    No normalization + rounded squares (everything else unchanged)
    """
    print(f"\n{'='*80}")
    print(f"Generating unnormalized heatmap for {method} (rounded cells)")
    print(f"{'='*80}")

    df = log2fc_df[sorted_columns].copy()
    df_filled = df.fillna(0)                     # plotting only

    col_colors = [pathway_colors.get(reaction_pathway_map.get(r, 'Other'), '#95a5a6')
                  for r in df_filled.columns]

    n_rows, n_cols = df_filled.shape
    cell = 0.2
    fig_w = n_cols * cell + 1.5
    fig_h = n_rows * cell + 1.0

    g = sns.clustermap(df_filled,
                       row_linkage=row_linkage,
                       col_cluster=False,
                       cmap=colormap,
                       center=0, vmin=-5, vmax=5,
                       cbar_pos=None,
                       figsize=(fig_w, fig_h),
                       col_colors=col_colors,
                       xticklabels=False,
                       yticklabels=True,
                       linewidths=0,
                       dendrogram_ratio=0.05)

    g.ax_heatmap.set_aspect('equal')

    # ---------- Rounded corners ----------
    make_heatmap_rounded_squares(g.ax_heatmap, df_filled,
                                 cmap=colormap, vmin=-5, vmax=5,
                                 radius=radius, linewidth=0, edgecolor='white')

    # ---------- Alignment adjustments ----------
    for _, spine in g.ax_row_dendrogram.spines.items():
        spine.set_linewidth(2.5)
    for line in g.ax_row_dendrogram.collections:
        line.set_linewidth(2.5)
    for label in g.ax_heatmap.get_yticklabels():
        label.set_weight('bold')

    ymin, ymax = g.ax_heatmap.get_ylim()
    shift = (ymax - ymin) / n_rows * 0.5
    dymin, dymax = g.ax_row_dendrogram.get_ylim()
    g.ax_row_dendrogram.set_ylim(dymin - shift, dymax - shift)

    pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([pos.x0, pos.y0 + 0.01, pos.width, pos.height])

    heat_pos = g.ax_heatmap.get_position()
    col_pos = g.ax_col_colors.get_position()
    g.ax_col_colors.set_position([heat_pos.x0, col_pos.y0,
                                  heat_pos.width, col_pos.height])

    # ---------- Title ----------
    g.fig.suptitle(f'{method} - Clustered Heatmap (No Normalization, Rounded Squares)',
                   fontsize=18, fontweight='bold', y=0.98)

    # ---------- Legend & colorbar ----------
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

    # ---------- Save ----------
    out_path = os.path.join(output_dir,
                            f'{method}_clustered_heatmap_no_norm_rounded.pdf')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved (no normalization + rounded corners): {out_path}")
    plt.close()

    return df

# Generate single-cell style heatmaps for each method (column- and row-normalized)
method_zscore_data = {}
method_row_zscore_data = {}
for method in methods:
    # Create column-normalized heatmap
    df_z, row_linkage, df_orig, sorted_cols = create_single_cell_style_heatmap(
        method, method_log2fc_data[method], reaction_pathway_map,
        pathway_colors, pathway_order, output_dir
    )
    method_zscore_data[method] = df_z

    # Save column Z-score data
    csv_path = os.path.join(output_dir, f'{method}_zscore_col_data.csv')
    df_z.to_csv(csv_path)
    print(f"Saved column Z-score data: {csv_path}")

    # Create row-normalized heatmap (reuse the same clustering/order)
    df_z_row = create_row_normalized_heatmap(
        method, df_orig, row_linkage, sorted_cols,
        reaction_pathway_map, pathway_colors, pathway_order, output_dir
    )
    method_row_zscore_data[method] = df_z_row

    # Save row Z-score data
    csv_path_row = os.path.join(output_dir, f'{method}_zscore_row_data.csv')
    df_z_row.to_csv(csv_path_row)
    print(f"Saved row Z-score data: {csv_path_row}")

    # Create unnormalized heatmap (reuse the same clustering/order)
    df_no_norm = create_no_normalization_heatmap(
        method, df_orig, row_linkage, sorted_cols,
        reaction_pathway_map, pathway_colors, pathway_order, output_dir
    )

    # Save unnormalized data
    csv_path_no_norm = os.path.join(output_dir, f'{method}_no_normalization_data.csv')
    df_no_norm.to_csv(csv_path_no_norm)
    print(f"Saved unnormalized data: {csv_path_no_norm}")

# ============================================================================
# Step 3.5: integrate all three methods and build combined heatmaps
# ============================================================================
print("\n" + "="*100)
print("Step 3.5: Integrating all three methods")
print("="*100)

def create_integrated_heatmap(method_log2fc_data, methods, reaction_pathway_map,
                               pathway_colors, pathway_order, output_dir, colormap=COLORMAP):
    """
    Integrate the three methods and produce both row- and column-standardized heatmaps.
    """
    print("\nIntegrating the three methods...")

    # Combine the three method matrices
    integrated_data = []
    for method in methods:
        df = method_log2fc_data[method].copy()
        # Annotate each strain with its method
        df.index = [f"{method}_{idx}" for idx in df.index]
        integrated_data.append(df)

    # Stack vertically
    df_integrated = pd.concat(integrated_data, axis=0)
    print(f"Integrated shape: {df_integrated.shape}")
    print(f"  Total samples: {len(df_integrated)} "
          f"({len(methods)} methods x {len(df_integrated)//len(methods)} samples/method)")
    print(f"  Total reactions: {len(df_integrated.columns)}")

    # Drop rows/columns that are entirely NaN
    df_integrated = df_integrated.dropna(axis=0, how='all')
    df_integrated = df_integrated.dropna(axis=1, how='all')
    print(f"Filtered integrated shape: {df_integrated.shape}")

    # ========== Column normalization ==========
    print("\n" + "="*80)
    print("Generating integrated column-normalized heatmap")
    print("="*80)

    # Z-score normalization (per column / reaction)
    print("\nStep 1: Column-wise Z-score normalization for each reaction...")
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

    print(f"Column Z-score range: [{df_zscore_col.min().min():.2f}, {df_zscore_col.max().max():.2f}]")

    # Sort reactions inside each pathway by mean |Z-score|
    print("\nStep 2: Order reactions within each pathway by Z-score intensity...")
    pathway_reactions = {}
    for pathway in pathway_order:
        pathway_reactions[pathway] = []

    for reaction in df_zscore_col.columns:
        pathway = reaction_pathway_map.get(reaction, 'Other')
        pathway_reactions[pathway].append(reaction)

    sorted_columns = []
    for pathway in pathway_order:
        reactions = pathway_reactions[pathway]
        if len(reactions) > 0:
            reaction_intensity = {}
            for reaction in reactions:
                values = df_zscore_col[reaction].values
                abs_values = np.abs(values[~np.isnan(values)])
                if len(abs_values) > 0:
                    reaction_intensity[reaction] = np.mean(abs_values)
                else:
                    reaction_intensity[reaction] = 0

            sorted_reactions = sorted(reaction_intensity.items(), key=lambda x: x[1], reverse=True)
            sorted_columns.extend([r for r, _ in sorted_reactions])

    df_zscore_col = df_zscore_col[sorted_columns]
    print(f"Column ordering complete across {len(sorted_columns)} reactions")

    # Fill NaNs with zeros
    df_filled_col = df_zscore_col.fillna(0)

    # Row clustering
    print("\nStep 3: Hierarchical clustering across integrated samples...")
    row_linkage = linkage(df_filled_col.values, method='ward', metric='euclidean')

    # Build the column-normalized heatmap
    print("\nStep 4: Create integrated heatmap (column normalized)...")
    col_colors = [pathway_colors.get(reaction_pathway_map.get(r, 'Other'), '#95a5a6')
                  for r in df_zscore_col.columns]

    g = sns.clustermap(df_filled_col,
                       row_linkage=row_linkage,
                       col_cluster=False,
                       cmap=colormap,
                       center=0,
                       vmin=-3, vmax=3,
                       cbar_pos=None,
                       figsize=(32, 16),
                       col_colors=col_colors,
                       xticklabels=False,
                       yticklabels=True,
                       linewidths=0,
                       dendrogram_ratio=0.08)

    g.fig.suptitle('Integrated Heatmap (Column Z-score normalized)\n(All methods combined, clustered samples, reactions sorted by Z-score intensity within each pathway)',
                   fontsize=20, fontweight='bold', y=0.99)

    # Add legend
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

    # Place colorbar beneath the legend
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

    output_path = os.path.join(output_dir, 'Integrated_clustered_heatmap_col_zscore.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved integrated heatmap (column normalized): {output_path}")
    plt.close()

    # Save column Z-score data
    csv_path = os.path.join(output_dir, 'Integrated_zscore_col_data.csv')
    df_zscore_col.to_csv(csv_path)
    print(f"Saved integrated column Z-score data: {csv_path}")

    # ========== Row normalization ==========
    print("\n" + "="*80)
    print("Generating integrated row-normalized heatmap")
    print("="*80)

    # Reuse the same column order
    df_integrated_sorted = df_integrated[sorted_columns].copy()

    # Z-score normalization (per row / strain)
    print("\nStep 1: Row-wise Z-score normalization for each sample...")
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

    print(f"Row Z-score range: [{df_zscore_row.min().min():.2f}, {df_zscore_row.max().max():.2f}]")

    # Fill NaNs with zeros
    df_filled_row = df_zscore_row.fillna(0)

    # Build the row-normalized heatmap using the same clustering
    print("\nStep 2: Create integrated heatmap (row normalized, shared clustering)...")

    g = sns.clustermap(df_filled_row,
                       row_linkage=row_linkage,
                       col_cluster=False,
                       cmap=colormap,
                       center=0,
                       vmin=-3, vmax=3,
                       cbar_pos=None,
                       figsize=(32, 16),
                       col_colors=col_colors,
                       xticklabels=False,
                       yticklabels=True,
                       linewidths=0,
                       dendrogram_ratio=0.08)

    g.fig.suptitle('Integrated Heatmap (Row Z-score normalized)\n(All methods combined, clustered samples, reactions sorted by color intensity within pathways, row normalized)',
                   fontsize=20, fontweight='bold', y=0.99)

    # Add legend
    legend = g.ax_heatmap.legend(handles=legend_elements,
                                loc='upper left',
                                bbox_to_anchor=(1.02, 1),
                                frameon=True,
                                title='Pathway',
                                fontsize=9)

    # Place colorbar beneath the legend
    legend_bbox = legend.get_window_extent(g.fig.canvas.get_renderer())
    legend_bbox = legend_bbox.transformed(g.fig.transFigure.inverted())

    cbar_ax = g.fig.add_axes([legend_bbox.x0, legend_bbox.y0 - 0.25,
                              legend_bbox.width, 0.015])

    norm = mpl.colors.Normalize(vmin=-3, vmax=3)
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = g.fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Z-score (Log2 FC)', fontsize=10)

    output_path = os.path.join(output_dir, 'Integrated_clustered_heatmap_row_zscore.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved integrated heatmap (row normalized): {output_path}")
    plt.close()

    # Save row Z-score data
    csv_path = os.path.join(output_dir, 'Integrated_zscore_row_data.csv')
    df_zscore_row.to_csv(csv_path)
    print(f"Saved integrated row Z-score data: {csv_path}")

# Run the integration routine
create_integrated_heatmap(method_log2fc_data, methods, reaction_pathway_map,
                          pathway_colors, pathway_order, output_dir)



# ============================================================================
# Step 4: metabolic cell map (pseudotime trajectory)
# ============================================================================
print("\n" + "="*100)
print("Step 4: Metabolic cell map (pseudotime trajectory)")
print("="*100)

def create_metabolic_trajectory(method_zscore_data, methods, output_dir):
    """
    Create a metabolic trajectory visualization similar to single-cell pseudotime.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for idx, method in enumerate(methods):
        ax = axes[idx]
        df_z = method_zscore_data[method]

        print(f"\n{method}:")
        print(f"  Data shape: {df_z.shape}")

        # Fill NaNs
        df_filled = df_z.fillna(0)

        # t-SNE embedding
        print("  Running t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df_filled)-1))
        embed = tsne.fit_transform(df_filled)

        # Use first component as pseudotime
        pseudotime = embed[:, 0]

        # Scatter plot
        scatter = ax.scatter(embed[:, 0], embed[:, 1],
                           c=pseudotime,
                           cmap='viridis',
                           s=100,
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)

        # Add labels (sparse selection)
        for i, label in enumerate(df_filled.index):
            if i % 3 == 0:  # annotate every third point
                ax.annotate(label, (embed[i, 0], embed[i, 1]),
                          fontsize=7, alpha=0.6)

        ax.set_title(f'{method}\nMetabolic Pseudotime Trajectory',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pseudotime', fontsize=10)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'metabolic_pseudotime_trajectory.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved metabolic trajectory plot: {output_path}")
    plt.close()

create_metabolic_trajectory(method_zscore_data, methods, output_dir)

# ============================================================================
# Step 5: PCA (principal component analysis)
# ============================================================================
print("\n" + "="*100)
print("Step 5: PCA analysis")
print("="*100)

from sklearn.decomposition import PCA

def create_pca_analysis(method_zscore_data, methods, output_dir):
    """Run PCA and visualize the scores."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    for idx, method in enumerate(methods):
        ax = axes[idx]
        df_z = method_zscore_data[method]
        df_filled = df_z.fillna(0)

        print(f"\n{method}:")

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_filled)

        print(f"  PC1 variance explained: {pca.explained_variance_ratio_[0]*100:.2f}%")
        print(f"  PC2 variance explained: {pca.explained_variance_ratio_[1]*100:.2f}%")

        # Plot
        scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1],
                           c=range(len(df_filled)),
                           cmap='Spectral',
                           s=100,
                           alpha=0.7,
                           edgecolors='black',
                           linewidth=0.5)

        # Add labels
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
    print(f"\n[OK] Saved PCA plot: {output_path}")
    plt.close()

create_pca_analysis(method_zscore_data, methods, output_dir)

# ============================================================================
# Step 6: pathway-level heatmaps (aggregated by pathway)
# ============================================================================
print("\n" + "="*100)
print("Step 6: Pathway-level heatmaps")
print("="*100)

def create_pathway_level_heatmap(method_log2fc_data, reaction_pathway_map,
                                 pathway_colors, pathway_order, methods, output_dir):
    """
    Create pathway-level aggregated heatmaps.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    for idx, method in enumerate(methods):
        ax = axes[idx]
        df = method_log2fc_data[method]

        # Aggregate reactions per pathway using the median
        pathway_data = {}
        for strain in df.index:
            pathway_values = {}
            for pathway in set(reaction_pathway_map.values()):
                reactions = [r for r in df.columns if reaction_pathway_map.get(r, 'Other') == pathway]
                if reactions:
                    values = df.loc[strain, reactions].values
                    # take the median ignoring NaNs
                    pathway_values[pathway] = np.nanmedian(values)
                else:
                    pathway_values[pathway] = np.nan
            pathway_data[strain] = pathway_values

        pathway_df = pd.DataFrame(pathway_data).T

        # Order columns according to pathway_order
        existing_pathways = [p for p in pathway_order if p in pathway_df.columns]
        pathway_df = pathway_df[existing_pathways]

        # Plot heatmap
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

        # Axis label formatting
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=8)

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'pathway_level_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved pathway-level heatmap: {output_path}")
    plt.close()

create_pathway_level_heatmap(method_log2fc_data, reaction_pathway_map,
                            pathway_colors, pathway_order, methods, output_dir)

# ============================================================================
# Step 7: reaction-level correlation networks
# ============================================================================
print("\n" + "="*100)
print("Step 7: Reaction correlation network analysis")
print("="*100)

def create_correlation_network(method, log2fc_df, reaction_pathway_map,
                               pathway_colors, output_dir, top_n=50):
    """
    Build a correlation network across reactions
    showing only the top_n most variable reactions.
    """
    print(f"\n{method}:")

    df = log2fc_df.copy()

    # Fill NaNs
    df_filled = df.fillna(0)

    # Select the top_n reactions with the largest variance
    reaction_var = df_filled.var(axis=0)
    # Remove zero or undefined variance features
    reaction_var = reaction_var[reaction_var > 0]
    top_reactions = reaction_var.nlargest(min(top_n, len(reaction_var))).index.tolist()
    df_top = df_filled[top_reactions]

    print(f"  Selected the {len(top_reactions)} reactions with largest variance")

    # Compute the correlation matrix
    corr_matrix = df_top.corr()

    # Replace NaNs with zeros
    corr_matrix = corr_matrix.fillna(0)

    # Keep only strong correlations (|corr| > 0.6)
    corr_matrix_filtered = corr_matrix.copy()
    corr_matrix_filtered[abs(corr_matrix_filtered) < 0.6] = 0

    # Plot a simple heatmap without clustering
    fig, ax = plt.subplots(figsize=(16, 14))

    # Prepare color annotations
    col_colors = [pathway_colors.get(reaction_pathway_map.get(r, 'Other'), '#95a5a6')
                  for r in corr_matrix_filtered.columns]

    # Heatmap without clustering
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

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'{method}_correlation_network.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [OK] Saved: {output_path}")
    plt.close()

# Generate correlation networks for each method
for method in methods:
    create_correlation_network(method, method_log2fc_data[method],
                              reaction_pathway_map, pathway_colors, output_dir)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*100)
print("Analysis complete!")
print("="*100)
print(f"\nAll outputs saved to: {output_dir}")
print("\nGenerated files:")
print("  1. Clustered heatmap (column normalized): *_clustered_heatmap_col_zscore.png")
print("  2. Clustered heatmap (row normalized): *_clustered_heatmap_row_zscore.png")
print("  3. Clustered heatmap (no normalization): *_clustered_heatmap.png")
print("  4. Integrated heatmap (column normalized): Integrated_clustered_heatmap_col_zscore.png")
print("  5. Integrated heatmap (row normalized): Integrated_clustered_heatmap_row_zscore.png")
print("  6. Metabolic trajectory plot: metabolic_pseudotime_trajectory.png")
print("  7. PCA plot: pca_analysis.png")
print("  8. Pathway-level heatmap: pathway_level_heatmap.png")
print("  9. Correlation network: *_correlation_network.png")
print(" 10. Column Z-score data: *_zscore_col_data.csv + Integrated_zscore_col_data.csv")
print(" 11. Row Z-score data: *_zscore_row_data.csv + Integrated_zscore_row_data.csv")
print(" 12. Unnormalized data: *_no_normalization_data.csv")
print("\nKey highlights:")
print("  - Includes wildtype data")
print("  - Hierarchical clustering + dendrograms")
print("  - Z-score normalization (columns and rows)")
print("  - Unnormalized raw-data heatmaps")
print("  - Integrated analysis across three methods")
print("  - Reactions within each pathway sorted by color intensity")
print("  - Colorbar positioned beneath the pathway legend")
print("  - Pathway annotation bar")
print("  - Pseudotime trajectory analysis")
print("  - PCA for dimensionality reduction")
print("  - Pathway-level aggregation")
print("  - Reaction correlation networks")

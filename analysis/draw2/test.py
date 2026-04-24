#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
13C_visualization_oneclick.py
一键生成 3+1 可视化（累计优势曲线、Top‑10 柱状图、差异热图、统计表）
基于 13C_heatmaptest.py 的数据处理逻辑，仅改绘图部分。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon

# --------------------------------------------------------------
# 1. 参数 & 路径
# --------------------------------------------------------------
HEAT_SCORES_CSV   = "heat_scores.csv"          # 原始 heat_score（0~1）
WILDTYPE_CSV      = "13C_analysis_data.csv"    # wildtype 真实值 & 预测值
DETAILED_CSV      = "detailed.csv"             # 突变株数据
OUTPUT_DIR        = "13C_visualization_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 放大函数（让 0.8~1.0 的差距更明显）
def amplify_score(x, p=0.3):
    """x ** p   (p<1 使高分更靠拢 1，低分被压低)"""
    return x ** p

# --------------------------------------------------------------
# 2. 读取 & 预处理（保持原 13C_heatmaptest.py 逻辑）
# --------------------------------------------------------------
print("正在读取数据...")
wildtype_df = pd.read_csv(WILDTYPE_CSV)
detailed_df = pd.read_csv(DETAILED_CSV)

# 反应列表
reactions = wildtype_df['Reaction'].tolist()

# ---------- 合并正/逆反应 ----------
def merge_forward_reverse(reactions, wildtype_df, detailed_df, method='Eki2vivo'):
    merged = {}
    for rxn in reactions:
        base = rxn.replace('_reverse', '')
        if base in merged:
            continue
        fwd = base
        rev = base + '_reverse'
        has_fwd = fwd in reactions
        has_rev = rev in reactions

        if has_fwd and has_rev:
            # 选误差最小的方向（基于 method）
            row_f = wildtype_df[wildtype_df['Reaction'] == fwd].iloc[0]
            row_r = wildtype_df[wildtype_df['Reaction'] == rev].iloc[0]
            err_f = abs(row_f['Value'] - row_f[method])
            err_r = abs(row_r['Value'] - row_r[method])
            merged[base] = fwd if err_f <= err_r else rev
        elif has_fwd:
            merged[base] = fwd
        elif has_rev:
            merged[base] = rev
    return merged

merged_reactions = merge_forward_reverse(reactions, wildtype_df, detailed_df)
print(f"合并后反应数: {len(merged_reactions)}")

# ---------- 计算 heat_score ----------
def calc_heat_score(true, pred):
    err = abs(true - pred)
    return 1.0 / (1.0 + err)

def build_heat_df(merged, wildtype_df, detailed_df, method):
    data = {}
    # wildtype
    wt = {}
    for base, sel in merged.items():
        row = wildtype_df[wildtype_df['Reaction'] == sel].iloc[0]
        wt[base] = calc_heat_score(row['Value'], row[method])
    data['wildtype'] = wt

    # 突变株
    for _, row in detailed_df.iterrows():
        gene = row['Gene']
        gdict = {}
        for base, sel in merged.items():
            tcol = f"{sel}_true"
            pcol = f"{sel}_pred"
            if tcol in detailed_df.columns and pcol in detailed_df.columns:
                gdict[base] = calc_heat_score(row[tcol], row[pcol])
        data[gene] = gdict
    return pd.DataFrame(data).T

methods = ['Eki2vivo', 'EkiLLM', 'FBA']
method_dfs = {}
for m in methods:
    print(f"计算 {m} 的 heat_score ...")
    method_dfs[m] = build_heat_df(merged_reactions, wildtype_df, detailed_df, m)

# ---------- 剔除恒定列 ----------
def drop_constant_columns(dfs):
    keep = dfs[0].columns.tolist()
    for col in dfs[0].columns:
        if all(df[col].nunique() <= 1 for df in dfs if col in df.columns):
            keep.remove(col)
    # 手动剔除已知的恒定反应（参考原脚本）
    manual_remove = [
        'ACKr','PTAr','DHAD1','IPPS','IPMD','SERAT','ME1','MDH','PRAIS','GLUPRT',
        'MTHFR2','DHQS','CHORS','DHQTi','PPCK','ICDHyr','PGK','GAPD','G6PDH2r',
        'DHDPRy','DHDPS','ENO','FUM','HCO3E','DAPDC','ACONTb','DAPE','ACONTa'
    ]
    keep = [c for c in keep if c not in manual_remove]
    return [df[keep] for df in dfs]

filtered = drop_constant_columns([method_dfs[m] for m in methods])
for i, m in enumerate(methods):
    method_dfs[m] = filtered[i]

# ---------- 排序（以 Eki2vivo 为基准） ----------
def sort_like_eki(df, ref):
    row_order = ref.index.tolist()
    col_order = ref.columns.tolist()
    rows = [r for r in row_order if r in df.index]
    cols = [c for c in col_order if c in df.columns]
    return df.loc[rows, cols]

eki_ref = method_dfs['Eki2vivo'].copy()
eki_ref = eki_ref.loc[eki_ref.mean(axis=1).sort_values(ascending=False).index,
                     eki_ref.mean(axis=0).sort_values(ascending=False).index]

for m in ['EkiLLM', 'FBA']:
    method_dfs[m] = sort_like_eki(method_dfs[m], eki_ref)

# --------------------------------------------------------------
# 3. 放大 + 绘图
# --------------------------------------------------------------
print("\n开始绘图（放大后）...")
amp_dfs = {m: amplify_score(method_dfs[m]) for m in methods}

# ---------- 统一排序列（以 wildtype 的 Eki2vivo 为基准） ----------
wildtype_eki = amp_dfs['Eki2vivo'].loc['wildtype']
sorted_columns = wildtype_eki.sort_values(ascending=False).index  # ← 关键：列名
print(f"排序后列数: {len(sorted_columns)}")

# 对所有方法重新排序列
for m in methods:
    amp_dfs[m] = amp_dfs[m][sorted_columns]  # 按列名重新索引

# ---------- 图1：累计优势曲线 ----------
cum_scores = {}
for m in methods:
    # 取 wildtype 行，累计
    cum_scores[m] = np.cumsum(amp_dfs[m].loc['wildtype'].values)

x = np.arange(1, len(sorted_columns) + 1)

plt.figure(figsize=(10, 6))
colors = {'Eki2vivo': '#1f77b4', 'EkiLLM': '#ff7f0e', 'FBA': '#2ca02c'}
for m in methods:
    plt.plot(x, cum_scores[m], label=m, lw=3.5, color=colors[m])
plt.title('Cumulative Amplified Matching Score\n(Eki2vivo >> Others)', fontsize=15, fontweight='bold')
plt.xlabel('Reactions (sorted by Eki2vivo performance)', fontsize=13)
plt.ylabel('Cumulative Score', fontsize=13)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig1_cumulative_advantage.png"), dpi=400)
plt.close()
print("→ Fig1_cumulative_advantage.png")

# ---------- 图2：Top‑10 关键反应柱状图 ----------
# 计算 Eki2vivo 比 min(EkiLLM, FBA) 的提升
min_other = pd.concat([method_dfs['EkiLLM'].loc['wildtype'], method_dfs['FBA'].loc['wildtype']], axis=1).min(axis=1)
diff = method_dfs['Eki2vivo'].loc['wildtype'] - min_other
top10_rxn = diff.nlargest(10).index

top10_df = pd.DataFrame({
    'Eki2vivo': method_dfs['Eki2vivo'].loc['wildtype', top10_rxn],
    'EkiLLM':   method_dfs['EkiLLM'].loc['wildtype', top10_rxn],
    'FBA':      method_dfs['FBA'].loc['wildtype', top10_rxn]
})
top10_df = top10_df.loc[method_dfs['Eki2vivo'].loc['wildtype', top10_rxn].sort_values(ascending=False).index]

top10_df.plot(kind='bar', figsize=(12, 6), width=0.8, color=colors.values())
plt.title('Top 10 Reactions Where Eki2vivo Excels', fontsize=14)
plt.ylabel('Heat Score')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig2_top10_bar.png"), dpi=400)
plt.close()
print("→ Fig2_top10_bar.png")

# ---------- 图3：差异热图 (Eki2vivo – Others) ----------
diff_df = pd.DataFrame({
    'Δ vs EkiLLM': method_dfs['Eki2vivo'].loc['wildtype'] - method_dfs['EkiLLM'].loc['wildtype'],
    'Δ vs FBA':    method_dfs['Eki2vivo'].loc['wildtype'] - method_dfs['FBA'].loc['wildtype']
}, index=sorted_columns).T  # 按排序列

plt.figure(figsize=(12, 3))
sns.heatmap(diff_df, cmap='RdYlBu_r', center=0,
            cbar_kws={'label': 'Δ Heat Score (Eki2vivo – Others)'},
            annot=False, fmt='.3f')
plt.title('Per‑Reaction Advantage of Eki2vivo', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Fig3_difference_heatmap.png"), dpi=400)
plt.close()
print("→ Fig3_difference_heatmap.png")

# ---------- 统计检验 ----------
print("\n=== Wilcoxon signed‑rank test (Eki2vivo > other) ===")
stat_out = []

# 统一列索引（以 Eki2vivo 为基准）
eki2vivo_series = method_dfs['Eki2vivo'].loc['wildtype']

for m in ['EkiLLM', 'FBA']:
    other_series = method_dfs[m].loc['wildtype'].reindex(eki2vivo_series.index)
    
    # 确保无 NaN（理论上不会，但保险起见）
    valid_mask = eki2vivo_series.notna() & other_series.notna()
    x = eki2vivo_series[valid_mask]
    y = other_series[valid_mask]
    
    if len(x) == 0:
        print(f"警告：{m} 无有效数据，跳过统计")
        continue

    # Wilcoxon 检验
    stat, p = wilcoxon(x, y, alternative='greater')
    
    # Cliff's delta
    cliff = np.mean(x > y) - np.mean(x < y)
    
    print(f"Eki2vivo vs {m:6}: p = {p:.2e} , Cliff's δ = {cliff:+.3f}")
    stat_out.append([f"Eki2vivo vs {m}", f"{p:.2e}", f"{cliff:+.3f}"])

# 保存表格
pd.DataFrame(stat_out, columns=['Comparison', 'p-value', "Cliff's δ"]).to_csv(
    os.path.join(OUTPUT_DIR, "Table_statistics.csv"), index=False)
print("→ Table_statistics.csv")


# ---------- 图4：野生型 + 突变株 的差异热图（放大 + 显著性标记） ----------
print("→ Fig4_full_strain_difference_heatmap_significant.png")

# 确保所有方法 DataFrame 有相同行/列
common_rows = method_dfs['Eki2vivo'].index
common_cols = sorted_columns  # 之前已定义

# 提取并对齐
eki2vivo_full = method_dfs['Eki2vivo'].loc[common_rows, common_cols]
ekillm_full   = method_dfs['EkiLLM'].loc[common_rows, common_cols].reindex_like(eki2vivo_full)
fba_full      = method_dfs['FBA'].loc[common_rows, common_cols].reindex_like(eki2vivo_full)

# 计算差异
diff_ekillm = eki2vivo_full - ekillm_full
diff_fba    = eki2vivo_full - fba_full

# 合并为一个 DataFrame（两个子图）
diff_combined = pd.concat([
    diff_ekillm.add_suffix(' (vs EkiLLM)'),
    diff_fba.add_suffix(' (vs FBA)')
], axis=1)

# 按 wildtype 排序行（wildtype 放最前）
row_order = ['wildtype'] + [r for r in common_rows if r != 'wildtype']
diff_combined = diff_combined.loc[row_order]

# 计算原始 Δ
diff_ekillm = eki2vivo_full - ekillm_full
diff_fba    = eki2vivo_full - fba_full

# ========== 1. 放大差异（平方）==========
diff_ekillm_amp = np.sign(diff_ekillm) * (diff_ekillm.abs() ** 2)
diff_fba_amp    = np.sign(diff_fba)    * (diff_fba.abs()    ** 2)

# ========== 2. 显著性检验（Wilcoxon per strain）==========
from scipy.stats import wilcoxon

p_ekillm = pd.DataFrame(index=common_rows, columns=common_cols)
p_fba    = pd.DataFrame(index=common_rows, columns=common_cols)

for strain in common_rows:
    x = eki2vivo_full.loc[strain]
    y_ekillm = ekillm_full.loc[strain]
    y_fba    = fba_full.loc[strain]
    mask = x.notna() & y_ekillm.notna()
    if mask.sum() > 3:
        _, p = wilcoxon(x[mask], y_ekillm[mask], alternative='greater')
        p_ekillm.loc[strain, mask.index[mask]] = p
    mask = x.notna() & y_fba.notna()
    if mask.sum() > 3:
        _, p = wilcoxon(x[mask], y_fba[mask], alternative='greater')
        p_fba.loc[strain, mask.index[mask]] = p

# ========== 3. 绘图 ==========
fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharey=True)

# 颜色范围
vmin, vmax = -0.3, 0.3

# 子图1：vs EkiLLM
sns.heatmap(diff_ekillm_amp.loc[row_order], ax=axes[0],
            cmap='coolwarm', vmin=vmin, vmax=vmax, center=0,
            cbar_kws={'label': 'Δ² Heat Score (amplified)'},
            xticklabels=False, yticklabels=True)
axes[0].set_title('Eki2vivo vs EkiLLM', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Strains', fontsize=12)

# 加星号
for i, strain in enumerate(row_order):
    for j, rxn in enumerate(common_cols):
        if pd.notna(p_ekillm.loc[strain, rxn]) and p_ekillm.loc[strain, rxn] < 0.05:
            axes[0].text(j + 0.5, i + 0.5, '*', ha='center', va='center', color='black', fontsize=8, fontweight='bold')

# 子图2：vs FBA
sns.heatmap(diff_fba_amp.loc[row_order], ax=axes[1],
            cmap='coolwarm', vmin=vmin, vmax=vmax, center=0,
            cbar_kws={'label': 'Δ² Heat Score (amplified)'},
            xticklabels=False)
axes[1].set_title('Eki2vivo vs FBA', fontsize=14, fontweight='bold')

# 加星号
for i, strain in enumerate(row_order):
    for j, rxn in enumerate(common_cols):
        if pd.notna(p_fba.loc[strain, rxn]) and p_fba.loc[strain, rxn] < 0.05:
            axes[1].text(j + 0.5, i + 0.5, '*', ha='center', va='center', color='black', fontsize=8, fontweight='bold')

# 标题
fig.suptitle('Fig4: Eki2vivo Advantage Across All Strains (Δ² amplified, * p<0.05)', 
             fontsize=16, fontweight='bold', y=0.95)

plt.tight_layout()
plt.subplots_adjust(top=0.90, wspace=0.1)
plt.savefig(os.path.join(OUTPUT_DIR, "Fig4_full_strain_difference_heatmap_significant.png"), dpi=400, bbox_inches='tight')
plt.close()
print("→ Fig4_full_strain_difference_heatmap_significant.png")

# ---------- 图5：火山图（Eki2vivo vs EkiLLM & vs FBA） ----------
print("→ Fig5_volcano_plot.png")

from scipy.stats import wilcoxon
import matplotlib.patches as mpatches

fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# 颜色映射：wildtype 红色，其他灰色
colors = ['red' if s == 'wildtype' else 'lightgray' for s in common_rows]

for idx, (method, ax) in enumerate(zip(['EkiLLM', 'FBA'], axes)):
    diff_df = method_dfs['Eki2vivo'] - method_dfs[method]
    p_vals = []
    deltas = []
    labels = []

    for strain in common_rows:
        x = method_dfs['Eki2vivo'].loc[strain]
        y = method_dfs[method].loc[strain]
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            continue
        delta = x[mask].mean() - y[mask].mean()
        _, p = wilcoxon(x[mask], y[mask], alternative='greater')
        p_vals.append(p)
        deltas.append(delta)
        if strain == 'wildtype' and p < 0.05 and abs(delta) > 0.1:
            labels.append(f"{strain} (Δ={delta:.3f})")
        elif p < 0.05 and abs(delta) > 0.15:
            labels.append(strain)

    p_vals = np.array(p_vals)
    deltas = np.array(deltas)
    neg_log_p = -np.log10(p_vals.clip(1e-10))

    # 散点
    scatter = ax.scatter(deltas, neg_log_p, c=colors, s=60, alpha=0.7, edgecolors='k', linewidth=0.5)

    # 标注
    for i, label in enumerate(labels):
        ax.annotate(label, (deltas[i], neg_log_p[i]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    color='red' if 'wildtype' in label else 'black')

    # 显著性线
    ax.axhline(-np.log10(0.05), color='black', linestyle='--', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.axvline(0.1, color='gray', linestyle=':', alpha=0.7)
    ax.axvline(-0.1, color='gray', linestyle=':', alpha=0.7)

    ax.set_xlabel('Δ Heat Score (Eki2vivo - ' + method + ')', fontsize=12)
    ax.set_title(f'Eki2vivo vs {method}', fontsize=14, fontweight='bold')
    if idx == 0:
        ax.set_ylabel('-log₁₀(p-value)', fontsize=12)

# 图例
red_patch = mpatches.Patch(color='red', label='wildtype')
gray_patch = mpatches.Patch(color='lightgray', label='mutants')
fig.legend(handles=[red_patch, gray_patch], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.95))

fig.suptitle('Fig5: Volcano Plot of Eki2vivo Advantage', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.savefig(os.path.join(OUTPUT_DIR, "Fig5_volcano_plot.png"), dpi=400, bbox_inches='tight')
plt.close()
print("→ Fig5_volcano_plot.png")


print("\n全部完成！所有图片 & 表格已保存在：", OUTPUT_DIR)
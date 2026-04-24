import pandas as pd

# === 输入文件路径 ===
flux_file = "PRO.tsv"       # 替换为你的流量文件名
kcat_file = "kcat.tsv"       # 替换为你的kcat+MW文件名

# === 读取数据 ===
flux_df = pd.read_csv(flux_file, sep="\t")
kcat_df = pd.read_csv(kcat_file, sep="\t")

# === 提取kcat 和 分子量（mg/mmol） ===
kcat_series = kcat_df.iloc[0, 1:].astype(float)
kcat_series.index = kcat_df.columns[1:]

mw_series = kcat_df.iloc[1, 1:].astype(float)
mw_series.index = kcat_df.columns[1:]

# === 计算蛋白质使用量（flux / kcat × MW） ===
protein_usage = pd.DataFrame()

for enzyme in kcat_series.index:
    if enzyme in flux_df.columns:
        flux = flux_df[enzyme]
        kcat = kcat_series[enzyme]
        mw = mw_series[enzyme]
        protein_usage[enzyme] = flux / mw  # 单位: mg/gDW/h

# === 添加 glucose 作为索引列（如果有） ===
if "glucose" in flux_df.columns:
    protein_usage["glucose"] = flux_df["glucose"]
    protein_usage = protein_usage.sort_values("glucose").set_index("glucose")

# === 可选：只导出关心的关键酶 ===
target_enzymes = ["AKGDH", "G6PDH2r", "PFK", "GLCptspp", "ASPK", "GLNS"]
export_data = protein_usage[target_enzymes]

# === 导出为 CSV 文件供 Prism 使用 ===
export_data.to_csv("extended_enzyme_usage_trend_for_prism.csv")
print("已导出: extended_enzyme_usage_trend_for_prism.csv")


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 读取数据
df = pd.read_csv("extended_enzyme_usage_trend_for_prism.csv")  # 替换为你的文件路径
df = df.set_index("glucose")

# 设置风格
sns.set_theme(style="whitegrid") 
colors = ["#F39C12", "#E74C3C", "#8E44AD", "#3498DB", "#f857c1", "#2ECC71"]  # 可根据图配色自定义

# 创建堆叠面积图
ax = df.plot(kind='area',
             stacked=True,
             figsize=(10, 6),
             color=colors,
             linewidth=0)

# 设置标签加粗
ax.set_xlabel("Glucose uptake", fontsize=14, fontweight='bold')
ax.set_ylabel("Estimated enzyme usage (a.u.)", fontsize=14, fontweight='bold')

# 图例加粗、位置调整
# legend = ax.legend(title=None, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
legend = ax.legend(title=None, loc='upper right', frameon=False)
for text in legend.get_texts():
    text.set_fontweight('bold')
    text.set_fontsize(12)

# 坐标轴刻度加粗
ax.tick_params(axis='both', which='major', labelsize=12, width=1.5)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# 去掉网格线
ax.grid(False)

# 美化边框和轴线
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("aaa.pdf", dpi=600)


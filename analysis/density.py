import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

df = pd.read_csv("proteomics_with_uniprot.csv", sep=",", encoding="utf-8")

# 示例数据
x = df['kcat_vitro_log10']
y = df['kcat_vivo_log10']
# 计算密度
xy = np.vstack([x, y])
density = gaussian_kde(xy)(xy)
# 将密度值添加到数据表
df['kcat_Density'] = density

# x = df['true_kms']
# y = df['pred_kms']
# xy = np.vstack([x, y])
# density = gaussian_kde(xy)(xy)
# df['km_Density'] = density

# 保存为新的 TSV 文件
df.to_csv("proteomics_with_uniprot_with_density.csv",
          sep=",",  # 使用制表符分隔
          index=False,  # 不保存行索引
          encoding="utf-8")  # 设置编码
print("save!")
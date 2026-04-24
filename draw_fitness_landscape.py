import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# ----------  适应度函数（多峰 landscape） ----------
def base_editor_landscape(x, y):
    """x,y ∈ [-10,10]"""
    # 1. 野生型 Cas 主峰
    cas  = 1.50 * np.exp(-(x**2 + y**2)/8)

    # 2. ABE 家族（左侧多峰）
    abe  = 0.80 * np.exp(-((x+5)**2 + (y-3)**2)/1.5)
    abe += 0.75 * np.exp(-((x+6)**2 + (y+2)**2)/2)
    abe += 0.70 * np.exp(-((x+4)**2 + (y+5)**2)/1.2)

    # 3. 高特异性 CBE / 窄窗 ABE（右侧尖峰）
    cbe  = 0.85 * np.exp(-((x-4)**2 + y**2)/0.5)   # σ 小→窄窗
    cbe += 0.65 * np.exp(-((x-3)**2 + (y-4)**2)/0.3)

    # 4. Off-target 低丘（后缘）
    off  = 0.20 * np.exp(-((x-2)**2 + (y+7)**2)/5)

    return cas + abe + cbe + off

# ---------- 网格采样 ----------
x = np.linspace(-7, 5.5, 400)
y = np.linspace(-7, 5.5, 400)
X, Y = np.meshgrid(x, y)
Z = base_editor_landscape(X, Y)

# ---------------- 画布 ----------------
fig = plt.figure(figsize=(8, 8))
# >>>>>>> 新增：统一背景色 <<<<<<<
# fig.patch.set_facecolor('#F4F8F1')          # figure 外框
ax = fig.add_subplot(111, projection='3d')
# ax.set_facecolor('#F4F8F1')                 # 3D 坐标系内部
ax = fig.add_subplot(111, projection='3d')

# 隐藏全部坐标轴、网格、背景、标签
ax.set_axis_off()
ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')

# 只画表面
surf = ax.plot_surface(X, Y, Z,
                       cmap='coolwarm',
                       linewidth=0,
                       antialiased=True,
                       rstride=1,
                       cstride=1,
                       alpha=0.5)

# 颜色条（可选，不想要就删掉下面一行）
# fig.colorbar(surf, shrink=0.6, aspect=12).ax.set_axis_off()
# ---------- 在你原来 rotate 函数之前插入 ----------
z_min = Z.min()          # 1. 地毯高度

# 2. 底部 2D 地毯（颜色跟 3D 完全一致）
ax.plot_surface(X, Y, np.full_like(Z, z_min),
                facecolors=plt.cm.coolwarm((Z - Z.min()) / (Z.max() - Z.min())),
                linewidth=0,
                alpha=0.5)          # 想淡就改 0.5

ax.set_axis_off()
for p in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    p.fill = False
    p.set_edgecolor('none')
# ---------------- 旋转动画 ----------------
def rotate(angle):
    ax.view_init(elev=25, azim=angle)

# ani = animation.FuncAnimation(fig, rotate,
#                               frames=np.linspace(0, 360, 120),
#                               interval=50, blit=False)

# ① 直接弹出窗口看动画
# plt.show()
plt.savefig("3d_fitness.png", dpi=600)
# plt.savefig("3d_fitness.pdf", dpi=600)

# ② 想保存为 mp4（需 ffmpeg）：
# ani.save("clean_rotate.mp4", writer='ffmpeg', dpi=200)
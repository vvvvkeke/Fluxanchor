import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from scipy.stats import gaussian_kde

# =========================== 核心参数配置区域 ===========================
# 用户需要根据实际数据修改以下参数
csv_file = 'data.csv'  # CSV数据文件路径
x_col = 'X_value'  # 自变量X的列名
y_col = 'Y_value'  # 因变量Y的列名
group_col = 'Group'  # 分组列的列名
x_label = 'X Value'  # X轴显示标签
y_label = 'Y Value'  # Y轴显示标签
output_filename = 'regression_plot .png'  # 输出图片文件名
figsize = (7, 7)  # 图片尺寸（宽, 高），单位英寸
confidence_level = 0.95  # 回归置信区间水平（默认95%）
version = 1  # 边缘图版本选择：1=箱线图；2=密度图；3=堆积直方图；4=抖动散点；5=堆积直方图+密度图；6=箱线图+半小提琴图；7=抖动散点+小提琴图；8=抖动散点+箱线图+小提琴图
colors = ['#96CCEA', '#B2A3DD', '#ED949A']  # 分组颜色列表（超过3组会循环使用）
# =====================================================================

# 设置全局绘图样式（符合期刊出版标准）
plt.rcParams.update({
    'font.family': 'Arial',  # 使用Arial字体（期刊常用）
    'font.weight': 'bold',  # 字体加粗
    'font.size': 14,  # 基础字体大小
})


def half_violin(ax, data, pos, side='right', width=0.3, color='blue', alpha=0.7):
    """
    绘制半小提琴图函数

    参数:
    ax: 目标坐标轴
    data: 数据数组
    pos: 位置坐标
    side: 方向 ('right', 'left', 'top', 'bottom')
    width: 小提琴宽度
    color: 填充颜色
    alpha: 透明度
    """
    # 计算核密度估计
    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 100)
    y = kde(x)

    # 归一化y值到指定宽度
    y = y / y.max() * width

    # 根据方向参数绘制不同方向的半小提琴图
    if side == 'right':
        ax.fill_betweenx(x, pos, pos + y, color=color, alpha=alpha)
    elif side == 'left':
        ax.fill_betweenx(x, pos - y, pos, color=color, alpha=alpha)
    elif side == 'top':
        ax.fill_between(x, pos, pos + y, color=color, alpha=alpha)
    elif side == 'bottom':
        ax.fill_between(x, pos - y, pos, color=color, alpha=alpha)


# =========================== 数据读取与初始化 ===========================
# 读取CSV数据文件
df = pd.read_csv(csv_file)

# 设置轴标签（如果未指定则使用列名）
x_label = x_label or x_col
y_label = y_label or y_col

# 获取所有唯一分组
groups = df[group_col].unique()

# 存储回归分析结果
results = []

# =========================== 图形布局设置 ===========================
# 创建图形对象和子图
fig = plt.figure(figsize=figsize)

# 定义子图位置参数（左距, 下距, 宽度, 高度, 间距）
left, width, bottom, height, spacing = 0.15, 0.60, 0.15, 0.60, 0.005

# 主图：散点图 + 回归线 + 置信区间
ax_scatter = plt.axes([left, bottom, width, height])

# 顶部边缘图：X轴数据分布
ax_histx = plt.axes([left, bottom + height + spacing, width, 0.15])

# 右侧边缘图：Y轴数据分布
ax_histy = plt.axes([left + width + spacing, bottom, 0.15, height])

# 设置主图坐标轴范围（添加8%边距，确保数据不贴边）
x_padding = (df[x_col].max() - df[x_col].min()) * 0.08
y_padding = (df[y_col].max() - df[y_col].min()) * 0.08
ax_scatter.set_xlim(df[x_col].min() - x_padding, df[x_col].max() + x_padding)
ax_scatter.set_ylim(df[y_col].min() - y_padding, df[y_col].max() + y_padding)

# 存储分组数据供后续边缘图使用
group_data_list = []

# =========================== 主图绘制：散点图与回归分析 ===========================
for i, group in enumerate(groups):
    # 提取当前分组数据
    group_data = df[df[group_col] == group]

    # 获取当前组颜色
    color = colors[i % len(colors)]

    # 存储分组数据
    group_data_list.append((group, group_data, color))

    # 绘制分组散点图
    ax_scatter.scatter(
        group_data[x_col], group_data[y_col],
        color=color, s=60, alpha=0.7,  # 点大小60，透明度0.7
        edgecolor='black', linewidth=1,  # 黑色描边，突出显示
    )

    # 线性回归分析（仅当样本量>1时进行）
    if len(group_data) > 1:
        # 准备回归数据
        X = group_data[[x_col]].values  # 自变量（需要2D数组）
        y = group_data[y_col].values  # 因变量

        # 拟合线性回归模型
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        # 计算回归统计量
        r2 = r2_score(y, y_pred)  # R²决定系数
        slope = model.coef_[0]  # 回归斜率
        intercept = model.intercept_  # 回归截距
        n = len(X)  # 样本量

        # 存储回归结果
        results.append({
            'group': group, 'r2': r2, 'slope': slope,
            'intercept': intercept, 'n': n, 'color': color
        })

        # 生成平滑回归线
        x_vals = np.linspace(X.min(), X.max(), 100)
        y_vals = model.predict(x_vals.reshape(-1, 1))

        # 计算95%置信区间
        mse = np.sum((y - y_pred) ** 2) / (n - 2)  # 均方误差
        # 标准误差计算
        se = np.sqrt(mse * (1 / n + (x_vals - X.mean()) ** 2 / np.sum((X - X.mean()) ** 2)))
        # t分布临界值（95%置信水平，自由度n-2）
        t_val = stats.t.ppf((1 + confidence_level) / 2, n - 2)

        # 绘制置信区间（半透明填充）
        ax_scatter.fill_between(
            x_vals, y_vals - t_val * se, y_vals + t_val * se,
            color=color, alpha=0.2, linewidth=0
        )

        # 绘制回归线（虚线样式）
        ax_scatter.plot(
            x_vals, y_vals, color=color, linestyle='--',
            linewidth=2.5, alpha=0.8
        )

# =========================== 边缘图绘制（根据版本选择） ===========================

# 版本1：箱线图边缘版
if version == 1:
    for i, (group, group_data, color) in enumerate(group_data_list):
        # 顶部边缘图：X轴水平箱线图
        bp_x = ax_histx.boxplot(
            [group_data[x_col].values], positions=[i + 1],
            vert=False, patch_artist=True,  # 水平方向，填充颜色
            widths=0.6, showfliers=False,  # 宽度0.6，不显示异常值
            boxprops=dict(linewidth=2, color=color),
            medianprops=dict(linewidth=2, color=color),
            whiskerprops=dict(linewidth=1.5, color=color),
            capprops=dict(linewidth=1.5, color=color)
        )
        bp_x['boxes'][0].set_facecolor(color)
        bp_x['boxes'][0].set_alpha(0.7)

        # 右侧边缘图：Y轴垂直箱线图
        bp_y = ax_histy.boxplot(
            [group_data[y_col].values], positions=[i + 1],
            vert=True, patch_artist=True,
            widths=0.6, showfliers=False,
            boxprops=dict(linewidth=2, color=color),
            medianprops=dict(linewidth=2, color=color),
            whiskerprops=dict(linewidth=1.5, color=color),
            capprops=dict(linewidth=1.5, color=color)
        )
        bp_y['boxes'][0].set_facecolor(color)
        bp_y['boxes'][0].set_alpha(0.7)

    # 调整箱线图坐标轴范围避免重叠
    ax_histx.set_ylim(0.5, len(groups) + 0.5)
    ax_histy.set_xlim(0.5, len(groups) + 0.5)

# 版本2：密度图边缘版
elif version == 2:
    all_x_densities = []
    all_y_densities = []

    for group, group_data, color in group_data_list:
        x_kde = gaussian_kde(group_data[x_col])
        y_kde = gaussian_kde(group_data[y_col])

        x_plot = np.linspace(ax_scatter.get_xlim()[0], ax_scatter.get_xlim()[1], 200)
        y_plot = np.linspace(ax_scatter.get_ylim()[0], ax_scatter.get_ylim()[1], 200)

        x_density = x_kde(x_plot)
        y_density = y_kde(y_plot)

        all_x_densities.append(x_density)
        all_y_densities.append(y_density)

        ax_histx.plot(x_plot, x_density, color=color, linewidth=2)
        ax_histx.fill_between(x_plot, x_density, alpha=0.3, color=color)

        ax_histy.plot(y_density, y_plot, color=color, linewidth=2)
        ax_histy.fill_betweenx(y_plot, y_density, alpha=0.3, color=color)

    # 计算统一的密度轴范围
    if all_x_densities and all_y_densities:
        max_x_density = max([density.max() for density in all_x_densities])
        max_y_density = max([density.max() for density in all_y_densities])
        unified_max_density = max(max_x_density, max_y_density) * 1.1  # 增加10%边距

        # 设置统一的密度轴范围
        ax_histx.set_ylim(0, unified_max_density)
        ax_histy.set_xlim(0, unified_max_density)

# 版本3：堆积直方图边缘版
elif version == 3:
    bins = 25
    x_min, x_max = ax_scatter.get_xlim()
    y_min, y_max = ax_scatter.get_ylim()

    x_bins = np.linspace(x_min, x_max, bins + 1)
    y_bins = np.linspace(y_min, y_max, bins + 1)

    x_hist_data = []
    y_hist_data = []

    for group, group_data, color in group_data_list:
        x_counts, _ = np.histogram(group_data[x_col], bins=x_bins)
        x_hist_data.append((x_counts, color, group))

        y_counts, _ = np.histogram(group_data[y_col], bins=y_bins)
        y_hist_data.append((y_counts, color, group))

    x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
    x_bin_width = x_bins[1] - x_bins[0]
    x_bottom = np.zeros(len(x_bin_centers))

    for x_counts, color, group in x_hist_data:
        ax_histx.bar(x_bin_centers, x_counts, width=x_bin_width,
                     bottom=x_bottom, color=color, alpha=0.7,
                     edgecolor='white', linewidth=0.5)
        x_bottom += x_counts

    y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2
    y_bin_width = y_bins[1] - y_bins[0]
    y_bottom = np.zeros(len(y_bin_centers))

    for y_counts, color, group in y_hist_data:
        ax_histy.barh(y_bin_centers, y_counts, height=y_bin_width,
                      left=y_bottom, color=color, alpha=0.7,
                      edgecolor='white', linewidth=0.5)
        y_bottom += y_counts

    # 计算统一的频率轴范围
    x_total_counts = np.sum([x_counts for x_counts, _, _ in x_hist_data], axis=0)
    y_total_counts = np.sum([y_counts for y_counts, _, _ in y_hist_data], axis=0)

    max_x_freq = np.max(x_total_counts) if len(x_total_counts) > 0 else 1
    max_y_freq = np.max(y_total_counts) if len(y_total_counts) > 0 else 1

    unified_max_freq = max(max_x_freq, max_y_freq) * 1.1

    ax_histx.set_ylim(0, unified_max_freq)
    ax_histy.set_xlim(0, unified_max_freq)

# 版本4：抖动散点边缘版
elif version == 4:
    # 计算抖动范围（数据范围的5%）
    x_jitter_range = (df[x_col].max() - df[x_col].min()) * 0.05
    y_jitter_range = (df[y_col].max() - df[y_col].min()) * 0.05

    for i, (group, group_data, color) in enumerate(group_data_list):
        # 为每个分组添加垂直偏移避免重叠
        y_offset = i * 0.8

        # 顶部边缘图：X轴抖动散点
        x_jitter = np.random.normal(0, x_jitter_range, len(group_data))
        ax_histx.scatter(group_data[x_col] + x_jitter,
                         np.full(len(group_data), y_offset),
                         color=color, alpha=0.6, s=30)

        # 右侧边缘图：Y轴抖动散点
        y_jitter = np.random.normal(0, y_jitter_range, len(group_data))
        ax_histy.scatter(np.full(len(group_data), y_offset),
                         group_data[y_col] + y_jitter,
                         color=color, alpha=0.6, s=30)

    # 调整坐标轴范围
    ax_histx.set_ylim(-0.5, len(groups) * 0.8 - 0.3)
    ax_histy.set_xlim(-0.5, len(groups) * 0.8 - 0.3)

# 版本5：堆积直方图+密度图
elif version == 5:
    bins = 25
    x_min, x_max = ax_scatter.get_xlim()
    y_min, y_max = ax_scatter.get_ylim()

    x_bins = np.linspace(x_min, x_max, bins + 1)
    y_bins = np.linspace(y_min, y_max, bins + 1)

    x_hist_data = []
    y_hist_data = []

    # 计算分组直方图
    for group, group_data, color in group_data_list:
        x_counts, _ = np.histogram(group_data[x_col], bins=x_bins)
        x_hist_data.append((x_counts, color, group))

        y_counts, _ = np.histogram(group_data[y_col], bins=y_bins)
        y_hist_data.append((y_counts, color, group))

    # 绘制X轴堆积直方图
    x_bin_centers = (x_bins[:-1] + x_bins[1:]) / 2
    x_bin_width = x_bins[1] - x_bins[0]
    x_bottom = np.zeros(len(x_bin_centers))

    for x_counts, color, group in x_hist_data:
        ax_histx.bar(x_bin_centers, x_counts, width=x_bin_width,
                     bottom=x_bottom, color=color, alpha=0.6,
                     edgecolor='white', linewidth=0.5)
        x_bottom += x_counts

    # 绘制Y轴堆积直方图
    y_bin_centers = (y_bins[:-1] + y_bins[1:]) / 2
    y_bin_width = y_bins[1] - y_bins[0]
    y_bottom = np.zeros(len(y_bin_centers))

    for y_counts, color, group in y_hist_data:
        ax_histy.barh(y_bin_centers, y_counts, height=y_bin_width,
                      left=y_bottom, color=color, alpha=0.6,
                      edgecolor='white', linewidth=0.5)
        y_bottom += y_counts

    # 计算统一的频率轴范围（用于直方图）
    x_total_counts = np.sum([x_counts for x_counts, _, _ in x_hist_data], axis=0)
    y_total_counts = np.sum([y_counts for y_counts, _, _ in y_hist_data], axis=0)

    max_x_freq = np.max(x_total_counts) if len(x_total_counts) > 0 else 1
    max_y_freq = np.max(y_total_counts) if len(y_total_counts) > 0 else 1
    unified_max_freq = max(max_x_freq, max_y_freq) * 1.1

    # 设置直方图的统一频率轴范围
    ax_histx.set_ylim(0, unified_max_freq)
    ax_histy.set_xlim(0, unified_max_freq)

    # 创建次坐标轴用于密度曲线（独立于直方图）
    ax_density_x = ax_histx.twinx()
    ax_density_y = ax_histy.twiny()

    # 收集所有密度数据
    all_x_densities = []
    all_y_densities = []

    for group, group_data, color in group_data_list:
        x_kde = gaussian_kde(group_data[x_col])
        y_kde = gaussian_kde(group_data[y_col])

        x_plot = np.linspace(ax_scatter.get_xlim()[0], ax_scatter.get_xlim()[1], 200)
        y_plot = np.linspace(ax_scatter.get_ylim()[0], ax_scatter.get_ylim()[1], 200)

        x_density = x_kde(x_plot)
        y_density = y_kde(y_plot)

        all_x_densities.append(x_density)
        all_y_densities.append(y_density)

    # 计算统一的密度轴范围（独立于直方图频率）
    if all_x_densities and all_y_densities:
        max_x_density = max([density.max() for density in all_x_densities])
        max_y_density = max([density.max() for density in all_y_densities])
        unified_max_density = max(max_x_density, max_y_density) * 1.1  # 增加10%边距

    # 绘制密度曲线（使用独立的密度坐标轴）
    for i, (group, group_data, color) in enumerate(group_data_list):
        x_density = all_x_densities[i]
        y_density = all_y_densities[i]

        # 绘制密度曲线（使用原始密度值）
        ax_density_x.plot(x_plot, x_density, color=color, linewidth=1, alpha=0.9, linestyle='--')
        ax_density_x.fill_between(x_plot, x_density, alpha=0.2, color=color)

        ax_density_y.plot(y_density, y_plot, color=color, linewidth=1, alpha=0.9, linestyle='--')
        ax_density_y.fill_betweenx(y_plot, y_density, alpha=0.15, color=color)

    # 设置密度曲线坐标轴范围（独立的密度轴）
    ax_density_x.set_ylim(0, unified_max_density)
    ax_density_y.set_xlim(0, unified_max_density)

    # 隐藏次坐标轴刻度
    ax_density_x.set_yticks([])
    ax_density_y.set_xticks([])

    # 隐藏次坐标轴边框
    for spine in ax_density_x.spines.values():
        spine.set_visible(False)
    for spine in ax_density_y.spines.values():
        spine.set_visible(False)


# 版本6：箱线图+半小提琴图
elif version == 6:
    violin_scale = 1.7  # 小提琴图缩放因子
    positions = np.arange(1, len(groups) + 1)

    for i, (group, group_data, color) in enumerate(group_data_list):
        pos = positions[i]

        # 顶部边缘图：水平半小提琴图 + 箱线图
        half_violin(ax_histx, group_data[x_col].values, pos,
                    side='top', width=0.3 * violin_scale,
                    color=color, alpha=0.5)

        # 水平箱线图（显示异常值）
        bp_x = ax_histx.boxplot(
            [group_data[x_col].values], positions=[pos],
            vert=False, patch_artist=True,
            widths=0.3, showfliers=True,
            boxprops=dict(linewidth=2, color=color, facecolor='white'),
            medianprops=dict(linewidth=2, color=color),
            whiskerprops=dict(linewidth=2, color=color),
            capprops=dict(linewidth=2, color=color),
            flierprops=dict(marker='o', markerfacecolor='white',
                            markeredgecolor=color, markersize=6)
        )

        # 右侧边缘图：垂直半小提琴图 + 箱线图
        half_violin(ax_histy, group_data[y_col].values, pos,
                    side='right', width=0.3 * violin_scale,
                    color=color, alpha=0.7)

        # 垂直箱线图（显示异常值）
        bp_y = ax_histy.boxplot(
            [group_data[y_col].values], positions=[pos],
            vert=True, patch_artist=True,
            widths=0.3, showfliers=True,
            boxprops=dict(linewidth=2, color=color, facecolor='white'),
            medianprops=dict(linewidth=2, color=color),
            whiskerprops=dict(linewidth=2, color=color),
            capprops=dict(linewidth=2, color=color),
            flierprops=dict(marker='o', markerfacecolor='white',
                            markeredgecolor=color, markersize=6)
        )

    # 调整坐标轴范围
    ax_histx.set_ylim(0.5, len(groups) + 0.5)
    ax_histy.set_xlim(0.5, len(groups) + 0.5)

    # 同步主图和边缘图坐标轴范围
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

# 版本7：抖动散点+小提琴图
elif version == 7:
    violin_scale = 1.7
    jitter_amount = 0.15
    scatter_offset = -0.10

    for i, (group, group_data, color) in enumerate(group_data_list):
        y_offset = i * 0.8  # 顶部边缘图垂直偏移
        x_offset = i * 0.8  # 右侧边缘图水平偏移

        # 顶部边缘图：小提琴图 + 抖动散点
        half_violin(ax_histx, group_data[x_col].values, y_offset,
                    side='top', width=0.3 * violin_scale,
                    color=color, alpha=0.3)

        # 添加抖动散点
        x_jitter = np.random.normal(0, jitter_amount * (df[x_col].max() - df[x_col].min()), len(group_data))
        scatter_y_positions = np.full(len(group_data), y_offset + scatter_offset)
        ax_histx.scatter(group_data[x_col] + x_jitter,
                         scatter_y_positions,
                         color=color, alpha=1, s=20,
                         edgecolor=color, linewidth=0.3, zorder=10)

        # 右侧边缘图：小提琴图 + 抖动散点
        half_violin(ax_histy, group_data[y_col].values, x_offset,
                    side='right', width=0.3 * violin_scale,
                    color=color, alpha=0.3)

        y_jitter = np.random.normal(0, jitter_amount * (df[y_col].max() - df[y_col].min()), len(group_data))
        scatter_x_positions = np.full(len(group_data), x_offset + scatter_offset)
        ax_histy.scatter(scatter_x_positions,
                         group_data[y_col] + y_jitter,
                         color=color, alpha=1, s=20,
                         edgecolor=color, linewidth=0.3, zorder=10)

    # 调整坐标轴范围
    ax_histx.set_ylim(-0.5, len(groups) * 0.8 - 0.3)
    ax_histy.set_xlim(-0.5, len(groups) * 0.8 - 0.3)
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

# 版本8：抖动散点+箱线图+小提琴图
elif version == 8:
    box_width = 0.2
    scatter_offset = -0.25
    violin_scale = 1.7
    jitter_amount = 0.15

    for i, (group, group_data, color) in enumerate(group_data_list):
        y_offset = i * 0.8
        x_offset = i * 0.8

        # 顶部边缘图：三合一（小提琴图 + 箱线图 + 抖动散点）
        half_violin(ax_histx, group_data[x_col].values, y_offset,
                    side='top', width=0.25 * violin_scale,
                    color=color, alpha=0.2)

        # 水平箱线图（不显示异常值，用散点代替）
        bp_x = ax_histx.boxplot(
            [group_data[x_col].values], positions=[y_offset],
            vert=False, patch_artist=True,
            widths=box_width, showfliers=False,
            boxprops=dict(linewidth=2, color=color, facecolor='white', alpha=0.8),
            medianprops=dict(linewidth=2, color=color),
            whiskerprops=dict(linewidth=2, color=color, alpha=0.8),
            capprops=dict(linewidth=2, color=color, alpha=0.8)
        )

        # 添加抖动散点
        x_jitter = np.random.normal(0, jitter_amount * (df[x_col].max() - df[x_col].min()), len(group_data))
        scatter_y_positions = np.full(len(group_data), y_offset + scatter_offset)
        ax_histx.scatter(group_data[x_col] + x_jitter,
                         scatter_y_positions,
                         color=color, alpha=1, s=15,
                         edgecolor=color, linewidth=0.3, zorder=10)

        # 右侧边缘图：三合一（小提琴图 + 箱线图 + 抖动散点）
        half_violin(ax_histy, group_data[y_col].values, x_offset,
                    side='right', width=0.25 * violin_scale,
                    color=color, alpha=0.2)

        # 垂直箱线图（不显示异常值，用散点代替）
        bp_y = ax_histy.boxplot(
            [group_data[y_col].values], positions=[x_offset],
            vert=True, patch_artist=True,
            widths=box_width, showfliers=False,
            boxprops=dict(linewidth=2, color=color, facecolor='white', alpha=0.8),
            medianprops=dict(linewidth=2, color=color),
            whiskerprops=dict(linewidth=2, color=color, alpha=0.8),
            capprops=dict(linewidth=2, color=color, alpha=0.8)
        )

        # 添加抖动散点
        y_jitter = np.random.normal(0, jitter_amount * (df[y_col].max() - df[y_col].min()), len(group_data))
        scatter_x_positions = np.full(len(group_data), x_offset + scatter_offset)
        ax_histy.scatter(scatter_x_positions,
                         group_data[y_col] + y_jitter,
                         color=color, alpha=1, s=15,
                         edgecolor=color, linewidth=0.3, zorder=10)

    # 调整坐标轴范围，考虑散点和小提琴图的偏移
    violin_width = 0.25 * violin_scale
    ax_histx.set_ylim(-0.2 - violin_width, len(groups) * 0.8 + scatter_offset + violin_width)
    ax_histy.set_xlim(-0.2 - violin_width, len(groups) * 0.8 + scatter_offset + violin_width)

    # 同步主图和边缘图坐标轴范围
    ax_histx.set_xlim(ax_scatter.get_xlim())
    ax_histy.set_ylim(ax_scatter.get_ylim())

# =========================== 图形美化与标注 ===========================

# 在主图左上角添加R²统计信息
y_pos = 0.93  # 起始位置（主图高度的93%处）
for result in results:
    ax_scatter.text(
        0.03, y_pos,
        f"{result['group']} : $R^2$ = {result['r2']:.3f}",  # 使用LaTeX格式的R²符号
        color=result['color'], transform=ax_scatter.transAxes,
        fontsize=14, fontweight='bold'
    )
    y_pos -= 0.06  # 每组文本向下移动6%

# 设置主图坐标轴标签
ax_scatter.set_xlabel(x_label, fontsize=16, fontweight='bold')
ax_scatter.set_ylabel(y_label, fontsize=16, fontweight='bold')

# 隐藏边缘图的所有坐标轴
for ax in [ax_histx, ax_histy]:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# 同步主图和边缘图的坐标轴范围
ax_histx.set_xlim(ax_scatter.get_xlim())
ax_histy.set_ylim(ax_scatter.get_ylim())

# =========================== 保存与输出结果 ===========================

# 保存为高分辨率图片（适合期刊投稿的600dpi）
plt.savefig(output_filename, dpi=600)

# 显示图片
plt.show()

# 在控制台输出回归分析结果（便于记录和分析）
print("=" * 50)
print("回归分析结果汇总")
print("=" * 50)
for i, result in enumerate(results, 1):
    print(f"\n【{result['group']}】")
    print(f"样本量 n = {result['n']}")
    print(f"R² = {result['r2']:.4f}")
    print(f"回归方程：Y = {result['intercept']:.4f} + {result['slope']:.4f} × X")
    if i < len(results):
        print("-" * 40)
print("=" * 50)
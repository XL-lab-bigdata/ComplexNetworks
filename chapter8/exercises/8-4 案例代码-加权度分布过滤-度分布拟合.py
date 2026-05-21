import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import r2_score

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取边数据（包含 source, target 和 weight）
edges_data = pd.read_csv('edges_with_ports.csv')

# 创建无权图和加权图
G_unweighted = nx.Graph()
G_weighted = nx.Graph()

# 添加无权图和加权图的边
for _, row in edges_data.iterrows():
    G_unweighted.add_edge(row['source_port'], row['target_port'])
    G_weighted.add_edge(row['source_port'], row['target_port'], weight=row['Weight'])

# 1. 无权度分布计算和累计度分布
degree_sequence_unweighted = sorted([d for n, d in G_unweighted.degree()], reverse=True)
degree_count_unweighted = np.bincount(degree_sequence_unweighted)
degree_prob_unweighted = degree_count_unweighted / sum(degree_count_unweighted)
cumulative_degree_prob_unweighted = np.cumsum(degree_prob_unweighted[::-1])[::-1]
degree_range_unweighted = np.arange(len(degree_prob_unweighted))

# 拟合无权图
fit_unweighted = np.polyfit(np.log(degree_range_unweighted[degree_prob_unweighted > 0]),
                            np.log(cumulative_degree_prob_unweighted[degree_prob_unweighted > 0]), 1)
fit_fn_unweighted = np.poly1d(fit_unweighted)

# 计算拟合优度 R^2
log_degree_unweighted = np.log(degree_range_unweighted[degree_prob_unweighted > 0])
log_cumulative_prob_unweighted = np.log(cumulative_degree_prob_unweighted[degree_prob_unweighted > 0])
fit_values_unweighted = fit_fn_unweighted(log_degree_unweighted)
r_squared_unweighted = r2_score(log_cumulative_prob_unweighted, fit_values_unweighted)

# 2. 加权度分布计算和累计度分布
degree_sequence_weighted = sorted([d for n, d in G_weighted.degree(weight='weight')], reverse=True)
degree_sequence_weighted_filtered = [d for d in degree_sequence_weighted if d > 0]

# 过滤并重新计算加权度分布
degree_count_weighted_filtered = np.bincount([int(d) for d in degree_sequence_weighted_filtered])
degree_prob_weighted_filtered = degree_count_weighted_filtered / sum(degree_count_weighted_filtered)
cumulative_degree_prob_weighted_filtered = np.cumsum(degree_prob_weighted_filtered[::-1])[::-1]
degree_range_weighted_filtered = np.arange(len(degree_prob_weighted_filtered))

# 拟合加权图的非零部分
valid_indices = (degree_prob_weighted_filtered > 0) & (degree_range_weighted_filtered > 0)
fit_weighted_filtered = np.polyfit(np.log(degree_range_weighted_filtered[valid_indices]),
                                   np.log(cumulative_degree_prob_weighted_filtered[valid_indices]), 1)
fit_fn_weighted_filtered = np.poly1d(fit_weighted_filtered)

# 计算拟合优度 R^2
log_degree_weighted_filtered = np.log(degree_range_weighted_filtered[valid_indices])
log_cumulative_prob_weighted_filtered = np.log(cumulative_degree_prob_weighted_filtered[valid_indices])
fit_values_weighted_filtered = fit_fn_weighted_filtered(log_degree_weighted_filtered)
r_squared_weighted_filtered = r2_score(log_cumulative_prob_weighted_filtered, fit_values_weighted_filtered)

# 绘制两张图在同一画布上
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# 绘制无权图的累计度分布
axes[0].loglog(degree_range_unweighted[degree_prob_unweighted > 0],
               cumulative_degree_prob_unweighted[degree_prob_unweighted > 0],
               marker='o', linestyle='none', markerfacecolor='none', markeredgecolor='#81c784', markeredgewidth=1)

axes[0].plot(degree_range_unweighted[degree_prob_unweighted > 0],
             np.exp(fit_fn_unweighted(np.log(degree_range_unweighted[degree_prob_unweighted > 0]))),
             color='#9c27b0', label=f'Fit: $k^{{{fit_unweighted[0]:.2f}}}$, $R^2={r_squared_unweighted:.2f}$')

axes[0].set_title('Cumulative Degree Distribution (Unweighted)', fontsize=12)
axes[0].set_xlabel('Degree (k)', fontsize=10)
axes[0].set_ylabel('P(k)', fontsize=10)
axes[0].set_ylim(1e-3, 1.15)
# 将图例放在左下角
axes[0].legend(loc='lower left')

# 添加底部的(A)
axes[0].text(0.5, -0.15, '(A)', transform=axes[0].transAxes, fontsize=12, va='top', ha='center')

# 绘制加权图的累计度分布
axes[1].loglog(degree_range_weighted_filtered[valid_indices],
               cumulative_degree_prob_weighted_filtered[valid_indices],
               marker='o', linestyle='none', markerfacecolor='none', markeredgecolor='green', markeredgewidth=1)

axes[1].plot(degree_range_weighted_filtered[valid_indices],
             np.exp(fit_fn_weighted_filtered(np.log(degree_range_weighted_filtered[valid_indices]))),
             color='#9c27b0', label=f'Fit: $k^{{{fit_weighted_filtered[0]:.2f}}}$, $R^2={r_squared_weighted_filtered:.2f}$')

axes[1].set_title('Cumulative Degree Distribution (Weighted)', fontsize=12)
axes[1].set_xlabel('Degree (k)', fontsize=10)
axes[1].set_ylabel('P(k)', fontsize=10)
axes[1].set_ylim(1e-3, 1.15)
# 将图例放在左下角
axes[1].legend(loc='lower left')

# 添加底部的(B)
axes[1].text(0.5, -0.15, '(B)', transform=axes[1].transAxes, fontsize=12, va='top', ha='center')

# 调整子图之间的间距
plt.tight_layout()

# 保存为 PDF，dpi=600
plt.savefig('cumulative_degree_distribution_combined_green.pdf', format='pdf', dpi=600)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 设置参数
num_nodes = 100  # 节点数量
p = 0.1  # 边的生成概率
threshold_range = np.linspace(0, 0.26, 100)  # 节点阈值范围
degree_range = np.arange(1, 17)  # 节点度范围
num_trials = 100  # 试验次数

# 创建ER随机网络
G = nx.erdos_renyi_graph(num_nodes, p)

# 计算每个节点的度
degrees = np.array([G.degree(n) for n in G.nodes()])

# 存储活跃节点数
active_counts = np.zeros((len(degree_range), len(threshold_range)))

# 遍历每个阈值和节点度
for i, threshold in enumerate(threshold_range):
    for j, degree in enumerate(degree_range):
        active_nodes = np.zeros(num_nodes, dtype=bool)

        # 随机选择一个节点作为活跃节点
        initial_active = np.random.choice(num_nodes)
        active_nodes[initial_active] = True

        # 进行传播
        for _ in range(num_trials):
            new_active_nodes = active_nodes.copy()
            for k in range(num_nodes):
                if not active_nodes[k]:
                    # 计算邻居的影响力
                    active_neighbors = sum(active_nodes[n] for n in G.neighbors(k))
                    if active_neighbors / G.degree(k) > threshold:
                        new_active_nodes[k] = True
            active_nodes = new_active_nodes

        # 统计活跃节点数
        active_counts[j, i] = np.sum(active_nodes)

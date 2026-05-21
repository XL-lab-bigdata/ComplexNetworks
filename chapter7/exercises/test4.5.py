import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt

# -------------------------------
# 1. 构建随机无向图
# -------------------------------
N = 100         # 节点数
avg_degree = 4  # 平均度
p = avg_degree / (N - 1)  # 随机图连边概率

G = nx.erdos_renyi_graph(N, p)

# -------------------------------
# 2. 初始化节点状态和阈值
# -------------------------------
state = {node: 0 for node in G.nodes()}  # 0表示未激活，1表示已激活
threshold = {node: random.uniform(0.1, 0.3) for node in G.nodes()}

# -------------------------------
# 3. 随机选择初始激活节点
# -------------------------------
initial_active = random.sample(G.nodes(), 5)
for node in initial_active:
    state[node] = 1

print(f"初始激活节点: {initial_active}")

# -------------------------------
# 4. 仿真传播过程
# -------------------------------
round_num = 0
activation_counts = []  # 每轮新激活节点数量

while True:
    new_activated = []
    for node in G.nodes():
        if state[node] == 0:
            neighbors = list(G.neighbors(node))
            if len(neighbors) == 0:
                continue
            active_neighbors = sum(state[n] for n in neighbors)
            if active_neighbors / len(neighbors) >= threshold[node]:
                new_activated.append(node)
    # 更新节点状态
    for node in new_activated:
        state[node] = 1

    activation_counts.append(len(new_activated))
    round_num += 1
    print(f"轮次 {round_num}: 新激活节点数 = {len(new_activated)}")

    if len(new_activated) == 0:
        break

# -------------------------------
# 5. 绘制每轮激活节点数量曲线
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(range(1, round_num+1), activation_counts, marker='o', color='orange')
plt.xlabel("轮次", fontsize=14)
plt.ylabel("新激活节点数", fontsize=14)
plt.title("线性阈值模型传播仿真", fontsize=16)
plt.grid(True)
plt.show()
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------
# （1）ER随机网络 G(N, M) ：固定节点数和边数
# ---------------------------------------
N1 = 10000  # 节点数
M1 = 20000  # 边数

# 根据边数计算连边概率 p
p1 = M1 * 2 / (N1 * (N1 - 1))

# 使用连边概率生成ER网络
G1 = nx.erdos_renyi_graph(N1, p1)

# 绘制度分布
degree_sequence1 = [d for n, d in G1.degree()]
degree_count1 = {}
for d in degree_sequence1:
    degree_count1[d] = degree_count1.get(d, 0) + 1

plt.figure(figsize=(8,5))
plt.bar(degree_count1.keys(), degree_count1.values(), color='skyblue')
plt.title(f'ER Random Graph G(N={N1}, M={M1}) via p={p1:.5f}')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.show()
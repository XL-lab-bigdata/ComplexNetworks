import networkx as nx
import matplotlib.pyplot as plt

# -------------------------------
# 参数设置
# -------------------------------
N = 10000       # 节点数
k = 10          # 每个节点初始连接的最近邻数量 (必须为偶数)
p = 0.2         # 重连概率

# -------------------------------
# 生成WS小世界网络
# -------------------------------
G = nx.watts_strogatz_graph(n=N, k=k, p=p)

# -------------------------------
# 计算度分布
# -------------------------------
degree_sequence = [d for n, d in G.degree()]
degree_count = {}
for d in degree_sequence:
    degree_count[d] = degree_count.get(d, 0) + 1

plt.figure(figsize=(8,5))
plt.bar(degree_count.keys(), degree_count.values(), color='skyblue')
plt.title(f'WS Small-World Network Degree Distribution (N={N}, <k>={k}, p={p})')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.show()

# -------------------------------
# 计算节点集聚系数分布
# -------------------------------
clustering_coeffs = nx.clustering(G)
clustering_values = list(clustering_coeffs.values())

plt.figure(figsize=(8,5))
plt.hist(clustering_values, bins=50, color='salmon', edgecolor='black')
plt.title(f'WS Small-World Network Clustering Coefficient Distribution (N={N}, <k>={k}, p={p})')
plt.xlabel('Clustering coefficient')
plt.ylabel('Number of nodes')
plt.show()
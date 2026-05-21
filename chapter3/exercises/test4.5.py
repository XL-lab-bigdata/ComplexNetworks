import networkx as nx
import matplotlib.pyplot as plt

# -------------------------------
# 参数设置
# -------------------------------
N = 100000  # 节点总数
m = 3       # 每个新节点连边数

# -------------------------------
# 生成BA无标度网络
# -------------------------------
G = nx.barabasi_albert_graph(n=N, m=m)

# -------------------------------
# 绘制度分布
# -------------------------------
degree_sequence = [d for n, d in G.degree()]
degree_count = {}
for d in degree_sequence:
    degree_count[d] = degree_count.get(d, 0) + 1

plt.figure(figsize=(8,5))
plt.bar(degree_count.keys(), degree_count.values(), color='skyblue')
plt.title(f'BA Scale-Free Network Degree Distribution (N={N}, m={m})')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.show()

# -------------------------------
# 计算平均最短路径长度
# -------------------------------
# 对大网络，为了速度可以使用近似值
if nx.is_connected(G):
    avg_shortest_path_length = nx.average_shortest_path_length(G)
else:
    # 对于大网络，取最大连通分量计算平均最短路径
    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc)
    avg_shortest_path_length = nx.average_shortest_path_length(G_lcc)

print(f"Average shortest path length: {avg_shortest_path_length:.4f}")

# -------------------------------
# 计算网络集聚系数
# -------------------------------
avg_clustering = nx.average_clustering(G)
print(f"Average clustering coefficient: {avg_clustering:.4f}")
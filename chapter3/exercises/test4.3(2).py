import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------
# （2）ER随机网络 G(N, p) ：固定节点数和连边概率
# ---------------------------------------
N2 = 10000
p2 = 0.6  # 直接给定连边概率

G2 = nx.erdos_renyi_graph(N2, p2)

# 绘制度分布
degree_sequence2 = [d for n, d in G2.degree()]
degree_count2 = {}
for d in degree_sequence2:
    degree_count2[d] = degree_count2.get(d, 0) + 1

plt.figure(figsize=(8,5))
plt.bar(degree_count2.keys(), degree_count2.values(), color='salmon')
plt.title(f'ER Random Graph G(N={N2}, p={p2})')
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.show()
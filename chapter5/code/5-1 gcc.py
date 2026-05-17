import networkx as nx
G = nx.barabasi_albert_graph(1000,3)
connected_components = list(nx.connected_components(G)
largest_cc = len(max(nx.connected_components(G), key=len))
nx.connected_components(G)
# 返回的是节点集合的生成器，每个节点对应一个G的分量（子图），然后通过max函数计算最大连通片规模。

import networkx as nx
import networkx.algorithms.community as nx_comm
G = nx.Graph()  # 构建空的网络
G.add_edges_from([('v1', 'v2'), ('v1', 'v3'), ('v1', 'v4'), ('v2', 'v3'), ('v2', 'v4'), ('v3', 'v4'), ('v4', 'v5'), ('v5', 'v6'), ('v5', 'v7'), ('v6', 'v7') # 往网络中添加边
print(nx_comm.modularity(G, [{'v1', 'v2', 'v3', 'v4'}, {'v5', 'v6', 'v7'}]))  # 使用networkx自带的函数计算模块度，其中，G表示网络，[{'v1', 'v2', 'v3', 'v4'}, {'v5', 'v6', 'v7'}]表示社团划分

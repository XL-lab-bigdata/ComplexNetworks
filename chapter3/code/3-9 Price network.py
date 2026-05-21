import networkx as nx
import random
# 生成 Price 无标度网络的函数，从孤立节点开始
def price_model(n, m):
    G = nx.DiGraph()  # 有向图
    G.add_nodes_from(range(m))  # 初始生成 m 个孤立节点
    for i in range(m, n):
        G.add_node(i)
        targets = _select_targets(G, m)
        for t in targets:
            G.add_edge(i, t)
    return G

# 选择目标节点的函数
def _select_targets(G, m):
    targets = set()
    nodes = list(G.nodes())
    in_degrees = [G.in_degree(node) + 1 for node in nodes]  # 防止节点入度为0的情况
    total_in_degrees = sum(in_degrees)
    probabilities = [in_degree / total_in_degrees for in_degree in in_degrees]
    while len(targets) < m:
        node = random.choices(nodes, weights=probabilities, k=1)[0]
        targets.add(node)   
    return targets

# 设置模型参数
n = 100  # 总节点数
m = 3    # 每个新节点引入的边数
G = price_model(n, m) # 生成 Price 模型网络
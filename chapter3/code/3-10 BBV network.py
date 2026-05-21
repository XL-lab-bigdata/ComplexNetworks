import networkx as nx
import numpy as np
# 生成 BBV 模型的函数
def bbv_model(n, m, beta):
    # 初始化图为完全图，包含 m 个节点
    G = nx.complete_graph(m)
    nx.set_edge_attributes(G, 1, 'weight')  # 初始化边的权重为 1

    # 逐步添加新节点，直到达到 n 个节点
    for i in range(m, n):
        # 选择目标节点，使用概率选择 m 个节点
        targets = _select_targets(G, m)
        G.add_node(i)  # 添加新节点 i

        # 将新节点连接到目标节点，并更新权重
        for t in targets:
            G.add_edge(i, t, weight=1)  # 新边的初始权重为 1
            _update_weights(G, i, t, beta)  # 根据邻居节点更新权重   
    return G

# 选择目标节点的函数，基于节点的连接概率
def _select_targets(G, m):
    targets = set()  # 存储选择的目标节点
    probabilities = _compute_attachment_probabilities(G)  # 计算节点的连接概率
    
    # 选择 m 个不同的目标节点
    while len(targets) < m:
        node = np.random.choice(list(G.nodes()), p=probabilities)  # 按概率选择节点
        targets.add(node)  # 添加到目标集合中
    
    return targets

# 计算所有节点的连接概率，基于节点的加权度
def _compute_attachment_probabilities(G):
    total_strength = sum(G.degree(node, weight='weight') for node in G.nodes())  # 计算网络的总加权度
    probabilities = []
    
    if total_strength == 0:
        # 如果没有总强度，返回均匀概率
        return [1 / len(G.nodes())] * len(G.nodes())
    
    # 计算每个节点的连接概率
    for node in G.nodes():
        degree_weight = G.degree(node, weight='weight')  # 获取节点的加权度
        probability = degree_weight / total_strength  # 连接概率为节点加权度占总加权度的比例
        probabilities.append(probability)
    
    # 确保概率归一化
    sum_probabilities = sum(probabilities)
    if sum_probabilities != 0:
        probabilities = [p / sum_probabilities for p in probabilities]  # 归一化处理
    else:
        probabilities = [1 / len(G.nodes())] * len(G.nodes())  # 如果出现问题，使用均匀概率
    
    return probabilities

# 更新节点间边的权重，根据 beta 参数和节点的加权度进行调整
def _update_weights(G, i, t, beta):
    for node in [i, t] + list(G.neighbors(t)):  # 遍历节点 i、t 及 t 的邻居节点
        G[i][t]['weight'] += beta * G.degree(node, weight='weight')  # 根据邻居的加权度更新边的权重

# 设置模型参数
n = 1000  # 节点总数
m = 3     # 每个新节点连接的目标节点数量
beta = 0.5  # 权重因子

# 生成 BBV 模型网络
G = bbv_model(n, m, beta)

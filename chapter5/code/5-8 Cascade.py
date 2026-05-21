import numpy as np
import networkx as nx
import random
from copy import deepcopy


def initialize_load_capacity_degree_based(G, beta):
    """
    初始化节点的负载和容量，基于度中心性。

    参数:
    G (networkx.Graph): 要初始化的图。
    beta (float): 容量系数，决定节点容量相对于其度的倍数。
    """
    for node in G.nodes():
        degree = G.degree[node]  # 获取节点的度
        G.nodes[node]['load'] = degree  # 初始负载与节点度成正比
        G.nodes[node]['capacity'] = (1 + beta) * degree  # 容量为节点度的 (1 + beta) 倍


def perform_cascade_failure(G):
    """
    执行级联失效过程。

    参数:
    G (networkx.Graph): 要执行级联失效的图。

    返回:
    float: 最大连通子图的比例。
    """
    while True:
        new_failures = []  # 存储新的失效节点
        for node in G.nodes():
            if G.nodes[node]['load'] > G.nodes[node]['capacity']:  # 如果节点负载超过其容量
                new_failures.append(node)  # 节点失效
        if not new_failures:
            break  # 如果没有新的失效节点，结束循环
        # 重新分配失效节点的负载
        for node in new_failures:
            load_to_redistribute = G.nodes[node]['load']  # 获取失效节点的负载
            neighbors = list(G.neighbors(node))  # 获取失效节点的邻居节点
            if len(neighbors) > 0:
                redistributed_load = load_to_redistribute / len(neighbors)
                for neighbor in neighbors:
                    G.nodes[neighbor]['load'] += redistributed_load  # 负载均等分配给邻居节点
        # 从网络中移除失效节点
        G.remove_nodes_from(new_failures)
    # 找到最大的连通子图
    if len(G) > 0:
        largest_cc = max(nx.connected_components(G), key=len)  # 找到最大连通子图
        return len(largest_cc) / len(G.nodes())  # 返回最大连通子图的比例
    else:
        return 0  # 如果网络为空，返回0


def simulate_cascade_failure(removal_ratio, G, attack_type='random'):
    """
    模拟给定移除比例下的级联失效。

    参数:
    removal_ratio (float): 要移除的节点比例（0到1之间）。
    G (networkx.Graph): 要模拟的图。
    attack_type (str): 攻击类型，'random' 或 'degree'。

    返回:
    float: 最大连通子图的比例。
    """
    G_copy = deepcopy(G)  # 创建网络的深拷贝，避免修改原始网络
    if attack_type == 'random':
        nodes_to_remove = random.sample(list(G_copy.nodes()), int(removal_ratio * len(G_copy)))  # 随机选择要移除的节点
    elif attack_type == 'degree':
        nodes_sorted_by_degree = sorted(G_copy.nodes(), key=lambda n: G_copy.degree(n), reverse=True)  # 根据度从高到低排序节点
        nodes_to_remove = nodes_sorted_by_degree[:int(removal_ratio * len(G_copy))]  # 选择度最高的节点进行移除
    else:
        raise ValueError("Unsupported attack type. Choose 'random' or 'degree'.")
    G_copy.remove_nodes_from(nodes_to_remove)  # 从网络中移除选定的节点
    return perform_cascade_failure(G_copy)  # 执行级联失效过程并返回结果

# 示例
# 创建一个示例图
G = nx.erdos_renyi_graph(n=100, p=0.05)

# 初始化节点的负载和容量
initialize_load_capacity_degree_based(G, beta=0.5)

# 模拟随机攻击导致的级联失效
result_random = simulate_cascade_failure(removal_ratio=0.1, G=G, attack_type='random')
print(f"随机攻击后的最大连通子图比例: {result_random}")

# 模拟基于度的攻击导致的级联失效
result_degree = simulate_cascade_failure(removal_ratio=0.1, G=G, attack_type='degree')
print(f"基于度攻击后的最大连通子图比例: {result_degree}")

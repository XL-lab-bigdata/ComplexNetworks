import numpy as np
import networkx as nx
import random
from copy import deepcopy


# 计算网络度的二阶矩
def calculate_degree_second_moment(G):
    degrees = [d for n, d in G.degree()]
    if len(degrees) == 0:
        return 0
    return np.mean([d ** 2 for d in degrees])


# 计算网络的平均度
def calculate_average_degree(G):
    degrees = [d for n, d in G.degree()]
    if len(degrees) == 0:
        return 0
    return np.mean(degrees)


def simulate_attack(G_, attack_type='random'):
    """
    模拟网络攻击，包括随机失效和基于度的蓄意攻击，记录每次移除节点后最大连通子图的比例
    """
    G = deepcopy(G_)  # 创建网络的深拷贝，避免修改原始网络
    number_of_nodes = G.number_of_nodes()  # 获取网络的节点数
    largest_all = np.zeros(number_of_nodes + 1)  # 初始化数组，用于记录每次攻击后最大连通子图的比例
    recorded_pc = False  # 标志位，用于记录临界移除比例
    nodes_sorted_by_degree = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)  # 根据度从高到低排序节点

    for i in range(number_of_nodes):
        if G.number_of_nodes() == 0:
            break  # 如果所有节点都被移除，则退出循环
        largest_cc = len(max(nx.connected_components(G), key=len))  # 计算最大连通子图的大小
        largest_all[i] = largest_cc / number_of_nodes  # 记录最大连通子图的比例

        if attack_type == 'random':
            remove_nodes = random.sample(list(G.nodes()), 1)
            G.remove_nodes_from(remove_nodes)  # 随机移除一个节点
        elif attack_type == 'degree':
            remove_nodes = [nodes_sorted_by_degree[i]]  # 按照节点度从高到低移除一个节点
            G.remove_nodes_from(remove_nodes)

        if not recorded_pc and calculate_degree_second_moment(G) / calculate_average_degree(G) <= 2:
            pc = (i + 1) / number_of_nodes  # 记录渗流阈值
            recorded_pc = True  # 更新标志位

    if not recorded_pc:
        pc = 1.0  # 如果没有记录到pc值，则默认pc值为1.0

    return largest_all, pc  # 返回每次攻击后最大连通子图的比例数组和pc值


def remove_nodes_by_betweenness(G):
    """
    按照节点介数从高到低移除网络中的一个节点
    """
    betweenness = nx.betweenness_centrality(G)  # 计算所有节点的介数中心性
    nodes_sorted_by_betweenness = sorted(betweenness.keys(), key=lambda n: betweenness[n], reverse=True)  # 根据介数从高到低排序节点
    remove_nodes = nodes_sorted_by_betweenness[:1]  # 选择介数最高的一个节点
    G.remove_nodes_from(remove_nodes)  # 从网络中移除该节点


def remove_nodes_by_degree(G):
    """
    按照节点度从高到低移除网络中的一个节点
    """
    nodes_sorted_by_degree = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)  # 根据度从高到低排序节点
    remove_nodes = nodes_sorted_by_degree[:1]  # 选择度最高的一个节点
    G.remove_nodes_from(remove_nodes)  # 从网络中移除该节点


def iterative_attack(G_, attack_type='degree'):
    """
    模拟迭代攻击，包括基于度和基于介数的攻击，记录每次移除节点后最大连通子图的比例
    """
    G = deepcopy(G_)  # 创建网络的深拷贝，避免修改原始网络
    number_of_nodes = G.number_of_nodes()  # 获取网络的节点数
    largest_all = np.zeros(number_of_nodes + 1)  # 初始化数组，用于记录每次攻击后最大连通子图的比例
    recorded_pc = False  # 标志位，用于记录pc值

    for i in range(number_of_nodes):
        if G.number_of_nodes() == 0:
            break  # 如果所有节点都被移除，则退出循环
        largest_cc = len(max(nx.connected_components(G), key=len))  # 计算最大连通子图的大小
        largest_all[i] = largest_cc / number_of_nodes  # 记录最大连通子图的比例

        if attack_type == 'degree':
            remove_nodes_by_degree(G)  # 移除当前网络度最高的一个节点
        elif attack_type == 'betweenness':
            remove_nodes_by_betweenness(G)  # 移除当前网络介数最高的一个节点，这里是迭代攻击的核心代码

        if not recorded_pc and calculate_degree_second_moment(G) / calculate_average_degree(G) <= 2:
            pc = (i + 1) / number_of_nodes  # 记录渗流阈值
            recorded_pc = True  # 更新标志位

    if not recorded_pc:
        pc = 1.0  # 如果没有记录到pc值，则默认pc值为1.0

    return largest_all, pc  # 返回每次攻击后最大连通子图的比例数组和pc值
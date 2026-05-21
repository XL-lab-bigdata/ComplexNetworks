import networkx as nx
import numpy as np
import random
from random import choice
from tqdm import tqdm
import pandas as pd
from scipy.sparse import lil_matrix
# 局部连边（Local Attachment）机制
def local_attachment(G, W, i, theta, p3):
    w0 = 1  # 初始权重
    i_neighbors = list(G.neighbors(i))  # 节点 i 的邻居节点列表
    if i_neighbors:  # i 不是孤立节点
        A = lil_matrix(W)
        wi = W[i]
        sum_wi = wi.sum()  # 节点 i 所有边的权重之和
        rnd_point = random.uniform(0, sum_wi)
        accumulator = 0.0
        for j, val in enumerate(wi):  # 选择邻居节点 j
            accumulator += val
            if accumulator >= rnd_point:
                j_neighbors = list(G.neighbors(j))
                j_neighbors.remove(i) if i in j_neighbors else None
                if j_neighbors:
                    wj = W[j].copy()
                    w_mid = wj[i]
                    wj[i] = 0  # 临时删除 i 到 j 的权重
                    sum_wj = wj.sum()
                    rnd_point1 = random.uniform(0, sum_wj)
                    accumulator1 = 0.0
                    for k, val1 in enumerate(wj):
                        accumulator1 += val1
                        if accumulator1 >= rnd_point1:
                            if not G.has_edge(i, k):
                                if random.random() < p3:
                                    G.add_edge(i, k, weight=w0)
                                    W[i, k] = W[k, i] = w0
                            else:
                                W[i, k] += theta
                                W[k, i] += theta
                            W[i, j] += theta
                            W[j, i] += theta
                            W[j, k] += theta
                            W[k, j] += theta
                            break
                break
    return G, W

# 全局连边（Global Attachment）机制
def global_attachment(G, W, i, pr):
    w0 = 1  # 初始权重
    i_neighbors = set(G.neighbors(i))
    potential_nodes = set(G.nodes) - i_neighbors - {i}

    if potential_nodes and random.random() < pr:
        l = choice(list(potential_nodes))
        G.add_edge(i, l, weight=w0)
        W[i, l] = W[l, i] = w0
    return G, W

# 节点删除（Node Deletion）机制
def node_deletion(G, W, i, pd):
    if random.random() < pd:
        neighbors = list(G.neighbors(i))
        G.remove_node(i)
        W[i, :] = W[:, i] = 0
        G.add_node(i)  # 重新加入节点 i
    return G, W

# KOSKK模型生成函数
def koskk(n, theta, pr=0.0005, pd=0.001, p3=0.15, T=100000):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    W = np.zeros((n, n))  # 初始化权重矩阵

    for _ in tqdm(range(T)):
        i = choice(list(G.nodes))
        G, W = global_attachment(G, W, i, pr)
        G, W = local_attachment(G, W, i, theta, p3)

        # 只对非孤立节点进行删除
        non_isolates = [node for node in G.nodes if G.degree(node) > 0]
        if non_isolates:
            i1 = choice(non_isolates)
            G, W = node_deletion(G, W, i1, pd)
    return G, W

# 生成KOSKK模型并保存数据
def generate_koskk(n, theta_list):
    for theta in tqdm(theta_list):
        G, W = koskk(n, theta)
        # 保存网络和权重矩阵
        nx.write_gpickle(G, f"KOSKK_network_theta_{theta:.2f}.gpickle")
        np.save(f"KOSKK_weights_theta_{theta:.2f}.npy", W)

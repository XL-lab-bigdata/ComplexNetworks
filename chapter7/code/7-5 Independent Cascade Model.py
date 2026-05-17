import networkx as nx
import random
import matplotlib.pyplot as plt

# 创建无向ER随机网络
def create_er_network(n, p):
    return nx.erdos_renyi_graph(n, p)

# 独立级联模型
def independent_cascade_model(G, initial_nodes, infection_prob):
    active_nodes = set(initial_nodes)  # 活跃节点集合
    newly_active_nodes = set(initial_nodes)  # 新增活跃节点集合
    all_active_nodes = set(initial_nodes)  # 记录所有活跃节点

    while newly_active_nodes:
        current_newly_active_nodes = set()
        for node in newly_active_nodes:
            # 遍历活跃节点的所有静默邻居
            for neighbor in G.neighbors(node):
                if neighbor not in all_active_nodes:  # 只考虑静默邻居
                    # 按照感染概率决定是否感染
                    if random.random() < infection_prob:
                        current_newly_active_nodes.add(neighbor)
                        all_active_nodes.add(neighbor)

        newly_active_nodes = current_newly_active_nodes  # 更新新增活跃节点集合

    return all_active_nodes

# 参数设置
n = 100  # 网络中节点数量
p = 0.1  # 边的生成概率
initial_nodes = [0]  # 初始活跃节点
infection_prob = 0.2  # 感染概率

# 创建网络并运行模型
G = create_er_network(n, p)
active_nodes = independent_cascade_model(G, initial_nodes, infection_prob)

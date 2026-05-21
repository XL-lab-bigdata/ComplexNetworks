import networkx as nx
# 生成 DHN 网络的函数
def generate_dhn(m, k, num_layers, total_nodes):
    G = nx.DiGraph()  # 创建一个有向图
    # 初始化包含 m 个节点的模块，并将节点编号为 1 到 m（节点 0 保留为中心节点）
    initial_module = list(range(1, m + 1))
    # 将初始模块中的节点添加到图中，并连接到中心节点 0
    G.add_nodes_from(initial_module)
    G.add_edges_from((0, node) for node in initial_module)
    # 当前层节点和大小初始化
    current_layer_nodes = initial_module[:]
    current_layer_size = m
    # 逐层生成网络
    for layer in range(1, num_layers + 1):
        next_layer_nodes = []  # 存储下一层的节点
        # 为当前层的每个节点生成 k 个副本
        for base_node in current_layer_nodes:
            for copy_id in range(k):
                # 生成新节点并加入图中
                new_nodes = [base_node + copy_id * current_layer_size + i for i in range(1, m + 1)]
                G.add_nodes_from(new_nodes)
                next_layer_nodes.extend(new_nodes)
                # 新节点连接到基础节点
                G.add_edges_from((base_node, new_node) for new_node in new_nodes)
        # 更新当前层的节点和大小
        current_layer_nodes = next_layer_nodes
        current_layer_size *= k  # 当前层节点数随每层增长
    # 截取图中指定数量的节点
    return G.subgraph(range(total_nodes)).copy()
# DHN 网络参数设置
m_dhn = 5  # 初始模块大小
k_dhn = 3  # 每层节点的副本数量
num_layers_dhn = 4  # 层数
n_dhn = 1000  # 总节点数
# 生成 DHN 网络
dhn_network = generate_dhn(m_dhn, k_dhn, num_layers_dhn, n_dhn)

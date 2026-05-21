def natural_connectivity(G):
    # 计算图 G 的邻接矩阵的特征值
    eig_value = np.linalg.eigvals(nx.to_numpy_matrix(G))
    # 计算特征值的指数
    exp_eig = np.exp(eig_value)
    # 计算特征值的指数和除以图的节点数量，然后取对数
    nature_connection = np.log(sum(exp_eig) / nx.number_of_nodes(G))
    return nature_connection

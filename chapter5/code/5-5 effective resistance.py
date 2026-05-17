def calculate_effective_resistance(G):
    # 计算网络的拉普拉斯矩阵
    L = nx.laplacian_matrix(G).toarray()
    # 计算拉普拉斯矩阵的特征值
    eigenvalues = np.linalg.eigvalsh(L)
    # 去除第一个零特征值
    nonzero_eigenvalues = eigenvalues[1:]
    # 计算有效电阻总和
    effective_resistance = sum(1 / eigenvalues for eigenvalues in nonzero_eigenvalues)/G.number_of_nodes()
    return effective_resistance

def calculate_spanning_trees(G):
    # 计算网络的拉普拉斯矩阵
    L = nx.laplacian_matrix(G).toarray()
    # 移除第一行和第一列，得到L的导出矩阵
    L_minor = np.delete(np.delete(L, 0, axis=0), 0, axis=1)
    # 计算拉普拉斯矩阵的导出矩阵的行列式
    determinant = np.linalg.det(L_minor)
    # 生成树数量即为行列式的值
    num_spanning_trees = int(round(determinant))
    return num_spanning_trees

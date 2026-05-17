import numpy as np
import scipy.sparse as sp

def calc_auc(train, test, sim, n):
    """
    计算AUC，输入计算的相似度矩阵
    """
    sim_dense = sim.toarray() if sp.issparse(sim) else sim
    train_dense = train.toarray() if sp.issparse(train) else train

    # sim_dense = np.triu(sim_dense - sim_dense * train_dense)
    sim_dense = np.triu(sim_dense.astype(int) - sim_dense.astype(int) * train_dense.astype(int))
    # 只保留测试集和不存在边集合中的边的相似度（自环除外）

    non = 1 - train_dense - test.toarray() - np.eye(max(train.shape))
    test_dense = test.toarray()
    test_dense = np.triu(test_dense)
    non = np.triu(non)
    # 分别取测试集和不存在边集合的上三角矩阵，用以取出他们对应的相似度分值

    test_num = np.count_nonzero(test_dense)
    non_num = np.count_nonzero(non)
    test_rd = np.ceil(test_num * np.random.rand(1, n)).astype(int) - 1
    # ceil是取大于等于的最小整数，n为抽样比较的次数
    non_rd = np.ceil(non_num * np.random.rand(1, n)).astype(int) - 1

    test_pre = sim_dense * test_dense
    non_pre = sim_dense * non

    test_data = test_pre[test_dense == 1].T
    # 行向量，test 集合存在的边的预测值
    non_data = non_pre[non == 1].T
    # 行向量，nonexist集合存在的边的预测值
    test_rd = test_data[test_rd]
    non_rd = non_data[non_rd]

    n1 = np.sum(test_rd > non_rd)
    n2 = np.sum(test_rd == non_rd)
    auc = (n1 + 0.5 * n2) / n

    return auc
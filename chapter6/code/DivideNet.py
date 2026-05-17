import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix

def divide_net(net, ratio_train):
    """
    划分训练集和测试集，保证训练集连通
    """
    num_test_links = int(np.ceil((1 - ratio_train) * np.count_nonzero(net) / 2))
    # 确定测试集的边数目
    x_index, y_index = np.nonzero(np.tril(net))
    linklist = np.vstack((x_index, y_index)).T
    # 将网络（邻接矩阵）中所有的边找出来，存入linklist

    # 使用 lil_matrix 来初始化和修改矩阵
    test = lil_matrix((net.shape[0], net.shape[1]))

    while test.nnz < num_test_links:

        # ---- 随机选择一条边
        index_link = np.random.randint(0, len(linklist))
        uid1, uid2 = linklist[index_link]

        # ---- 判断所选边两端节点uid1和uid2是否可达，若可达则可放入测试集，否则重新挑选一条边
        net[uid1, uid2] = 0
        net[uid2, uid1] = 0
        # 将这条边从网络中移除用以判断挖掉后的网络是否还连通

        temp_vector = net[uid1, :]
        sign = 0
        uid1_t0_uid2 = temp_vector.dot(net) + temp_vector
        # uid1TOuid2表示二步内可达的点

        if uid1_t0_uid2[uid2] > 0:
            sign = 1
            # 二步即可达
        else:
            while np.count_nonzero(np.sign(uid1_t0_uid2) - temp_vector) != 0:
                temp_vector = np.sign(uid1_t0_uid2)
                uid1_t0_uid2 = temp_vector.dot(net) + temp_vector
                # 此步的uid1TOuid2表示K步内可达的点

                if uid1_t0_uid2[uid2] > 0:
                    sign = 1
                    break

        # ---- 若此边可删除，则将之放入测试集中，并将此边从linklist中移除
        if sign == 1:
            linklist = np.delete(linklist, index_link, axis=0)
            test[uid1, uid2] = 1
        else:
            linklist = np.delete(linklist, index_link, axis=0)
            net[uid1, uid2] = 1
            net[uid2, uid1] = 1
        # 结束-判断此边是否可以删除并作相应处理

    # 结束（while）-测试集中的边选取完毕
    train = net
    test = sp.csr_matrix(test + test.T)
    # 返回训练集和测试集

    return sp.csr_matrix(train), test
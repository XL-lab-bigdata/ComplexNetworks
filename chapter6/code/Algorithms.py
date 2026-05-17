import numpy as np
import scipy.sparse as sp
import CalcAUC
from scipy.linalg import pinv
from scipy.linalg import inv

def cn(train, test):
    """
    计算CN指标并返回AUC值
    """
    # 相似度矩阵的计算
    sim = train.dot(train)  # 使用矩阵乘法计算相似度矩阵
    # 评测，计算该指标对应的AUC
    this_auc = CalcAUC.calc_auc(train, test, sim, 10000)
    return this_auc

def salton(train, test):
    """
    计算Salton指标并返回AUC值
    """
    # 转换为稠密矩阵进行计算
    if sp.issparse(train):
        train_dense = train.toarray()
    else:
        train_dense = train
    # 计算度的平方根
    temp_deg = np.sqrt(np.sum(train_dense, axis=1))  # 计算每行的和的平方根
    temp_deg = np.outer(temp_deg, temp_deg)  # 计算外积
    # 计算相似度矩阵
    sim = train_dense @ train_dense.T  # 计算分子
    temp_deg[temp_deg == 0] = 1  # 避免除以零
    sim = sim / temp_deg  # 计算相似度矩阵
    # 将 NaN 和 Inf 值设为 0
    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0
    # 计算AUC
    this_auc = CalcAUC.calc_auc(train, test, sp.csr_matrix(sim), 10000)  # 调用AUC计算函数
    return this_auc

def jaccard(train, test):
    """
    计算Jaccard指标并返回AUC值
    """
    # 转换为稠密矩阵进行计算
    if sp.issparse(train):
        train_dense = train.toarray()
    else:
        train_dense = train
    # 计算分子
    sim = train_dense @ train_dense.T
    # 计算度的和
    deg_row = np.sum(train_dense, axis=0)
    deg_row = np.outer(deg_row, np.ones(train_dense.shape[0]))
    # 只保留分子不为0对应的元素
    deg_row = np.triu(deg_row) + np.triu(deg_row.T)
    # 计算相似度矩阵
    sim = sim / (deg_row * (sim != 0) - sim)
    # 将 NaN 和 Inf 值设为 0
    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0
    # 计算AUC
    this_auc = CalcAUC.calc_auc(train, test, sp.csr_matrix(sim), 10000)  # 调用AUC计算函数
    return this_auc

def hpi(train, test):
    """
    计算HPI指标并返回AUC值
    """
    # 转换为稠密矩阵进行计算
    if sp.issparse(train):
        train_dense = train.toarray()
    else:
        train_dense = train
    # 计算分子
    sim = train_dense @ train_dense.T
    # 计算度的向量并构建矩阵
    deg_row = np.sum(train_dense, axis=0)
    deg_row = np.outer(deg_row, np.ones(train_dense.shape[0]))
    # 计算度的最小值矩阵
    deg_row = np.minimum(deg_row, deg_row.T)
    # 计算相似度矩阵
    sim = sim / (deg_row + np.eye(train_dense.shape[0]))  # 避免除以0的情况
    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0
    # 计算AUC
    this_auc = CalcAUC.calc_auc(train, test, sp.csr_matrix(sim), 10000)  # 调用AUC计算函数
    return this_auc

def hdi(train, test):
    """
    计算HDI指标并返回AUC值
    """
    # 转换为稠密矩阵进行计算
    if sp.issparse(train):
        train_dense = train.toarray()
    else:
        train_dense = train
    # 计算分子
    sim = train_dense @ train_dense.T
    # 计算度的向量并构建矩阵
    deg_row = np.sum(train_dense, axis=0)
    deg_row = np.outer(deg_row, np.ones(train_dense.shape[0]))
    # 计算度的最大值矩阵
    deg_row = np.maximum(deg_row, deg_row.T)
    # 计算相似度矩阵
    sim = sim / (deg_row + np.eye(train_dense.shape[0]))  # 避免除以0的情况
    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0
    # 计算AUC
    this_auc = CalcAUC.calc_auc(train, test, sp.csr_matrix(sim), 10000)  # 调用AUC计算函数
    return this_auc

def lhn(train, test):
    """
    计算LHN指标并返回AUC值
    """
    # 转换为稠密矩阵进行计算
    if sp.issparse(train):
        train_dense = train.toarray()
    else:
        train_dense = train
    # 计算分子
    sim = train_dense @ train_dense.T
    # 计算度的向量并构建矩阵
    deg = np.sum(train_dense, axis=1)
    deg = np.outer(deg, deg)  # 构建度的矩阵
    # 计算相似度矩阵
    sim = sim / (deg + np.eye(train_dense.shape[0]))  # 避免除以0的情况
    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0
    # 计算AUC
    this_auc = CalcAUC.calc_auc(train, test, sp.csr_matrix(sim), 10000)  # 调用AUC计算函数
    return this_auc

def aa(train, test):
    """
    计算AA指标并返回AUC值
    """
    train_sum = np.array(train.sum(axis=1)).flatten()
    # 替换0值以避免除以0的错误
    train_sum[train_sum == 0] = 1
    # 转换为密集矩阵来进行除法操作
    dense_train = train.toarray()
    # 计算每个节点的权重，1/log(k_i)
    train1 = dense_train / np.log(train_sum).reshape(-1, 1)
    # 将除数为0得到的异常值置为0
    train1[np.isnan(train1)] = 0
    train1[np.isinf(train1)] = 0
    # 转换回稀疏矩阵
    train1 = sp.csr_matrix(train1)
    # 实现相似度矩阵的计算
    sim = train.dot(train1)
    # 评测，计算该指标对应的AUC
    this_auc = CalcAUC.calc_auc(train, test, sim, 10000)
    return this_auc

def ra(train, test):
    """
    计算RA指标并返回AUC值
    """
    # 转换为稠密矩阵进行计算
    if sp.issparse(train):
        train_dense = train.toarray()
    else:
        train_dense = train
    # 计算每个节点的权重
    node_degrees = np.sum(train_dense, axis=1)
    node_degrees[node_degrees == 0] = 1  # 避免除以0
    train1 = train_dense / node_degrees[:, np.newaxis]
    # 处理无效值
    train1[np.isnan(train1)] = 0
    train1[np.isinf(train1)] = 0
    # 计算相似度矩阵
    sim = train_dense @ train1
    # 计算AUC
    this_auc = CalcAUC.calc_auc(train, test, sp.csr_matrix(sim), 10000)  # 调用AUC计算函数
    return this_auc

def pa(train, test):
    """
    计算PA指标并返回AUC值
    """
    # 转换为稠密矩阵进行计算
    if sp.issparse(train):
        train_dense = train.toarray()
    else:
        train_dense = train
    # 计算每个节点的度
    deg_row = np.sum(train_dense, axis=1)
    # 计算相似度矩阵
    sim = np.outer(deg_row, deg_row)
    # 计算AUC
    this_auc = CalcAUC.calc_auc(train, test, sp.csr_matrix(sim), 10000)  # 调用AUC计算函数
    return this_auc

def lp(train, test, lambda_param):
    """
    计算LP指标并返回AUC值
    """
    if sp.issparse(train):
        train_dense = train.toarray()
    else:
        train_dense = train
    # 计算二阶路径
    sim = train_dense @ train_dense
    # 计算三阶路径
    sim_three = train_dense @ sim
    # 加权二阶路径和三阶路径
    sim = sim + lambda_param * sim_three
    # 将计算结果转回稀疏矩阵格式
    sim_sparse = sp.csr_matrix(sim)
    # 计算AUC
    this_auc = CalcAUC.calc_auc(train, test, sim_sparse, 10000)  # 调用AUC计算函数
    return this_auc

def katz(train, test, lambda_param):
    """
    计算Katz指标并返回AUC值
    """
    # 将训练数据转换为稠密矩阵
    if sp.issparse(train):
        train_dense = train.toarray()
    else:
        train_dense = train
    # 构造单位矩阵
    I = np.eye(train_dense.shape[0])
    # 计算Katz相似度矩阵
    sim_dense = inv(I - lambda_param * train_dense)
    # 减去单位矩阵
    sim_dense = sim_dense - I
    # 转回稀疏矩阵格式
    sim_sparse = sp.csr_matrix(sim_dense)
    # 计算AUC
    this_auc = CalcAUC.calc_auc(train, test, sim_sparse, 10000)  # 调用AUC计算函数
    return this_auc

def act(train, test):
    """
    计算ACT指标并返回AUC值
    """
    # 生成稀疏的单位矩阵
    D = sp.eye(train.shape[0], format='csr')
    # 生成度矩阵（对角线元素为同下标节点的度）
    D.setdiag(np.array(train.sum(axis=1)).flatten())
    # 拉普拉斯矩阵的伪逆
    L = D - train
    pinvL = pinv(L.toarray())  # 注意这里转换为密集矩阵后计算伪逆
    # 取对角线元素
    lxx = np.diag(pinvL)
    # 将对角线元素向量扩展为n×n阶矩阵
    lxx = np.tile(lxx, (train.shape[0], 1))
    # 求相似度矩阵
    sim = 1.0 / (lxx + lxx.T - 2 * pinvL)
    # 处理无效值
    sim[np.isnan(sim)] = 0
    sim[np.isinf(sim)] = 0
    # 评测，计算该指标对应的AUC
    this_auc = CalcAUC.calc_auc(train, test, sim, 10000)
    return this_auc

def rwr(train, test, lambda_param):
    """
    计算RWR指标并返回AUC值
    """
    # 将训练数据转换为稠密矩阵
    if sp.issparse(train):
        train_dense = train.toarray()
    else:
        train_dense = train
    # 计算节点的度
    deg = np.sum(train_dense, axis=1, keepdims=True)
    # 计算转移矩阵
    train_normalized = train_dense / deg
    I = np.eye(train_dense.shape[0])
    # 计算相似度矩阵
    sim_dense = (1 - lambda_param) * inv(I - lambda_param * train_normalized.T) @ I
    sim_dense = sim_dense + sim_dense.T  # 对称化
    # 转回稀疏矩阵格式
    sim_sparse = sp.csr_matrix(sim_dense)
    # 还原邻接矩阵
    train_sparse = sp.csr_matrix(train_dense > 0)
    # 计算AUC
    thisauc = CalcAUC.calc_auc(train_sparse, test, sim_sparse, 10000)  # 调用AUC计算函数
    return thisauc


def lrw(train, test, steps, lambda_param):
    """
    计算LRW指标并返回AUC值
    """
    # 将训练数据转换为稠密矩阵
    if sp.issparse(train):
        train_dense = train.toarray()
    else:
        train_dense = train
    # 计算节点的度
    deg = np.sum(train_dense, axis=1, keepdims=True)
    # 计算转移矩阵
    train_normalized = train_dense / deg
    I = np.eye(train_dense.shape[0])
    # 初始化相似度矩阵
    sim = I
    # 随机游走的迭代
    for _ in range(steps):
        sim = (1 - lambda_param) * I + lambda_param * train_normalized.T @ sim
    # 对称化相似度矩阵
    sim = sim + sim.T
    # 转回稀疏矩阵格式
    sim_sparse = sp.csr_matrix(sim)
    # 还原邻接矩阵
    train_sparse = sp.csr_matrix(train_dense > 0)
    # 计算AUC
    thisauc = CalcAUC.calc_auc(train_sparse, test, sim_sparse, 10000)  # 调用AUC计算函数
    return thisauc

def srw(train, test, steps, lambda_param):
    """
    计算SRW指标并返回AUC值
    """
    # 将训练数据转换为稠密矩阵
    if sp.issparse(train):
        train_dense = train.toarray()
    else:
        train_dense = train
    # 计算节点的度
    deg = np.sum(train_dense, axis=1, keepdims=True)
    # 计算转移矩阵
    train_normalized = train_dense / deg
    I = np.eye(train_dense.shape[0])
    # 初始化相似度矩阵和临时矩阵
    tempsim = I
    sim = np.zeros_like(I)
    # 随机游走的迭代
    for _ in range(steps):
        tempsim = (1 - lambda_param) * I + lambda_param * train_normalized.T @ tempsim
        sim += tempsim
    # 对称化相似度矩阵
    sim = sim + sim.T
    # 转回稀疏矩阵格式
    sim_sparse = sp.csr_matrix(sim)
    # 还原邻接矩阵
    train_sparse = sp.csr_matrix(train_dense > 0)
    # 计算AUC
    thisauc = CalcAUC.calc_auc(train_sparse, test, sim_sparse, 10000)  # 调用AUC计算函数
    return thisauc
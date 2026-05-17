import os
import time
import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat
import DivideNet
import Algorithms

def main():
    # 参数设定
    ratio_train = 0.9  # 训练集比例
    num_of_experiment = 100  # 独立实验的次数

    # 用到的数据集名称
    data_name = ['USAir', 'Yeast', 'metabolic']
    data_path = 'C:/Users/Administrator/OneDrive/Python_Code/书稿示例/数据/'  # 数据集路径

    # 链路预测过程
    for ith_data, data in enumerate(data_name):
        # 遍历每一个数据
        print(f'正在处理第 {ith_data + 1} 个数据... {data}')
        start_time = time.time()

        this_data_path = os.path.join(data_path, f'{data}.mat')  # 第 ith 个数据的路径
        # net = loadmat(this_data_path)['G']  # 导入数据

        auc_of_all_predictor = []  # 用于存储100次实验的结果
        predictors_name = []  # 记录预测器的顺序

        # 开始100次实验的循环
        for ith_experiment in range(num_of_experiment):

            if ith_experiment % 10 == 0:
                print(f'{ith_experiment}%...')

            net = loadmat(this_data_path)['G']  # 导入数据

            # step-1 划分训练集和测试集，保证训练集的连通性
            train, test = DivideNet.divide_net(net, ratio_train)  # 划分训练集和测试集
            train = sp.csr_matrix(train)
            test = sp.csr_matrix(test)
            train = sp.csr_matrix((train + train.T).astype(bool))
            test = sp.csr_matrix((test + test.T).astype(bool))

            ith_auc_vector = []  # 用于存储当前实验中所有预测器的精度
            predictors = []  # 存储预测器名称

            # step-2 根据train set计算test set和nonexistent set中所有节点对产生（或存在）连边的可能性，并得出AUC
            # CN指标
            print('CN...')
            temp_auc = Algorithms.cn(train, test)  # Adar-Adamic Index
            predictors.append('CN')
            ith_auc_vector.append(temp_auc)
            # Salton指标
            print('Salton...')
            temp_auc = Algorithms.salton(train, test)  # Adar-Adamic Index
            predictors.append('Salton')
            ith_auc_vector.append(temp_auc)
            # Jaccard指标
            print('Jaccard...')
            temp_auc = Algorithms.jaccard(train, test)  # Adar-Adamic Index
            predictors.append('Jaccard')
            ith_auc_vector.append(temp_auc)
            # HPI指标
            print('HPI...')
            temp_auc = Algorithms.hpi(train, test)  # Adar-Adamic Index
            predictors.append('HPI')
            ith_auc_vector.append(temp_auc)
            # HDI指标
            print('HDI...')
            temp_auc = Algorithms.hdi(train, test)  # Adar-Adamic Index
            predictors.append('HDI')
            ith_auc_vector.append(temp_auc)
            # LHN指标
            print('LHN...')
            temp_auc = Algorithms.lhn(train, test)  # Adar-Adamic Index
            predictors.append('LHN')
            ith_auc_vector.append(temp_auc)
            # AA指标
            print('AA...')
            temp_auc = Algorithms.aa(train, test)  # Adar-Adamic Index
            predictors.append('AA')
            ith_auc_vector.append(temp_auc)
            # RA指标
            print('RA...')
            temp_auc = Algorithms.ra(train, test)  # Adar-Adamic Index
            predictors.append('RA')
            ith_auc_vector.append(temp_auc)
            # PA指标
            print('PA...')
            temp_auc = Algorithms.pa(train, test)  # Adar-Adamic Index
            predictors.append('PA')
            ith_auc_vector.append(temp_auc)
            # LP指标
            print('LP...')
            temp_auc = Algorithms.lp(train, test, 0.0001)  # Adar-Adamic Index
            predictors.append('LP')
            ith_auc_vector.append(temp_auc)
            # Katz指标
            print('Katz 0.01...')
            temp_auc = Algorithms.katz(train, test, 0.01)  # Adar-Adamic Index
            predictors.append('Katz')
            ith_auc_vector.append(temp_auc)
            # ACT指标
            print('ACT...')
            temp_auc = Algorithms.act(train, test)  # Adar-Adamic Index
            predictors.append('ACT')
            ith_auc_vector.append(temp_auc)
            # RWR指标
            print('RWR 0.85...')
            temp_auc = Algorithms.rwr(train, test, 0.85)  # Adar-Adamic Index
            predictors.append('RWR')
            ith_auc_vector.append(temp_auc)
            # LRW指标
            print('LRW 3 0.85...')
            temp_auc = Algorithms.lrw(train, test, 3, 0.85)  # Adar-Adamic Index
            predictors.append('LRW')
            ith_auc_vector.append(temp_auc)
            # SRW指标
            print('SRW 3 0.85...')
            temp_auc = Algorithms.srw(train, test, 3, 0.85)  # Adar-Adamic Index
            predictors.append('SRW')
            ith_auc_vector.append(temp_auc)

            auc_of_all_predictor.append(ith_auc_vector)
            predictors_name = predictors
        # numOfExperiment次独立循环结束
        # 写入当前数据集的结果
        avg_auc = np.mean(auc_of_all_predictor, axis=0)
        var_auc = np.var(auc_of_all_predictor, axis=0)
        res_path = os.path.join(data_path, 'result', f'{data}_res.txt')
        with open(res_path, 'w') as f:
            f.write('\t'.join(predictors_name) + '\n')
            np.savetxt(f, np.vstack([auc_of_all_predictor, avg_auc, var_auc]), fmt='%.4f', delimiter='\t')

        print(f'处理完成，耗时: {time.time() - start_time:.2f} 秒')

if __name__ == "__main__":
    main()
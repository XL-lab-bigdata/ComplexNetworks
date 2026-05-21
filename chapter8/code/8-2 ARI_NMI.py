import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, confusion_matrix
import numpy as np
# 创建图 (省略图的创建细节，假设图G已经创建)
#1. 真实社团的划分结果
true_communities = [{'v1', 'v2', 'v3', 'v4'}, {'v5', 'v6', 'v7'}]
#2. 检测算法得到的划分结果
predicted_communities = [{'v1', 'v2', 'v3'}, {'v4', 'v5', 'v6', 'v7'}]
#3. 创建节点到社团的映射并构建标签列表
node_to_community_true = {node: i for i, community in enumerate(true_communities) for node in community}
node_to_community_predicted = {node: i for i, community in enumerate(predicted_communities) for node in community}
nodes = list(G.nodes())
true_labels = [node_to_community_true[node] for node in nodes]
predicted_labels = [node_to_community_predicted[node] for node in nodes]
#4. 使用 sklearn 计算 ARI 和 NMI
ARI = adjusted_rand_score(true_labels, predicted_labels)
nmi_value = normalized_mutual_info_score(true_labels, predicted_labels)
print(f"ARI: {ARI}")
print(f"NMI: {nmi_value}")
#5. 使用 sklearn 中的 confusion_matrix 函数查看混淆矩阵
matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(matrix)

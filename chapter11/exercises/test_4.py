import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import urllib.request
import os

# --- 1. 数据加载与特征工程 (最终修正版) ---
def download_data():
    """下载所需的数据文件"""
    base_url = 'https://raw.githubusercontent.com/TobiasSkovgaardJepsen/posts/master/HowToDoDeepLearningOnGraphsWithGraphConvolutionalNetworks/Part2_SemiSupervisedLearningWithSpectralGraphConvolutions/'
    files = ['karate.edgelist', 'karate.attributes.csv']
    for file in files:
        if not os.path.exists(file):
            print(f"正在下载 {file}...")
            urllib.request.urlretrieve(base_url + file, file)
    print("数据文件已准备就绪。")

def load_and_prepare_data():
    """加载数据，创建两种特征，并返回两个PyG的Data对象"""
    # 使用networkx加载图结构
    G_nx = nx.read_edgelist('karate.edgelist', nodetype=int)
    
    # 使用pandas加载属性
    attributes_df = pd.read_csv('karate.attributes.csv', index_col=0)
    
    # 1. 创建标签 y
    y_labels = []
    for community_string in attributes_df['community']:
        if 'Administrator' in community_string:
            y_labels.append(0)
        else:
            y_labels.append(1)
    y = torch.tensor(y_labels, dtype=torch.long) # 直接创建最终的 y 张量

    # 2. 创建 edge_index
    # 仅使用 from_networkx 获取边信息
    edge_index = from_networkx(G_nx).edge_index
    
    # 3. 创建训练和测试掩码
    num_nodes = G_nx.number_of_nodes()
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    admin_node = attributes_df[attributes_df['role'] == 'Administrator'].index[0]
    instructor_node = attributes_df[attributes_df['role'] == 'Instructor'].index[0]
    
    train_mask[admin_node] = True
    train_mask[instructor_node] = True
    
    # 将所有非训练节点设置为测试节点
    test_mask[~(train_mask)] = True
            
    # --- 4. 组装数据对象 ---
    # 模型1: 单位矩阵特征
    x1 = torch.eye(num_nodes)
    data1 = Data(x=x1, edge_index=edge_index, y=y, 
                 train_mask=train_mask, test_mask=test_mask)

    # 模型2: 增强特征 (单位矩阵 + 距离特征)
    dist_to_admin = nx.shortest_path_length(G_nx, target=admin_node)
    dist_to_instructor = nx.shortest_path_length(G_nx, target=instructor_node)
    
    distance_features = torch.zeros((num_nodes, 2))
    for node in G_nx.nodes():
        distance_features[node][0] = dist_to_admin.get(node, 0)
        distance_features[node][1] = dist_to_instructor.get(node, 0)

    x2 = torch.cat([x1, distance_features], dim=1)
    data2 = Data(x=x2, edge_index=edge_index, y=y, 
                 train_mask=train_mask, test_mask=test_mask)

    print("数据处理与特征工程完成。")
    return data1, data2, G_nx

# --- 2. GCN模型定义 (PyG版本) ---
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# --- 3. 训练与评估函数 ---
def run_experiment(model, data, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(data)
        predictions = out.argmax(dim=1)
        test_correct = (predictions[data.test_mask] == data.y[data.test_mask]).sum()
        test_acc = int(test_correct) / int(data.test_mask.sum())
    
    print(f"训练完成！测试集准确率: {test_acc:.4f}")
    return predictions.cpu()

# --- 4. 可视化函数 ---
def visualize_graph(G, color_map, title):
    plt.figure(figsize=(8, 8))
    nx.draw_networkx(
        G,
        pos=nx.spring_layout(G, seed=42),
        node_color=color_map,
        cmap='viridis',
        with_labels=True,
        node_size=500
    )
    plt.title(title, fontsize=16)
    plt.show()

# --- 5. 主程序 ---
if __name__ == '__main__':
    download_data()
    data1, data2, G_nx = load_and_prepare_data()

    # --- 实验1：使用单位矩阵特征 ---
    print("\n=== 实验 1: 使用单位矩阵作为特征 ===")
    model1 = GCN(
        num_features=data1.num_node_features,
        num_hidden=16,
        num_classes=2
    )
    preds1 = run_experiment(model1, data1, epochs=300)
    visualize_graph(G_nx, preds1, "模型1预测结果 (单位矩阵特征)")

    # --- 实验2：使用增强特征 ---
    print("\n=== 实验 2: 使用单位矩阵 + 距离作为特征 ===")
    model2 = GCN(
        num_features=data2.num_node_features,
        num_hidden=16,
        num_classes=2
    )
    preds2 = run_experiment(model2, data2, epochs=300)
    visualize_graph(G_nx, preds2, "模型2预测结果 (增强特征)")
import torch
torchversion = torch.__version__

import numpy as np
import networkx as nx
from sklearn.manifold import TSNE # 用于高维数据的降维和可视化
import matplotlib.pyplot as plt

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

# 中文绘图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

dataset = Planetoid(root=".", name="CiteSeer")

data = dataset[0]

# 读取data.pt文件的内容
data2 = torch.load('./CiteSeer/processed/data.pt')
	
# 输出数据集的信息
print(f'Dataset: {dataset}')
print('-------------------')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# 输出图的信息
print(f'Graph:')
print('------')
print(f'Edges are directed: {data.is_directed()}')
print(f'Graph has isolated nodes: {data.has_isolated_nodes()}')
print(f'Graph has loops: {data.has_self_loops()}')

# # 将图数据转换为NetworkX图对象
# G = to_networkx(data, to_undirected=True)

# # 计算每个节点的度
# node_degrees = [degree for _, degree in G.degree()]

# # 绘制度分布图
# plt.figure(figsize=(10, 6))
# plt.hist(node_degrees, bins=np.arange(1, max(node_degrees)+1), color='skyblue', alpha=0.9)
# plt.xlabel('度', fontsize=14)
# plt.ylabel('频率', fontsize=14)
# plt.title('CiteSeer 图的度分布', fontsize=16)
# plt.yscale('log')  # 使用对数刻度显示频率
# plt.grid(True, linestyle='--', alpha=0.5)
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.show()

# # 绘制整个图，节点大小根据度的大小变化
# plt.figure(figsize=(15, 15))
# plt.axis('off')
# node_sizes = [degree * 10 for degree in node_degrees]  # 节点大小与其度成比例
# nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0), node_size=node_sizes, width=0.1, node_color='purple', alpha=0.5)
# plt.show()

# # 获取所有的连通子图
# connected_components = [G.subgraph(c).copy() for c in nx.connected_components(G)]

# # 设置绘图
# fig, axes = plt.subplots(len(connected_components), 1, figsize=(15, len(connected_components) * 3))
# if len(connected_components) == 1:
#     axes = [axes]  # 保证axes是一个列表

# # 绘制每个连通子图
# for i, subgraph in enumerate(connected_components):
#     axes[i].axis('off')
#     subgraph_degrees = [degree for _, degree in subgraph.degree()]
#     subgraph_node_sizes = [degree * 10 for degree in subgraph_degrees]  # 节点大小与其度成比例
#     nx.draw_networkx(subgraph, pos=nx.spring_layout(subgraph, seed=0), node_size=subgraph_node_sizes, width=0.1, node_color='purple', alpha=0.5, ax=axes[i])

# plt.tight_layout()
# plt.show()



# G = to_networkx(data, to_undirected=True)
# plt.figure(figsize=(18,18))
# plt.axis('off')
# nx.draw_networkx(G, pos=nx.spring_layout(G, seed=0),with_labels=False,
# node_size=50, node_color=data.y, width=2, edge_color="grey")
# plt.show()


import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATConv, GATv2Conv
class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=8):
    """继承自 torch.nn.Module 的类，代表了图注意力网络模型。构造函数中初始化了两个 GATv2Conv 图注意力层和一个 Adam 优化器，以及一些必要的超参数。"""
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.005,
                                      weight_decay=5e-4)
    
  def forward(self, x, edge_index):
    h = F.dropout(x, p=0.6, training=self.training)
    h = self.gat1(x, edge_index)
    h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index)
    return h, F.log_softmax(h, dim=1)
  

class StaticGAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, heads=8):
        super().__init__()
        self.gat1 = GATConv(dim_in, dim_h, heads=heads)
        self.gat2 = GATConv(dim_h*heads, dim_out, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=0.005,
                                          weight_decay=5e-4)
        
    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.6, training=self.training)
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.gat2(h, edge_index)
        return h, F.log_softmax(h, dim=1)


def accuracy(pred_y, y):
    """计算预测准确率的函数，根据预测值和真实值的比较来计算准确率。"""
    return ((pred_y == y).sum() / len(y)).item()
	
def train(model, data):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer
    epochs = 200
    model.train()
    for epoch in range(epochs+1):
    # Training
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    # Validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc*100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc*100:.2f}%')
          
    return model


def test(model, data):
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc


# 创建了一个名为gat的GAT模型的实例。传递了三个参数给GAT构造函数：输入特征的维度（dataset.num_features），隐藏层的维度为 8，以及类别数量（dataset.num_classes）
gat = GAT(dataset.num_features, 8, dataset.num_classes)
# gat = StaticGAT(dataset.num_features, 8, dataset.num_classes)
print(gat)
# 调用之前定义的 train 函数，对gat模型进行训练。这个函数将执行多个epoch的训练，每个epoch包括前向传播、计算损失、反向传播和参数更新，同时也会计算验证集上的性能指标
train(gat, data)
# 调用 test 函数，对经过训练的模型在测试集上进行性能评估。这个函数将返回测试集上的准确率。
acc = test(gat, data)
print(f'\nGAT test accuracy: {acc*100:.2f}%\n')

# 对经过训练的gat模型进行前向传播，得到节点的嵌入表示h。与之前的操作类似，这一步不进行训练，只是获取模型对输入数据的表示。
h, _ = gat(data.x, data.edge_index)
	
# 创建了一个 t-SNE 对象，将经过训练后的节点嵌入表示 h 降维到 2 维，以进行可视化。
tsne = TSNE(n_components=2, learning_rate='auto',
         init='pca').fit_transform(h.detach())
	
# 绘图
plt.figure(figsize=(10, 10))
plt.axis('off')
# 使用散点图绘制 t-SNE 降维后的节点表示。tsne[:, 0] 和 tsne[:, 1] 是 t-SNE 降维后的两个维度，s=50 表示散点的大小，c=data.y 表示根据节点的标签 data.y 对散点进行着色。
plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
plt.show()


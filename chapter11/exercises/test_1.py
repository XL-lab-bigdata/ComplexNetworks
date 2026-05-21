import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
# -----------------------------------------------------------------------------
# 使用 torch_geometric 中的 Planetoid 类直接加载 Cora 数据集
# root: 指定数据集存储的文件夹
# name: 指定数据集的名称
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0] # 获取图中唯一的图数据对象

# 打印数据集信息
print(f'数据集: {dataset.name}')
print('-------------------')
print(f'节点数量: {data.num_nodes}')
print(f'边的数量: {data.num_edges}')
print(f'节点特征维度: {dataset.num_node_features}')
print(f'类别数量: {dataset.num_classes}')
print(f'训练集节点数量: {data.train_mask.sum()}')
print(f'验证集节点数量: {data.val_mask.sum()}')
print(f'测试集节点数量: {data.test_mask.sum()}')


# 2. GCN 模型定义
# -----------------------------------------------------------------------------
# 经典的GCN模型通常包含两个图卷积层
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(GCN, self).__init__()
        # 第一个图卷积层：将节点特征从输入维度转换到隐藏维度
        self.conv1 = GCNConv(num_features, num_hidden)
        # 第二个图卷积层：将节点特征从隐藏维度转换到最终的类别维度
        self.conv2 = GCNConv(num_hidden, num_classes)

    def forward(self, x, edge_index):
        # x: 节点特征矩阵 [num_nodes, num_features]
        # edge_index: 边的连接信息 [2, num_edges]

        # 第一层卷积 + ReLU激活
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Dropout用于防止过拟合，在训练时随机丢弃一部分神经元
        x = F.dropout(x, p=0.5, training=self.training)
        
        # 第二层卷积
        x = self.conv2(x, edge_index)

        # 应用 log_softmax 得到对数概率，便于后续使用 NLLLoss 计算损失
        return F.log_softmax(x, dim=1)


# 3. 训练准备
# -----------------------------------------------------------------------------
# 检查是否有可用的GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化模型，并将其移动到指定设备
model = GCN(
    num_features=dataset.num_node_features,
    num_hidden=16, # GCN的隐藏层维度，16是该任务的常用值
    num_classes=dataset.num_classes
).to(device)

# 将数据也移动到指定设备
data = data.to(device)

# 定义优化器，Adam是常用的选择
# lr: 学习率
# weight_decay: L2正则化，防止过拟合
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 定义损失函数，负对数似然损失函数
criterion = torch.nn.NLLLoss()


# 4. 训练与评估函数
# -----------------------------------------------------------------------------
def train():
    """执行单次训练迭代"""
    model.train() # 将模型设置为训练模式
    optimizer.zero_grad() # 清除历史梯度
    
    # 执行前向传播
    out = model(data.x, data.edge_index)
    
    # 计算损失：只使用训练集节点的预测结果和真实标签
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # 反向传播和参数更新
    loss.backward()
    optimizer.step()
    
    return loss.item()

def test():
    """在测试集上评估模型性能"""
    model.eval() # 将模型设置为评估模式
    with torch.no_grad(): # 在评估时不计算梯度
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1) # 获取概率最大的类别作为预测结果
        
        # 计算测试集上的准确率
        correct = pred[data.test_mask] == data.y[data.test_mask]
        acc = int(correct.sum()) / int(data.test_mask.sum())
    return acc


# 5. 执行训练
# -----------------------------------------------------------------------------
epochs = 200
losses = []
accuracies = []

print("\n开始训练...")
for epoch in range(1, epochs + 1):
    loss = train()
    losses.append(loss)
    
    # 每10个epoch在测试集上评估一次模型
    if epoch % 10 == 0:
        test_acc = test()
        accuracies.append(test_acc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Test Accuracy: {test_acc:.4f}')

print("训练完成!")
final_accuracy = test()
print(f'\n最终测试集准确率: {final_accuracy:.4f}')

# 可视化损失和准确率变化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(losses)
ax1.set_title("Training Loss Curve")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.plot(range(10, epochs + 1, 10), accuracies)
ax2.set_title("Test Accuracy Curve")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
plt.show()
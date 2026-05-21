import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
# -----------------------------------------------------------------------------
# TUDataset is a collection of benchmark datasets for graph-level tasks.
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')

# 打印数据集信息
print(f'数据集: {dataset.name}')
print('-------------------')
print(f'图的数量: {len(dataset)}')
print(f'节点特征维度: {dataset.num_node_features}')
print(f'类别数量: {dataset.num_classes}')

# 由于TUDataset没有预设的训练/测试集划分，我们手动进行划分。
# 首先打乱数据集以保证随机性
torch.manual_seed(42) # 设置随机种子以保证结果可复现
dataset = dataset.shuffle()

# 划分训练集和测试集 (例如 80% 训练, 20% 测试)
train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

print(f'\n训练集大小: {len(train_dataset)}')
print(f'测试集大小: {len(test_dataset)}')

# 创建 DataLoader, 用于生成批次数据
# DataLoader 会将批次中的所有图合并成一个大的图对象，并创建一个 `batch` 向量
# 来指明每个节点属于哪个原始图。
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 2. GCN 模型定义 (用于图分类)
# -----------------------------------------------------------------------------
class GCN_Graph(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(GCN_Graph, self).__init__()
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_hidden)
        self.conv3 = GCNConv(num_hidden, num_hidden)
        
        # 用于最终分类的全连接层
        self.lin = Linear(num_hidden, num_classes)

    def forward(self, x, edge_index, batch):
        # x: 节点特征 [num_nodes, num_features]
        # edge_index: 边 [2, num_edges]
        # batch: 批次向量 [num_nodes]，指明每个节点属于哪个图
        
        # 1. 节点嵌入 (通过GCN层)
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()

        # 2. 全局池化层 (Readout)
        # global_mean_pool 会将同一图中所有节点的特征向量取平均值，
        # 从而为每个图生成一个单一的、固定大小的表示向量。
        x = global_mean_pool(x, batch) # [batch_size, num_hidden]
        
        # 3. 最终分类器
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

# 3. 训练准备
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN_Graph(
    num_features=dataset.num_node_features,
    num_hidden=64,
    num_classes=dataset.num_classes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 对于多分类问题，CrossEntropyLoss 是标准选择，它内部包含了Softmax。
criterion = torch.nn.CrossEntropyLoss()

# 4. 训练与评估函数
# -----------------------------------------------------------------------------
def train():
    """在整个训练集上训练一个epoch"""
    model.train()
    total_loss = 0
    for data_batch in train_loader: # 从 DataLoader 中迭代获取批次数据
        data_batch = data_batch.to(device)
        optimizer.zero_grad()
        # 前向传播，注意传入 batch 向量
        out = model(data_batch.x, data_batch.edge_index, data_batch.batch)
        # 计算损失，标签 data_batch.y 是图级别的标签
        loss = criterion(out, data_batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_batch.num_graphs
    return total_loss / len(train_loader.dataset)

def test(loader):
    """在指定的数据加载器上评估模型"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for data_batch in loader:
            data_batch = data_batch.to(device)
            out = model(data_batch.x, data_batch.edge_index, data_batch.batch)
            pred = out.argmax(dim=1) # 获取预测类别
            correct += int((pred == data_batch.y).sum())
    return correct / len(loader.dataset) # 返回准确率


# 5. 执行训练
# -----------------------------------------------------------------------------
epochs = 100
train_losses, test_accs, train_accs = [], [], []

print("\n开始训练...")
for epoch in range(1, epochs + 1):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    
    train_losses.append(loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

print("\n训练完成!")

# 可视化损失和准确率变化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(train_losses, label="Training Loss")
ax1.set_title("Training Loss Curve")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.plot(train_accs, label="Train Accuracy")
ax2.plot(test_accs, label="Test Accuracy")
ax2.set_title("Accuracy Curves")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()
plt.show()
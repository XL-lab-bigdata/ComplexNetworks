import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
# -----------------------------------------------------------------------------
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')

# 打印数据集信息
print(f'数据集: {dataset.name}')
print('-------------------')
print(f'图的数量: {len(dataset)}')
print(f'节点特征维度: {dataset.num_node_features}')
print(f'类别数量: {dataset.num_classes}')

# 手动划分训练集和测试集
torch.manual_seed(42) # 设置随机种子以保证结果可复现
dataset = dataset.shuffle()

train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

print(f'\n训练集大小: {len(train_dataset)}')
print(f'测试集大小: {len(test_dataset)}')

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 2. GIN 模型定义
# -----------------------------------------------------------------------------
# GINConv层使用一个MLP来更新节点表示，这里我们定义这个MLP
class GIN(torch.nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super(GIN, self).__init__()

        # 为第一个GIN层创建一个MLP
        nn1 = Sequential(
            Linear(num_features, num_hidden),
            ReLU(),
            Linear(num_hidden, num_hidden)
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm(num_hidden) # BatchNorm有助于稳定训练

        # 为第二个GIN层创建一个MLP
        nn2 = Sequential(
            Linear(num_hidden, num_hidden),
            ReLU(),
            Linear(num_hidden, num_hidden)
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm(num_hidden)

        # 为第三个GIN层创建一个MLP
        nn3 = Sequential(
            Linear(num_hidden, num_hidden),
            ReLU(),
            Linear(num_hidden, num_hidden)
        )
        self.conv3 = GINConv(nn3)
        self.bn3 = BatchNorm(num_hidden)
        
        # 用于最终分类的全连接层
        self.lin = Linear(num_hidden, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. 节点嵌入 (通过GIN层)
        x = self.conv1(x, edge_index)
        x = self.bn1(x).relu()
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x).relu()

        x = self.conv3(x, edge_index)
        x = self.bn3(x).relu()

        # 2. 全局池化层 (Readout)
        # global_add_pool 对同一图中的所有节点特征求和，是GIN论文中推荐的池化方式。
        x = global_add_pool(x, batch)
        
        # 3. 最终分类器
        x = self.lin(x)
        
        return F.log_softmax(x, dim=-1) # 使用log_softmax以配合NLLLoss

# 3. 训练准备
# -----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(
    num_features=dataset.num_node_features,
    num_hidden=64,  # 隐藏层维度
    num_classes=dataset.num_classes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.NLLLoss() # 配合log_softmax使用

# 4. 训练与评估函数 (与MUTAG问题中的结构相同)
# -----------------------------------------------------------------------------
def train():
    """训练一个epoch"""
    model.train()
    total_loss = 0
    for data_batch in train_loader:
        data_batch = data_batch.to(device)
        optimizer.zero_grad()
        out = model(data_batch.x, data_batch.edge_index, data_batch.batch)
        loss = criterion(out, data_batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_batch.num_graphs
    return total_loss / len(train_loader.dataset)

def test(loader):
    """评估模型"""
    model.eval()
    correct = 0
    with torch.no_grad():
        for data_batch in loader:
            data_batch = data_batch.to(device)
            out = model(data_batch.x, data_batch.edge_index, data_batch.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == data_batch.y).sum())
    return correct / len(loader.dataset)

# 5. 执行训练
# -----------------------------------------------------------------------------
epochs = 200
train_losses, test_accs, train_accs = [], [], []

print("\n开始使用GIN模型进行训练...")
for epoch in range(1, epochs + 1):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    
    train_losses.append(loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    # 每10个epoch打印一次结果
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

print("\n训练完成!")
final_train_acc = test(train_loader)
final_test_acc = test(test_loader)
print(f'最终训练集准确率: {final_train_acc:.4f}')
print(f'最终测试集准确率: {final_test_acc:.4f}')

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
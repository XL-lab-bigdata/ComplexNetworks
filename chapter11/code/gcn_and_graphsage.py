import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Load dataset
dataset = Planetoid(root=".", name="CiteSeer")
data = dataset[0]

# Define GCN model
class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.conv1 = GCNConv(dim_in, dim_h)
        self.conv2 = GCNConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

# Define GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.conv1 = SAGEConv(dim_in, dim_h)
        self.conv2 = SAGEConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.conv2(h, edge_index)
        return h, F.log_softmax(h, dim=1)

# Define accuracy function
def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()

# Define train function
def train(model, data):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer
    epochs = 200
    model.train()
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        _, out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
        if epoch % 10 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: '
                  f'{acc * 100:>6.2f}% | Val Loss: {val_loss:.2f} | '
                  f'Val Acc: {val_acc * 100:.2f}%')
    return model

# Define test function
def test(model, data):
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

# Train and test GCN model
gcn = GCN(dataset.num_features, 16, dataset.num_classes)
print(gcn)
train(gcn, data)
acc = test(gcn, data)
print(f'\nGCN test accuracy: {acc * 100:.2f}%\n')

# Visualize GCN embeddings
h, _ = gcn(data.x, data.edge_index)
tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(h.detach())
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
plt.title('GCN Embeddings')
plt.show()

# Train and test GraphSAGE model
graphsage = GraphSAGE(dataset.num_features, 16, dataset.num_classes)
print(graphsage)
train(graphsage, data)
acc = test(graphsage, data)
print(f'\nGraphSAGE test accuracy: {acc * 100:.2f}%\n')

# Visualize GraphSAGE embeddings
h, _ = graphsage(data.x, data.edge_index)
tsne = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(h.detach())
plt.figure(figsize=(10, 10))
plt.axis('off')
plt.scatter(tsne[:, 0], tsne[:, 1], s=50, c=data.y)
plt.title('GraphSAGE Embeddings')
plt.show()

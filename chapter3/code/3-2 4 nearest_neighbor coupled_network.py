# 导入NetworkX库
import networkx as nx
# 使用watts_strogatz_graph函数构建最近邻耦合网络
G = nx.watts_strogatz_graph(n=10, k=4, p=0)  #网络规模为10，每个节点与其四个最近邻节点相连，重连边概率为0

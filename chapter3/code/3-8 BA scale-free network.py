# 导入程序需要的库
import networkx as nx
# 创建网络规模为500，新增节点平均度为3，初始完全连通子图的节点数为3的BA无标度网络
G = nx.barabasi_albert_graph(N=500,m=3,m0=3)

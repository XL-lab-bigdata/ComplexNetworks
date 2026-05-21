# 导入程序需要的库
import networkx as nx
# 创建网络规模为500，新增节点平均度分别为2和4的BA 无标度网络
G1 = nx.barabasi_albert_graph(n = 500, m = 2, m0 = 2)
G2 = nx.barabasi_albert_graph(n = 500, m = 4, m0 = 4)
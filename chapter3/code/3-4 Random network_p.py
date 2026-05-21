# 导入程序需要的库
import networkx as nx
# 创建网络规模为10，固定连边概率为0.5的ER随机网络
G= nx.erdos_renyi_graph(n=10, p=0.5)

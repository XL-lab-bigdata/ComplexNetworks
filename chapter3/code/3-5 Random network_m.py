# 导入程序需要的库
import networkx as nx
# 节点数、边数和概率  
N=10
M=20
p=M*2/(N*(N-1))  #根据给定边数计算节点之间的连边概率
# 生成ER随机网络
G = nx.erdos_renyi_graph(N, p)

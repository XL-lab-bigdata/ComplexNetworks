# 导入程序需要的库
import networkx as nx
# 创建网络规模为1000，平均度为6，重连边概率为0.2的WS小世界网络
G = nx.watts_strogatz_graph(n=1000, k=6, p=0.2)

# 导入程序需要的库
import networkx as nx
#创建规模为1000，平均度为6，随机加边概率为0.2的NW小世界网络
G = nx.newman_watts_strogatz_graph(n=1000, k=6, p=0.2)

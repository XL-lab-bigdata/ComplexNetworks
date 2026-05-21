import matplotlib.pyplot as plt
import xgi

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体和解决中文乱码
plt.rcParams['axes.unicode_minus'] = False

# 内置数据集加载
H_enron = xgi.load_xgi_data("email-enron")
print(f"该超图包括{H_enron.num_nodes}个节点和{H_enron.num_edges}条边")

# ==================超图结果可视化==================
print(xgi.is_connected(H_enron))
H_enron_cleaned = H_enron.cleanup(multiedges=False, singletons=False, isolates=False, relabel=True, in_place=False)
print(xgi.is_connected(H_enron_cleaned))
fig, ax = plt.subplots(figsize=(8,10))  # 绘制图像
xgi.draw(H_enron_cleaned,node_size=H_enron_cleaned.nodes.degree,node_lw=H_enron_cleaned.nodes.average_neighbor_degree,node_fc=H_enron_cleaned.nodes.degree,ax=ax)
plt.savefig('H_enron_cleaned.svg', format='svg')
plt.show()

# ==================绘制超边度直方图==================
list_of_edges_sizes = H_enron_cleaned.edges.size.aslist()  # 获取边的大小列表
fig, ax = plt.subplots(figsize=(10, 6))  # 调整图形大小
ax.hist(
    list_of_edges_sizes,
    bins=range(min(list_of_edges_sizes), max(list_of_edges_sizes) + 1, 1),
    color='#e1bee7',  # 设置柱状图颜色
    edgecolor='black'  # 设置边框颜色
)
ax.set_xlabel("边的大小", fontsize=12)
ax.set_ylabel("频数", fontsize=12)
ax.set_title("Enron邮件超图网络边的大小分布", fontsize=14)
plt.tight_layout()
plt.savefig("enron_edges_histogram.svg", format='svg')
plt.show()

# ==================绘制节点度直方图==================
list_of_nodes_degrees = H_enron_cleaned.nodes.degree.aslist()  # 获取节点的度列表
fig, ax = plt.subplots(figsize=(10, 6))  # 调整图形大小
ax.hist(
    list_of_nodes_degrees,
    bins=range(min(list_of_nodes_degrees), max(list_of_nodes_degrees) + 1, 1),
    color='#9575cd',  # 设置柱状图颜色
    edgecolor='black'  # 设置边框颜色
)
ax.set_xlabel("度", fontsize=12)
ax.set_ylabel("频数", fontsize=12)
ax.set_title("Enron邮件超图网络节点度分布", fontsize=14)
plt.tight_layout()
plt.savefig("enron_nodes_degree_histogram.svg", format='svg')
plt.show()

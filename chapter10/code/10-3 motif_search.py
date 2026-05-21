import matplotlib.pyplot as plt
import seaborn as sns
from dotmotif import Motif, GrandIsoExecutor
from dotmotif.ingest import CSVEdgelistConverter

# 读取图数据
CSV_EDGELIST = "soma_subgraph_synapses_spines_v185.csv"
graph = CSVEdgelistConverter(
    CSV_EDGELIST,
    "pre_root_id",
    "post_root_id",
).to_graph()

# 创建搜索引擎
E = GrandIsoExecutor(graph=graph)

# 指定模体结构
motifs = [Motif("""A -> B \n B -> C \n C -> A"""),
    Motif("""A -> B  \n B -> C \n C -> B \n C -> A"""),
    Motif("""A -> B \n B -> C \n C -> B \n A -> C \n C -> A"""),
    Motif("""A -> B \n B -> A \n B -> C \n C -> B \n A -> C \n C -> A"""),
    Motif("""A -> B \n B -> C \n A -> C"""),
    Motif("""B -> A \n B -> C \n A -> C \n C -> A"""),
    Motif("""A -> B \n C -> B \n A -> C \n C -> A"""),
    Motif("""B -> A \n B -> C"""),
    Motif("""B -> A \n C -> B"""),
    Motif("""A -> B \n C -> B"""),
    Motif("""A -> B \n B -> A \n B -> C"""),
    Motif("""A -> B \n B -> A \n C -> B"""),
    Motif("""A -> B \n B -> A \n B -> C \n C -> B""")]

results = []
for motif in motifs:
    results.append(len(E.find(motif)))
print(results) # 打印每个模体结果的长度

# 绘制直方图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(14, 6))
plt.title("模体统计")
sns.barplot(x=[f'motif{i+1}' for i in range(len(results))], y=results)
plt.ylabel("出现次数")
plt.xlabel("模体类型")
plt.savefig('模体统计图.svg', format='svg')
plt.show()
from sklearn.metrics import auc #导入auc函数
R =  auc([x / len(G) for x in range(len(G) + 1)], largest_all)
# 使用sklearn.metrics模块中的AUC函数来计算鲁棒性曲线下的面积。其中，largest_all是一个记录逐个删除节点时网络相对最大连通片规模的列表，x / len(G)表示删除节点的比例。瓦解过程代码详见5.3节文件5-9 malicious.py。

import numpy as np

# 1) 读取边列表，忽略以“#”开头的注释行
rows, cols = [], []
with open('roadNet-CA.txt', 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        u, v = map(int, line.split())
        # SNAP 的节点从 0 开始；若从 1 开始，则需减 1
        rows.append(u)
        cols.append(v)
        # 因为是无向图，添加对称边
        rows.append(v)
        cols.append(u)

# 2) 构造 COO 格式稀疏矩阵
n = max(max(rows), max(cols)) + 1
data = np.ones(len(rows), dtype=int)

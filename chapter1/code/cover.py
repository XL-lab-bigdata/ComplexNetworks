import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# =========================
# 0. 参数区
# =========================
NODE_FILE = "nodes.csv"   # grid_id,x,y,visitors
EDGE_FILE = "edges.csv"   # source,target,weight
OUTPUT_FILE = "cover.png"

MIN_WEIGHT = 2        # 只绘制流量 >= 500 的边
MAX_EDGES_TO_DRAW = None  # 可设为 30000 之类；None 表示全画
EDGE_ALPHA = 0.5         # 连边透明度
EDGE_WIDTH_SCALE = 0.8   # 连边宽度缩放
NODE_ALPHA = 0.95
DPI = 300

# 视角参数
ELEV = 36                 # 仰角
AZIM = -63                # 方位角

# 高度映射参数
HEIGHT_SCALE = 130       # 节点高度总体缩放
ARC_HEIGHT_SCALE = 0   # 连边拱高缩放
ARC_HEIGHT_POWER = 0  # 距离对拱高的影响

# 点大小映射参数
NODE_SIZE_MIN = 1
NODE_SIZE_MAX = 10

# 连边离散段数，越大渐变越平滑，但越慢
ARC_SEGMENTS = 8

# 背景与配色
BACKGROUND = "#000000"
BASE_GRID_COLOR = (0.30, 0.55, 0.95, 0.10)   # 底部细网格线颜色
LOW_COLOR  = "#3a0ca3"   # 深紫
MID_COLOR  = "#ff9e00"   # 橙
HIGH_COLOR = "#ff0054"   # 红
'''
LOW_COLOR  = "#3a0ca3"   # 深紫
MID_COLOR  = "#ff9e00"   # 橙
HIGH_COLOR = "#ff0054"   # 红

LOW_COLOR  = "#14213d"
MID_COLOR  = "#00bbf9"
HIGH_COLOR = "#ffd60a"
'''

# =========================
# 1. 工具函数
# =========================
def build_cmap():
    return LinearSegmentedColormap.from_list(
        "cyanwhtpink",
        [LOW_COLOR, MID_COLOR, HIGH_COLOR],
        N=256
    )

def rescale_series(values, out_min, out_max, method="sqrt"):
    values = np.asarray(values, dtype=float)
    if method == "log":
        t = np.log1p(values)
    elif method == "sqrt":
        t = np.sqrt(np.clip(values, 0, None))
    else:
        t = values.copy()

    vmin, vmax = np.min(t), np.max(t)
    if vmax == vmin:
        return np.full_like(t, (out_min + out_max) / 2.0, dtype=float)
    t = (t - vmin) / (vmax - vmin)
    return out_min + t * (out_max - out_min)

def make_arc_points(x1, y1, z1, x2, y2, z2, n=20, arc_height=10.0):
    """
    构造 3D 拱形曲线。
    使用参数曲线：
    x(t), y(t) 线性插值
    z(t) = 线性底座 + 抛物线拱起
    """
    t = np.linspace(0, 1, n)
    x = (1 - t) * x1 + t * x2
    y = (1 - t) * y1 + t * y2
    z_base = (1 - t) * z1 + t * z2
    z_arc = 4 * arc_height * t * (1 - t)  # 中间最高
    z = z_base + z_arc
    return np.column_stack([x, y, z])

def make_gradient_segments(points):
    """
    将折线点序列转为 Line3DCollection 所需 segments
    """
    return np.stack([points[:-1], points[1:]], axis=1)

def blend_colors(c1, c2, n):
    """
    在两个 RGBA 颜色之间线性插值，返回 n-1 段的颜色
    """
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    t = np.linspace(0, 1, n - 1)[:, None]
    return (1 - t) * c1 + t * c2

def draw_base_grid(ax, xmin, xmax, ymin, ymax, step=5, z=0):
    """
    画底部平面细网格，增强参考图那种“空间骨架”感
    """
    xs = np.arange(math.floor(xmin), math.ceil(xmax) + 1, step)
    ys = np.arange(math.floor(ymin), math.ceil(ymax) + 1, step)

    for xv in xs:
        ax.plot([xv, xv], [ymin, ymax], [z, z], color=BASE_GRID_COLOR, linewidth=0.35)
    for yv in ys:
        ax.plot([xmin, xmax], [yv, yv], [z, z], color=BASE_GRID_COLOR, linewidth=0.35)


# =========================
# 2. 读取数据
# =========================
nodes = pd.read_csv(NODE_FILE)
edges = pd.read_csv(EDGE_FILE)

# 兼容你前文里 visiters / visitors 的可能拼写
if "visitors" not in nodes.columns:
    if "visiters" in nodes.columns:
        nodes = nodes.rename(columns={"visiters": "visitors"})
    else:
        raise ValueError("节点表中未找到 'visitors' 或 'visiters' 列。")

required_node_cols = {"grid_id", "x", "y", "visitors"}
required_edge_cols = {"source", "target", "weight"}

if not required_node_cols.issubset(nodes.columns):
    raise ValueError(f"节点表缺少字段：{required_node_cols - set(nodes.columns)}")
if not required_edge_cols.issubset(edges.columns):
    raise ValueError(f"边表缺少字段：{required_edge_cols - set(edges.columns)}")

nodes = nodes.copy()
edges = edges.copy()

# 基本清洗
nodes = nodes.dropna(subset=["grid_id", "x", "y", "visitors"])
edges = edges.dropna(subset=["source", "target", "weight"])

nodes["grid_id"] = nodes["grid_id"].astype(str)
edges["source"] = edges["source"].astype(str)
edges["target"] = edges["target"].astype(str)

nodes["x"] = pd.to_numeric(nodes["x"], errors="coerce")
nodes["y"] = pd.to_numeric(nodes["y"], errors="coerce")
nodes["visitors"] = pd.to_numeric(nodes["visitors"], errors="coerce")
edges["weight"] = pd.to_numeric(edges["weight"], errors="coerce")

nodes = nodes.dropna(subset=["x", "y", "visitors"])
edges = edges.dropna(subset=["weight"])

# 只保留正访问量节点
nodes = nodes[nodes["visitors"] > 0].copy()

# 过滤边
edges = edges[edges["weight"] >= MIN_WEIGHT].copy()

# 如果需要限制绘图边数，优先保留高权重边
if MAX_EDGES_TO_DRAW is not None and len(edges) > MAX_EDGES_TO_DRAW:
    edges = edges.nlargest(MAX_EDGES_TO_DRAW, "weight").copy()


# =========================
# 3. 节点映射：高度 / 大小 / 颜色
# =========================
cmap = build_cmap()
norm_vis = Normalize(vmin=nodes["visitors"].min(), vmax=nodes["visitors"].max())

def exp_scale(values, out_min, out_max, alpha=0.8):
    """
    指数缩放（稳定版本）
    alpha 越大，对比越强（建议 3~6）
    """
    values = np.asarray(values, dtype=float)
    vmin, vmax = np.min(values), np.max(values)

    if vmax == vmin:
        return np.full_like(values, (out_min + out_max) / 2.0)

    # 归一化
    t = (values - vmin) / (vmax - vmin)

    # 指数增强（归一化版本）
    t_exp = (np.exp(alpha * t) - 1) / (np.exp(alpha) - 1)

    return out_min + t_exp * (out_max - out_min)
import numpy as np

def saturating_scale(values, out_min, out_max, k=30, p=2.5):
    """
    饱和型缩放：
    - 小值较低
    - 中值快速抬升
    - 高值逐渐平台化
    k: 半饱和点，越小越早接近最高
    p: 曲线陡峭程度
    """
    values = np.asarray(values, dtype=float)
    values = np.clip(values, 0, None)

    t = (values ** p) / (values ** p + k ** p)
    return out_min + t * (out_max - out_min)
'''
nodes["z"] = saturating_scale(
    nodes["visitors"].values,
    out_min=1.5,
    out_max=10,
    k=300,      # 关键参数：越小越早“封顶”
    p=2      # 关键参数：越大过渡越陡
)
'''
# 使用指数缩放
nodes["z"] = exp_scale(
    nodes["visitors"].values,
    out_min=1.5,
    out_max=HEIGHT_SCALE,
    alpha=2  # ⭐ 核心参数
)
'''
# 高度：建议对数缩放，更稳定
#nodes["z"] = rescale_series(nodes["visitors"].values, 2.0, HEIGHT_SCALE, method="log")
'''
# 点大小：建议平方根缩放
nodes["size"] = rescale_series(nodes["visitors"].values, NODE_SIZE_MIN, NODE_SIZE_MAX, method="sqrt")

# 颜色
node_rgba = cmap(norm_vis(nodes["visitors"].values))
node_rgba[:, 3] = NODE_ALPHA
nodes["rgba"] = list(node_rgba)

# 建立节点索引
node_dict = nodes.set_index("grid_id")[["x", "y", "z", "visitors", "rgba"]].to_dict("index")

# 去掉 source/target 不在节点表中的边
edges = edges[edges["source"].isin(node_dict.keys()) & edges["target"].isin(node_dict.keys())].copy()

# 线宽
edges["line_width"] = rescale_series(edges["weight"].values, 0.15, 2.6, method="sqrt")


# =========================
# 4. 开始绘图
# =========================
fig = plt.figure(figsize=(14, 10))
fig.patch.set_alpha(0)
ax = fig.add_subplot(111, projection="3d")
fig.patch.set_facecolor(BACKGROUND)
ax.set_facecolor((0, 0, 0, 0))

# 去掉 pane
for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
    try:
        axis.pane.set_facecolor((0, 0, 0, 0))
        axis.pane.set_edgecolor((0, 0, 0, 0))
    except Exception:
        pass

# 关闭坐标轴
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_xlabel("")
ax.set_ylabel("")
ax.set_zlabel("")
ax.grid(False)
ax.set_axis_off()
# 先画底部细网格
xmin, xmax = nodes["x"].min(), nodes["x"].max()
ymin, ymax = nodes["y"].min(), nodes["y"].max()


# =========================
# 5. 画边：渐变拱线
# =========================
for row in edges.itertuples(index=False):
    s = node_dict[row.source]
    t = node_dict[row.target]

    x1, y1, z1 = s["x"], s["y"], s["z"]
    x2, y2, z2 = t["x"], t["y"], t["z"]

    dist = math.hypot(x2 - x1, y2 - y1)
    arc_h = ARC_HEIGHT_SCALE * (dist ** ARC_HEIGHT_POWER)

    pts = make_arc_points(
        x1, y1, z1,
        x2, y2, z2,
        n=ARC_SEGMENTS,
        arc_height=arc_h
    )
    segs = make_gradient_segments(pts)

    c1 = np.array(s["rgba"], dtype=float).copy()
    c2 = np.array(t["rgba"], dtype=float).copy()
    c1[3] = EDGE_ALPHA
    c2[3] = EDGE_ALPHA
    seg_colors = blend_colors(c1, c2, ARC_SEGMENTS)

    lc = Line3DCollection(
        segs,
        colors=seg_colors,
        linewidths=row.line_width,
        linestyles="solid"
    )
    ax.add_collection3d(lc)

# =========================
# 6. 画节点：先画浅色光晕，再画主体点
# =========================
# 光晕层
ax.scatter(
    nodes["x"], nodes["y"], nodes["z"],
    s=nodes["size"] * 2.8,
    c=nodes["rgba"].tolist(),
    alpha=0.06,
    linewidths=0,
    depthshade=False
)

# 主体层
ax.scatter(
    nodes["x"], nodes["y"], nodes["z"],
    s=nodes["size"],
    c=nodes["rgba"].tolist(),
    alpha=NODE_ALPHA,
    linewidths=0,
    depthshade=False
)

# 少量白色高亮点，增强“闪烁”感
top_nodes = nodes.nlargest(max(10, len(nodes)//120), "visitors")
ax.scatter(
    top_nodes["x"], top_nodes["y"], top_nodes["z"] + 0.15,
    s=np.clip(top_nodes["size"] * 0.18, 4, 20),
    c="white",
    alpha=0.9,
    linewidths=0,
    depthshade=False
)

# =========================
# 7. 视图与范围
# =========================
ax.view_init(elev=ELEV, azim=AZIM)

zmax = nodes["z"].max()
xrange = xmax - xmin
yrange = ymax - ymin

margin_ratio = 0.02  # ⭐ 可调（0~0.05）

ax.set_xlim(xmin - xrange * margin_ratio, xmax + xrange * margin_ratio)
ax.set_ylim(ymin - yrange * margin_ratio, ymax + yrange * margin_ratio)
ax.set_zlim(0, zmax * 1.05)
ax.set_position([0, 0, 1, 1])
# 让 x/y/z 比例更像“地面+峰值”
try:
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax * 0.65))
except Exception:
    pass
plt.rcParams['svg.fonttype'] = 'none'
plt.tight_layout()
plt.savefig(
    OUTPUT_FILE,
    dpi=DPI,
    transparent=True,
    edgecolor="none",
    bbox_inches="tight",
    pad_inches=0
)
plt.close(fig)

print(f"图片已导出到: {os.path.abspath(OUTPUT_FILE)}")
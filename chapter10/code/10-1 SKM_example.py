import matplotlib.pyplot as plt
import numpy as np
import xgi
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码
plt.rcParams['axes.unicode_minus'] = False

# 创建一个包含两个2-单纯形和一个1-单纯形的单纯复形
S = xgi.SimplicialComplex([[1, 2, 3], [2, 3, 4], [4, 5]])
print(S.edges.members())
xgi.draw(S)
plt.show()

orientations = {idd: 0 for idd in list(S.edges.filterby("order", 1, mode="geq"))} # 定义单纯形的参考方向
order = 1 # 振荡单纯形的阶数（如0表示节点，1表示边，2表示面等）
n = len(S.edges.filterby("order", order)) # 振荡单纯形的数量
omega = np.random.rand(n, 1) # 振荡器的固有频率
theta0 = 2 * np.pi * np.random.rand(n, 1) # 振荡器的初始相位
sigma = 0.4 # 振荡单纯形之间相互作用的强度
T = 30 # 模拟的时间范围
n_steps = 5000 # 积分步数

# 计算单纯阶参数
(theta,theta_minus,theta_plus,om1_dict,o_dict,op1_dict,) = xgi.synchronization.simulate_simplicial_kuramoto(S, orientations, order, omega, sigma, theta0, T, n_steps, True)
r = xgi.synchronization.compute_simplicial_order_parameter(theta_minus, theta_plus)

# 可视化
fig, axs = plt.subplots(2, 1)
fig.set_figheight(7)
fig.set_figwidth(8)

labels_list = ["[%s]" % ", ".join(map(str, list(S.edges.members()[idx])))
    for idx in list(o_dict.values())]

axs[0].plot(np.linspace(0, T, n_steps), np.sin(np.transpose(theta)))
axs[0].set_title("单纯形的相位演化动态")
axs[0].legend(labels_list, loc='upper left', bbox_to_anchor=(1, 1))

axs[1].plot(np.linspace(0, T, n_steps), r)
axs[1].set_title("单纯阶参数")
axs[1].set_ylim((0, 1))

# 调整布局以确保图例完全可见
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()
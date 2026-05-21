import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 设置字体为楷体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.rcParams['font.size'] = 14  # 设置全局字体大小
plt.rcParams['axes.labelsize'] = 22  # 设置坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 22  # 设置X轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 22  # 设置Y轴刻度字体大小


# SIS传播模型模拟
def sis_simulation(G, beta, mu, num_steps, immune_nodes=None):
    if immune_nodes is None:
        immune_nodes = []
    infected_nodes = set(np.random.choice(G.nodes(), int(0.1 * len(G)), replace=False))
    infected_fraction = []

    for _ in range(num_steps):
        new_infected = set()
        for node in G.nodes():
            if node in immune_nodes:
                continue
            if node in infected_nodes:
                if np.random.rand() < mu:
                    new_infected.discard(node)
                else:
                    new_infected.add(node)
            else:
                for neighbor in G.neighbors(node):
                    if neighbor in infected_nodes and np.random.rand() < beta:
                        new_infected.add(node)
                        break
        infected_nodes = new_infected
        infected_fraction.append(len(infected_nodes) / len(G))

    return infected_fraction


# 随机免疫
def random_immunization(G, immunization_fraction):
    num_immune = int(immunization_fraction * len(G))
    return np.random.choice(G.nodes(), num_immune, replace=False)


# 目标免疫
def target_immunization(G, immunization_fraction):
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    num_immune = int(immunization_fraction * len(G))
    return sorted_nodes[:num_immune]


# 参数设置
beta = 0.1  # 感染率
mu = 0.2  # 恢复率
num_steps = 80
immunization_fraction = 0.1
num_simulations = 10

# 网络设置
N = 1000
k = 10
p = 0.1
m = 5

# 创建网络
ws = nx.watts_strogatz_graph(N, k, p)
ba = nx.barabasi_albert_graph(N, m)

# 模拟不同策略和网络
networks = [ws, ba]
network_names = ['WS', 'BA']
immunization_strategies = [random_immunization, target_immunization]
strategy_names = ['Random', 'Target']

results = {}
for network, network_name in zip(networks, network_names):
    for strategy, strategy_name in zip(immunization_strategies, strategy_names):
        all_simulations = []
        for _ in range(num_simulations):
            immune_nodes = strategy(network, immunization_fraction)
            infected_fraction = sis_simulation(network, beta, mu, num_steps, immune_nodes)
            all_simulations.append(infected_fraction)
        avg_infected_fraction = np.mean(all_simulations, axis=0)
        results[(network_name, strategy_name)] = avg_infected_fraction

# 创建两个子图
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：WS网络
markers = ['^', 's']
for i, strategy_name in enumerate(strategy_names):
    axes[0].plot(results[('WS', strategy_name)], marker=markers[i], linestyle='', label=f'随机免疫' if strategy_name == 'Random' else f'目标免疫')
axes[0].set_xlabel('时间', fontsize=22)
axes[0].set_ylabel('感染比例', fontsize=22)
#axes[0].set_title('WS网络中的免疫策略', fontsize=18)
axes[0].grid(False)
axes[0].legend(fontsize=22, loc='lower right')
axes[0].text(-0.1, 1.05, '(A)', transform=axes[0].transAxes, fontsize=22, fontweight='bold', va='top', ha='right')

# 右图：BA网络
for i, strategy_name in enumerate(strategy_names):
    axes[1].plot(results[('BA', strategy_name)], marker=markers[i], linestyle='', label=f'随机免疫' if strategy_name == 'Random' else f'目标免疫')
axes[1].set_xlabel('时间', fontsize=22)
axes[1].set_ylabel('感染比例', fontsize=22)
#axes[1].set_title('BA网络中的免疫策略', fontsize=18)
axes[1].grid(False)
axes[1].legend(fontsize=22, loc='lower right')
axes[1].text(-0.1, 1.05, '(B)', transform=axes[1].transAxes, fontsize=22, fontweight='bold', va='top', ha='right')

plt.tight_layout()
plt.show()
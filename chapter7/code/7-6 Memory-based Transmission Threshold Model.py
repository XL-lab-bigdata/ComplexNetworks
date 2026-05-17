import numpy as np

# 状态常量
SUSCEPTIBLE = 0  # 易感态
INFECTED = 1     # 传播态
NEUTRAL = 2      # 中立态

class Node:
    def __init__(self, state, threshold):
        self.state = state                  # 当前状态
        self.threshold = threshold          # 剂量阈值
        self.doses = []                    # 过去的剂量记录

    def update_state(self):
        # 更新状态逻辑
        if self.state == SUSCEPTIBLE:
            cumulative_dose = sum(self.doses)
            if cumulative_dose >= self.threshold:
                self.state = INFECTED
        elif self.state == INFECTED:
            cumulative_dose = sum(self.doses)
            if cumulative_dose < self.threshold:
                self.state = NEUTRAL
        elif self.state == NEUTRAL:
            if np.random.rand() < 0.1:  # 以10%概率恢复为易感者
                self.state = SUSCEPTIBLE

def simulate(nodes, steps, dose_distribution, threshold_distribution):
    for step in range(steps):
        for node in nodes:
            # 节点相互接触
            for neighbor in nodes:
                if node != neighbor:
                    if node.state == SUSCEPTIBLE and neighbor.state == INFECTED:
                        # 以一定概率受到影响
                        dose = np.random.choice(dose_distribution)
                        node.doses.append(dose)

            # 更新节点状态
            node.update_state()

        # 可选：输出每一步的状态
        print(f"Step {step + 1}: {[node.state for node in nodes]}")

# 初始化参数
num_nodes = 10
steps = 5
dose_distribution = [0.1, 0.2, 0.3, 0.4, 0.5]  # 剂量分布
threshold_distribution = np.random.uniform(0.5, 1.0, num_nodes)  # 随机阈值分布

# 创建节点
nodes = [Node(SUSCEPTIBLE, threshold_distribution[i]) for i in range(num_nodes)]
# 随机选择初始感染者
nodes[np.random.randint(0, num_nodes)].state = INFECTED

# 运行模拟
simulate(nodes, steps, dose_distribution, threshold_distribution)

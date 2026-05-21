import random
import numpy as np

class PeerToPeerDiffusion:
    def __init__(self, network, diffusion_prob, reaction_time_distribution):
        """
        网络信息传播模型
        network: dict, 邻接表表示的社交网络
        diffusion_prob: float, 扩散概率
        reaction_time_distribution: list, 节点的反应时间分布
        """
        self.network = network
        self.diffusion_prob = diffusion_prob
        self.reaction_time_distribution = reaction_time_distribution

    def select_initial_publisher(self):
        return random.choice(list(self.network.keys()))  # 随机选择发布者

    def should_forward(self):
        return random.random() < self.diffusion_prob  # 根据扩散概率判断是否转发

    def get_reaction_time(self):
        return np.random.choice(self.reaction_time_distribution)  # 根据反应时间分布获取反应时间

    def run_diffusion(self):
        # 初始化
        time_axis = []  # 存储转发时刻
        active_nodes = set()  # 记录活跃节点
        publisher = self.select_initial_publisher()
        active_nodes.add(publisher)

        print(f"初始发布者: {publisher}")

        while True:
            current_time = len(time_axis)  # 当前时间（按转发事件数模拟）
            # 判断消息接收者是否转发
            for neighbor in self.network[publisher]:
                if neighbor not in active_nodes and self.should_forward():
                    active_nodes.add(neighbor)
                    reaction_time = self.get_reaction_time()
                    time_axis.append((neighbor, current_time + reaction_time))  # 记录转发时刻

            # 如果没有新的转发时刻，结束扩散
            if not time_axis:
                print("扩散过程结束。")
                break

            # 按时间排序获取下一个转发事件
            time_axis.sort(key=lambda x: x[1])
            next_publisher, next_time = time_axis.pop(0)
            publisher = next_publisher
            print(f"新的发布者: {publisher} 在时刻: {next_time}")


# -------------------------------
# 示例网络和参数
# -------------------------------
network_example = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'G'],
    'F': ['C'],
    'G': ['E']
}

diffusion_probability = 0.4  # 扩散概率
reaction_time_distribution = [1, 2, 3]  # 反应时间分布

# 创建扩散模型实例并运行
model = PeerToPeerDiffusion(network_example, diffusion_probability, reaction_time_distribution)
model.run_diffusion()
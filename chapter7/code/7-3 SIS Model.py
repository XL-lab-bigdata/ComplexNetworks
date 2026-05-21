import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# -------------------------------
# 定义SIS模型微分方程
# -------------------------------
def SIS_model(y, t, beta, mu):
    S, I = y
    dSdt = mu * I - beta * S * I
    dIdt = beta * S * I - mu * I
    return [dSdt, dIdt]

# -------------------------------
# 初始条件
# -------------------------------
S0 = 0.99  # 初始易感个体比例
I0 = 0.01  # 初始感染个体比例
y0 = [S0, I0]

# -------------------------------
# 参数设置
# -------------------------------
mu = 0.1            # 恢复率
t = np.linspace(0, 160, 160)  # 时间点
beta_values = [0.05, 0.3]     # 两组感染率

# -------------------------------
# 创建子图
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, beta in enumerate(beta_values):
    solution = odeint(SIS_model, y0, t, args=(beta, mu))
    S, I = solution[:, 0], solution[:, 1]

    axes[idx].plot(t, S, label='易感态 (S)', color='blue')
    axes[idx].plot(t, I, label='感染态 (I)', color='red')
    axes[idx].set_title(f'SIS模型 (β={beta})', fontsize=18)
    axes[idx].set_xlabel('时间', fontsize=14)
    axes[idx].set_ylabel('比例', fontsize=14)
    axes[idx].legend(fontsize=12)
    axes[idx].grid(True)

# -------------------------------
# 调整布局并显示
# -------------------------------
plt.tight_layout()
plt.show()
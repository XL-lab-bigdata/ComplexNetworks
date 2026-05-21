import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# -------------------------------
# 定义SI模型微分方程
# -------------------------------
def si_model(y, t, beta):
    S, I = y
    dSdt = -beta * S * I
    dIdt = beta * S * I
    return [dSdt, dIdt]

# -------------------------------
# 初始条件
# -------------------------------
S0 = 0.99  # 初始易感个体比例
I0 = 0.01  # 初始感染个体比例
y0 = [S0, I0]

# -------------------------------
# 时间点
# -------------------------------
t = np.linspace(0, 160, 160)

# -------------------------------
# 创建两个子图
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# -------------------------------
# 第一个SI模型，beta=0.05
# -------------------------------
beta_1 = 0.05
solution_1 = odeint(si_model, y0, t, args=(beta_1,))
S1, I1 = solution_1[:, 0], solution_1[:, 1]

axes[0].plot(t, S1, label='易感态(S)', color='blue')
axes[0].plot(t, I1, label='感染态(I)', color='red')
axes[0].set_title(f'SI模型 (β={beta_1})', fontsize=18)
axes[0].set_xlabel('时间', fontsize=14)
axes[0].set_ylabel('比例', fontsize=14)
axes[0].legend(fontsize=12)
axes[0].grid(True)

# -------------------------------
# 第二个SI模型，beta=0.3
# -------------------------------
beta_2 = 0.3
solution_2 = odeint(si_model, y0, t, args=(beta_2,))
S2, I2 = solution_2[:, 0], solution_2[:, 1]

axes[1].plot(t, S2, label='易感态(S)', color='blue')
axes[1].plot(t, I2, label='感染态(I)', color='red')
axes[1].set_title(f'SI模型 (β={beta_2})', fontsize=18)
axes[1].set_xlabel('时间', fontsize=14)
axes[1].set_ylabel('比例', fontsize=14)
axes[1].legend(fontsize=12)
axes[1].grid(True)

# -------------------------------
# 自动调整布局并显示
# -------------------------------
plt.tight_layout()
plt.show()
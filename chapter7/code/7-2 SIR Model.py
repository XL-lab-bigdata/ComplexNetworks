import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import odeint 
from matplotlib import font_manager 
# 定义SIR模型的微分方程 
def sir_model(y, t, beta, mu): 
    S, I, R = y 
    dSdt = -beta * S * I 
    dIdt = beta * S * I - mu * I 
    dRdt = mu * I 
    return [dSdt, dIdt, dRdt] 
# 初始条件 
S0 = 0.99  # 初始易感个体比例 
I0 = 0.01  # 初始感染个体比例 
R0 = 0.0   # 初始恢复个体比例 
y0 = [S0, I0, R0] 
# 参数：感染率beta和恢复率mu 
mu = 0.1
# 时间点 
t = np.linspace(0, 160, 160) 
# 创建两个子图 
fig, axes = plt.subplots(1, 2, figsize=(16, 6)) 
beta_1 = 0.05 
solution_1 = odeint(sir_model, y0, t, args=(beta_1, mu)) 
S1, I1, R1 = solution_1[:, 0], solution_1[:, 1], solution_1 [:, 2] 
axes[0].plot(t, S1, label='易感态(S)', color='blue') 
axes[0].plot(t, I1, label='感染态(I)', color='red') 
axes[0].plot(t, R1, label='恢复态(R)', color='green') 
#axes[0].set_title(f'SIR 模型 (beta={beta_1})', fontsize=18) 
axes[0].set_xlabel('时间', fontsize=22) 
axes[0].set_ylabel('比例', fontsize=22) 
axes[0].grid(False) 
# 求解和绘制第二个SIR模型，beta=0.3 
beta_2 = 0.3 
solution_2 = odeint(sir_model, y0, t, args=(beta_2, mu)) 
S2, I2, R2 = solution_2[:, 0], solution_2[:, 1], solution_2 [:, 2] 
axes[1].plot(t, S2, label='易感态(S)', color='blue') 
axes[1].plot(t, I2, label='感染态(I)', color='red') 
axes[1].plot(t, R2, label='恢复态(R)', color='green') 
#axes[1].set_title(f'SIR 模型 (beta={beta_2})', fontsize=18) 
axes[1].set_xlabel('时间', fontsize=22) 
axes[1].set_ylabel('比例', fontsize=22) 
plt.show() 
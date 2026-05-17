import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 定义SIS模型的微分方程
def SIS_model(y, t, beta, delta):
    S, I = y
    dSdt = delta * I - beta * S * I
    dIdt = beta * S * I - mu * I
    return [dSdt, dIdt]

# 参数设置
N = 1.0  # 总人口归一化
I0 = 0.01  # 初始感染比例
S0 = N - I0  # 初始易感比例
y0 = [S0, I0]  # 初始状态

# 时间点
t = np.linspace(0, 160, 160)

# SIS模型参数
beta = 0.3  # 传播率
mu = 0.1  # 恢复率

# 求解微分方程
solution = odeint(SIS_model, y0, t, args=(beta, mu))
S, I = solution.T

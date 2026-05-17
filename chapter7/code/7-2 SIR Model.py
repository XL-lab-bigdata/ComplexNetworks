import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# 定义 SIR 模型的微分方程组
def SIR_model(y, t, beta, mu):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - mu * I
    dRdt = mu * I
    return [dSdt, dIdt, dRdt]

# 初始条件和参数
N = 1  # 总人口归一化
I0 = 0.01  # 初始感染比例
S0 = 1 - I0  # 初始易感比例
R0 = 0  # 初始恢复比例
y0 = [S0, I0, R0]  # 初始状态

# 时间点
t = np.linspace(0, 160, 160)

# 设置lambda 值 (beta/mu)
lambdas = 1  # 这里lambda取1

# 设置不同 lambda 值下的 beta 和 mu
beta = lambdas * 0.1  # 假设 mu = 0.1
mu = 0.1  # 恢复率

# 求解微分方程
solution = odeint(SIR_model, y0, t, args=(beta, mu))
S, I, R = solution.T

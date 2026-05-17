import numpy as np
import matplotlib.pyplot as plt

# 参数设置
N = 1000  # 种群规模
beta = 0.3  # 感染率
I0 = 1  # 初始感染个体数
S0 = N - I0  # 初始易感个体数
time = np.linspace(0, 10, 100)  # 时间范围

# Logistic增长方程
def logistic_growth(t, N, I0, beta):
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    
    S[0] = S0
    I[0] = I0
    
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N
        
        S[i] = S[i-1] + dS
        I[i] = I[i-1] + dI

    return S, I

# 计算易感和感染个体的比例
S, I = logistic_growth(time, N, I0, beta)
S_ratio = S / N
I_ratio = I / N

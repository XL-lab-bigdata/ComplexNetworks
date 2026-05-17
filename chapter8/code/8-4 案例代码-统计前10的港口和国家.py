import pandas as pd
import matplotlib.pyplot as plt

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# 读取数据
ports_df = pd.read_csv('data.csv')

# 将 Hong Kong 归入 China
ports_df['Country Name, Full'] = ports_df['Country Name, Full'].replace('China, Hong Kong Special Administrative Region', 'China')

# Step 1: 找出按容量排序的前10个港口
top_ports = ports_df.nlargest(10, 'Capacity')[['port', 'Capacity']]

# Step 2: 按国家总容量进行汇总，找出前10个国家
top_countries = ports_df.groupby('Country Name, Full').sum().nlargest(10, 'Capacity').reset_index()

# 画两个并排的柱状图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# (A) 图：按容量排序的前10个港口，使用两种绿色
axes[0].barh(top_ports['port'], top_ports['Capacity'], color='#81c784')  # 淡绿
axes[0].set_title('Top 10 Ports by Capacity')
axes[0].set_xlabel('Capacity')
axes[0].set_ylabel('Port')
axes[0].text(0.5, -0.15, '(A)', transform=axes[0].transAxes, fontsize=12, va='top', ha='center')

# (B) 图：按国家总容量排序的前10个国家，使用深绿色
axes[1].barh(top_countries['Country Name, Full'], top_countries['Capacity'], color='#43a047')  # 深绿
axes[1].set_title('Top 10 Countries by Total Port Capacity')
axes[1].set_xlabel('Total Capacity')
axes[1].set_ylabel('Country')
axes[1].text(0.5, -0.15, '(B)', transform=axes[1].transAxes, fontsize=12, va='top', ha='center')

# 调整布局并显示
plt.tight_layout()
# 保存为 PDF，dpi=600
plt.savefig('前10港口和国家.pdf', format='pdf', dpi=600)
plt.show()
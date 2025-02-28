import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取Excel文件
df = pd.read_excel("strategies_b_1.6.xlsx")  # 根据你的文件名调整

# 假设节点编号从0到2499，生成一个50x50的网格
grid_size = 50
strategy_grid = np.zeros((grid_size, grid_size))

# 将策略填入网格
for index, row in df.iterrows():
    node = row['Node']
    strategy = row['Strategy']
    x = node // grid_size
    y = node % grid_size
    strategy_grid[x, y] = strategy

# 定义颜色映射：深蓝色表示合作（0），白色表示背叛（1）
cmap = plt.cm.colors.ListedColormap(['darkblue', 'white'])

# 绘制策略分布图
plt.figure(figsize=(6, 6))
plt.imshow(strategy_grid, cmap=cmap, interpolation='nearest')
plt.xticks([])  # 隐藏x坐标刻度
plt.yticks([])  # 隐藏y坐标刻度
plt.gca().spines['top'].set_visible(True)  # 显示顶部框
plt.gca().spines['right'].set_visible(True)  # 显示右侧框
plt.gca().spines['left'].set_visible(True)  # 显示左侧框
plt.gca().spines['bottom'].set_visible(True)  # 显示底部框

plt.tight_layout()  # 这个命令能让图形的布局更紧凑，减少留白
plt.show()


# Mean Cooperator Ratio in Stable State for b = 1: 0.897346

# Mean Cooperator Ratio in Stable State for b = 1.1: 0.841836

# Mean Cooperator Ratio in Stable State for b = 1.2: 0.803016

# Mean Cooperator Ratio in Stable State for b = 1.3: 0.756242

# Mean Cooperator Ratio in Stable State for b = 1.4: 0.724818

# Mean Cooperator Ratio in Stable State for b = 1.5: 0.664724

# Mean Cooperator Ratio in Stable State for b = 1.6: 0.567018

# Mean Cooperator Ratio in Stable State for b = 1.7: 0.24221

# Mean Cooperator Ratio in Stable State for b = 1.8: 0.166156

# Mean Cooperator Ratio in Stable State for b = 1.9: 0.142994

# Mean Cooperator Ratio in Stable State for b = 2.0: 0.074036


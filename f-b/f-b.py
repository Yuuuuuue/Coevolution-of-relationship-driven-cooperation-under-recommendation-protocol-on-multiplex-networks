import matplotlib.pyplot as plt
import numpy as np

c_heat = [
    [0.305652, 0.8445920000000001, 0.933964, 1.0],
    [0.204918, 0.71026, 0.8826900000000001, 1.0],
    [0.129144, 0.41922, 0.83894, 1.0],
    [0.11131600000000003, 0.28073400000000004, 0.7839680000000001,  0.962144],
    [0.066498, 0.23450200000000002, 0.748764, 0.9494999999999999],
    [0.026312000000000002, 0.12030799999999999,0.7280760000000001,0.8841239999999999],
    [0.025757999999999997, 0.123538, 0.715376, 0.868854],
    [0.018518, 0.056954000000000005, 0.658256, 0.823896],
    [0.005309999999999999, 0.05257799999999999, 0.379036, 0.8032739999999999],
    [0.010533999999999998, 0.032431999999999996, 0.24733400000000003, 0.7625319999999999],
    [0.0015180000000000005, 0.031132, 0.06950200000000001, 0.7358060000000001]
]

# x 的范围改为适合 b的数据
x = np.linspace(1, 2, len(c_heat))  # 将范围调整为 1 到 2
y = np.array(c_heat).T  # 转置为正确的形状

plt.figure(figsize=(2, 1.3), dpi=500)  # 调整图形大小
plt.plot(x, y[0], marker='o', markersize=3, linewidth=1.5, label="HL", color='green')
plt.plot(x, y[1], marker='o', markersize=3, linewidth=1.5, label="SL", color='orange')
plt.plot(x, y[2], marker='o', markersize=3, linewidth=1.5, label="XL", color='dodgerblue')
plt.plot(x, y[3], marker='o', markersize=3, linewidth=1.5, label="WS", color='red')

plt.legend(loc='upper right', bbox_to_anchor=(1, 0.7), frameon=False, fontsize=4, ncol=1)

plt.xlabel('b', fontsize=6, labelpad=0.1)
plt.ylabel('$f_c$', fontsize=6, rotation=0, labelpad=2.5)

plt.xticks(np.arange(1, 2.1, 0.2), size=4)
plt.yticks(np.arange(0, 1.1, 0.2), size=4)

plt.tick_params(axis='both', pad=1)  # 增加 pad 的值可调整距离

plt.grid(True)

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.17)  # 调整 bottom 的值

plt.show()

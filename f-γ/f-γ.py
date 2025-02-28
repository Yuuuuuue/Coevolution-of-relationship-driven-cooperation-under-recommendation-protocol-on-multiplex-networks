import matplotlib.pyplot as plt
import numpy as np

c_heat = [
    [0.1169344, 0.5263880000000001, 0.7537564, 0.9853376],
    [0.08723879999999999, 0.2571952, 0.7285788, 0.9023008000000001],
    [0.053102800000000006, 0.126806, 0.6815652, 0.8725252],
    [0.0333192, 0.0789788, 0.2908644, 0.8522607999999999],
    [0.021119199999999994, 0.0481404, 0.1338756, 0.8009524],
    [0.018084, 0.029570799999999998, 0.0843004, 0.7564708],
    [0.0125844, 0.020319599999999997, 0.0499276, 0.6871328],
    [0.0072575999999999995, 0.013394, 0.029264000000000002, 0.32664000000000004],
    [0.0056944, 0.00912, 0.0157336, 0.1462844],
    [0.0027304, 0.0041972, 0.005820399999999999, 0.0855052],
    [0.0009416000000000001, 0.0026456, 0.0040612, 0.0480032]
]

# x 的范围改为适合 gamma 的数据
x = np.linspace(0, 1, len(c_heat))  # 生成从 0 到 1 的等间距数据
y = np.array(c_heat).T  # 转置为正确的形状

plt.figure(figsize=(2, 1.3), dpi=500)  # 调整图形大小
plt.plot(x, y[0], marker='o', markersize=3, linewidth=1.5, label="HL", color='green')
plt.plot(x, y[1], marker='o', markersize=3, linewidth=1.5, label="SL", color='orange')
plt.plot(x, y[2], marker='o', markersize=3, linewidth=1.5, label="XL", color='dodgerblue')
plt.plot(x, y[3], marker='o', markersize=3, linewidth=1.5, label="WS", color='red')

plt.legend(loc='upper right', bbox_to_anchor=(1, 0.7), frameon=False, fontsize=4, ncol=1)

plt.xlabel('γ', fontsize=6, labelpad=0.1)
plt.ylabel('$f_c$', fontsize=6, rotation=0, labelpad=2.5)

plt.xticks(np.arange(0, 1.1, 0.2), size=4)
plt.yticks(np.arange(0, 1.1, 0.2), size=4)

plt.tick_params(axis='both', pad=1)  # 增加 pad 的值可调整距离

plt.grid(True)

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.17)  # 将 bottom 的值调小

plt.show()


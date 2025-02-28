import matplotlib.pyplot as plt
import numpy as np

c_heat = [
    [0.0363652, 0.08536000000000002, 0.22447200000000003, 0.8232584],
    [0.0412452, 0.072496, 0.45628799999999997, 0.8562416],
    [0.047108800000000006, 0.092182, 0.607162, 0.8703104],
    [0.048052000000000004, 0.08340800000000001, 0.667256, 0.8651439999999999],
    [0.06421439999999999, 0.12696199999999996, 0.7006699999999999, 0.8816767999999999],
    [0.0764268, 0.16634999999999997, 0.705986, 0.8832036],
    [0.084416,  0.171306, 0.71095, 0.8780324],
    [0.0720012, 0.207348, 0.7201159999999999, 0.885484],
    [0.0740556, 0.22796999999999998, 0.7294839999999999, 0.9375956],
    [0.0811804, 0.29380399999999996, 0.7426179999999999, 0.9364448],
    [0.071444, 0.291262, 0.7284999999999999, 0.9416524000000002]
]

# x 的范围改为适合 gamma 的数据
x = np.linspace(0.5, 1, len(c_heat))  # 生成从 0 到 1 的等间距数据
y = np.array(c_heat).T  # 转置为正确的形状

plt.figure(figsize=(2, 1.3), dpi=500)  # 调整图形大小
plt.plot(x, y[0], marker='o', markersize=3, linewidth=1.5, label="HL", color='green')
plt.plot(x, y[1], marker='o', markersize=3, linewidth=1.5, label="SL", color='orange')
plt.plot(x, y[2], marker='o', markersize=3, linewidth=1.5, label="XL", color='dodgerblue')
plt.plot(x, y[3], marker='o', markersize=3, linewidth=1.5, label="WS", color='red')

plt.legend(loc='upper right', bbox_to_anchor=(1, 0.7), frameon=False, fontsize=4, ncol=1)

plt.xlabel('p', fontsize=6, labelpad=0.1)
plt.ylabel('$f_c$', fontsize=6, rotation=0, labelpad=2.5)

plt.xticks(np.arange(0.5, 1.05, 0.1), size=4)
plt.yticks(np.arange(0, 1.1, 0.2), size=4)

plt.tick_params(axis='both', pad=1)  # 增加 pad 的值可调整距离

plt.grid(True)

plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.17)  # 将 bottom 的值调小

plt.show()


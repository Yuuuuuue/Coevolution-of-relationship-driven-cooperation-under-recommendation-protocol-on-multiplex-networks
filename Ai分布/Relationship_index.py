import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取 Excel 文件
# file_path = '3_relationship_index_results.xlsx'
# file_path = '4_relationship_index_results.xlsx'
# file_path = '6_relationship_index_results.xlsx'
file_path = 'WS_relationship_index_results.xlsx'
data = pd.read_excel(file_path, sheet_name='Sheet1')

# 提取四列数据
initial_relationship_index = data['Initial Relationship Index (m=0)'].dropna()
final_relationship_index_m0 = data['Final Relationship Index (m=0)'].dropna()
final_relationship_index_m05 = data['Final Relationship Index (m=0.5)'].dropna()
final_relationship_index_m1 = data['Final Relationship Index (m=1)'].dropna()

# 设置图形大小
plt.figure(figsize=(10, 6))

# 绘制每一列的分布图
sns.kdeplot(initial_relationship_index, fill=True, color='red', label='Initial Relationship Index', alpha=0.2, linestyle='--', clip=(0, None))
sns.kdeplot(final_relationship_index_m0, fill=True, color='blue', label='Final Relationship Index (m=0)', alpha=0.2, clip=(0, None))
sns.kdeplot(final_relationship_index_m05, fill=True, color='green', label='Final Relationship Index (m=0.5)', alpha=0.2, clip=(0, None))
sns.kdeplot(final_relationship_index_m1, fill=True, color='orange', label='Final Relationship Index (m=1)', alpha=0.2, clip=(0, None))

# 调整边距和布局
plt.subplots_adjust(left=0.11, right=0.98, bottom=0.12, top=0.98)
plt.grid(True, alpha=0.7)

# 设置图表标签
plt.xlabel('Relationship Index',fontsize=20)
plt.ylabel('Density',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=12)
plt.show()

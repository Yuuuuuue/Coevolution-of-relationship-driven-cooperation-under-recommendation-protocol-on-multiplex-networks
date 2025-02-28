import networkx as nx
import random
import numpy as np
import pandas as pd

# 固定参数设置
b = 1.5  # 背叛诱惑
p = 0.9
d = 0.05  # 权值变化
K = 0.1  # 噪声参数
gamma = 0.1  # 阈值

R = np.array([[1, 0], [b, 0]])  # 收益矩阵
m_values = [0, 0.5, 1]
mean_relationship_indices = []  # 用于存储每次实验的平均关系指数

initial_relationship_index = None  # 用于存储第一个 m 下的初态
all_final_relationship_indices = []  # 用于存储稳态关系指数

# 实验循环
for i, m in enumerate(m_values):
    print(f"Running experiment with m = {m}")
    edge_weights = {}  # 存储边的权值和对应的节点

    # 初始化网络
    G_coop = nx.random_graphs.random_regular_graph(3, 2500)  # 第一层：合作网络
    G_game = nx.Graph()  # 第二层：博弈网络
    G_game.add_nodes_from(G_coop.nodes())  # 添加与合作网络相同的节点

    # 添加合作网络的边及权值
    for (u, v) in G_coop.edges():
        edge_weights[(u, v)] = edge_weights[(v, u)] = np.random.uniform(0, 1)

    # 添加博弈网络的边
    for u in G_game.nodes():
        neighbors_u = list(G_coop.neighbors(u))
        if neighbors_u:
            x = max(neighbors_u, key=lambda neighbor: edge_weights.get((u, neighbor), 0))
            for v in neighbors_u:
                u_v_weight = edge_weights.get((u, v), 0)
                if G_coop.has_edge(u, v) and u_v_weight > gamma:
                    if random.random() < p:
                        G_game.add_edge(u, v)
                if v != x and u_v_weight > gamma and not G_coop.has_edge(x, v):
                    if random.random() < p:
                        G_game.add_edge(x, v)

    # 存储历史数据
    strategies = np.random.randint(2, size=G_coop.number_of_nodes())

    # 计算各节点合作指数
    def compute_relationship_index(node):
        neighbors = list(G_coop.neighbors(node))
        cooperation_values = []
        for neighbor in neighbors:
            cooperation_values.append(edge_weights[(node, neighbor)])
        relationship_index[node] = sum(cooperation_values)  # 当前节点在合作层的所有邻居节点的合作连边权值之和

    # 初始化关系指数
    relationship_index = np.zeros(G_coop.number_of_nodes())

    # 计算各节点的初始合作指数
    for node in G_coop.nodes():
        compute_relationship_index(node)

    # 保存第一个 m 的初态
    if i == 0:
        initial_relationship_index = relationship_index.copy()

    # 策略更新
    def update_strategies():
        for node in G_game.nodes():
            neighbors = list(G_coop.neighbors(node))
            if len(neighbors) > 0:
                neighbor_weights = np.array([edge_weights[(node, neighbor)] for neighbor in neighbors], dtype=np.float64)
                total_weight = np.sum(neighbor_weights)
                if total_weight > 0:
                    neighbor_weights /= total_weight  # 归一化权重
                else:
                    neighbor_weights = np.ones_like(neighbor_weights) / len(neighbors)  # 等概率选择

                selected_neighbor = np.random.choice(neighbors, p=neighbor_weights)

                Ax = relationship_index[node]
                Ay = relationship_index[selected_neighbor]
                P_x = R[strategies[node]][strategies[selected_neighbor]]
                P_y = R[strategies[selected_neighbor]][strategies[node]]
                Fx = m * P_x + Ax
                Fy = m * P_y + Ay
                prob = 1 / (1 + np.exp((Fx - Fy) / K))
                strategies[node] = np.random.choice([strategies[node], strategies[selected_neighbor]], p=[1 - prob, prob])

    # 更新合作层权值
    def update_network():
        for edge in G_game.edges():
            node1, node2 = edge
            # 双方都合作
            if strategies[node1] == 0 and strategies[node2] == 0:
                if G_coop.has_edge(node1, node2):
                    edge_weights[(node1, node2)] = min(edge_weights[(node1, node2)] + d, 1)  # 权值增加，但不超过1
            # 双方都背叛
            elif strategies[node1] == 1 and strategies[node2] == 1:
                if G_coop.has_edge(node1, node2):
                    edge_weights[(node1, node2)] = max(edge_weights[(node1, node2)] - d, 0)  # 权值减少，但不小于0

    all_relationship_indices = []  # 用于存储每一步的关系指数

    for step in range(1000):
        for node in G_coop.nodes():
            compute_relationship_index(node)
        update_strategies()
        update_network()

        # 记录每一步的关系指数
        all_relationship_indices.append(relationship_index.copy())

    # 计算每个运行的平均关系指数
    average_relationship_indices = np.mean(all_relationship_indices, axis=0)
    mean_average_relationship_index = np.mean(average_relationship_indices)
    print(f"Mean Relationship Index in Final State for m = {m}:", mean_average_relationship_index)

    # 记录结果
    mean_relationship_indices.append(mean_average_relationship_index)

    # 处理初始态和稳态的非负关系指数
    initial_relationship_index = np.clip(initial_relationship_index, 0, None)  # 限制初始态为非负
    average_relationship_indices = np.clip(average_relationship_indices, 0, None)  # 限制稳态为非负

    # 保存稳态关系指数
    all_final_relationship_indices.append(average_relationship_indices)

# 创建 DataFrame 并保存到 Excel
df_results = pd.DataFrame({
    'Node': range(G_coop.number_of_nodes()),
    'Initial Relationship Index (m=0)': initial_relationship_index,
    'Final Relationship Index (m=0)': all_final_relationship_indices[0],
    'Final Relationship Index (m=0.5)': all_final_relationship_indices[1],
    'Final Relationship Index (m=1)': all_final_relationship_indices[2]
})

# 保存结果到 Excel 文件
output_file = '3_relationship_index_results.xlsx'
df_results.to_excel(output_file, index=False)

print(f"Results saved to {output_file}")


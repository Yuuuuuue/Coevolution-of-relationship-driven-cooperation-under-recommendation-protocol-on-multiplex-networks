import networkx as nx
import random
import numpy as np

# 固定参数设置
b = 1  # 背叛诱惑
p = 0.9
d = 0.05  # 权值变化
K = 0.1  # 噪声参数
gamma = 0.1  # 阈值

R = np.array([[1, 0], [b, 0]])  # 收益矩阵
m_values = [0, 0.5, 1]
mean_relationship_indices = []  # 用于存储每次实验的平均关系指数

# 实验循环
for m in m_values:
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

    # 计算各节点关系指数
    relationship_index = np.zeros(G_coop.number_of_nodes())

    # 计算初始关系指数
    for node in G_coop.nodes():
        relationship_index[node] = sum(edge_weights[(node, neighbor)] for neighbor in G_coop.neighbors(node))

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
                strategies[node] = np.random.choice([strategies[node], strategies[selected_neighbor]],
                                                    p=[1 - prob, prob])

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

    num_runs = 1  # 设定独立运行次数
    all_relationship_indices = []

    for run in range(num_runs):
        print(f"开始第 {run + 1} 次独立运行")  # 打印当前运行次数
        strategies = np.random.randint(2, size=G_coop.number_of_nodes())
        edge_weights = {key: np.random.uniform(0, 1) for key in edge_weights}

        for step in range(1000):
            for node in G_coop.nodes():
                relationship_index[node] = sum(edge_weights[(node, neighbor)] for neighbor in G_coop.neighbors(node))
            update_strategies()
            update_network()

        # 记录最终状态下的关系指数
        all_relationship_indices.append(relationship_index.copy())

    # 计算每个运行的平均关系指数
    average_relationship_indices = np.mean(all_relationship_indices, axis=0)
    mean_average_relationship_index = np.mean(average_relationship_indices)
    print(f"Mean Relationship Index in Final State for m = {m}:", mean_average_relationship_index)

    # 记录结果
    mean_relationship_indices.append(mean_average_relationship_index)

# 打印最终结果
print("Mean Relationship Indices for each m value:", mean_relationship_indices)

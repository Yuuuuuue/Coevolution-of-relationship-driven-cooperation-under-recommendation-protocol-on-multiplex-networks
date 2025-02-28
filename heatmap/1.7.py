import networkx as nx
import random
import numpy as np

# 固定参数设置
b = 1.7  # 背叛诱惑
p = 0.9  # 第一层有边的条件下第二层连边概率
d = 0.05  # 权值变化
K = 0.1  # 噪声参数
gamma = 0.5  # 阈值

R = np.array([[1, 0], [b, 0]])  # 收益矩阵
m_values = np.arange(0, 1.05, 0.05)  # m的取值从0到1，每次增加0.05
mean_cooperator_ratios = []  # 用于存储每次实验的平稳状态下合作者比例均值

# 实验循环
for m in m_values:
    print(f"Running experiment with m = {m}")
    edge_weights = {}  # 存储边的权值和对应的节点

    # 初始化网络
    G_coop = nx.random_graphs.random_regular_graph(6, 2500)  # 第一层：合作网络
    G_game = nx.Graph()  # 第二层：博弈网络
    G_game.add_nodes_from(G_coop.nodes())  # 添加与合作网络相同的节点

    # 添加合作网络的边及权值
    for (u, v) in G_coop.edges():
        edge_weights[(v, u)] = edge_weights[(u, v)] = np.random.uniform(0, 1)

    # 添加博弈网络的边
    for u in G_game.nodes():
        neighbors_u = list(G_coop.neighbors(u))
        if neighbors_u:
            # 找到与u关系值最大的邻居x
            x = max(neighbors_u, key=lambda neighbor: edge_weights.get((u, neighbor), 0))
            for v in neighbors_u:
                u_v_weight = edge_weights.get((u, v), 0)
                # 如果u和v在关系层有直接连边，并且关系值大于γ，则以概率p添加边
                if G_coop.has_edge(u, v) and u_v_weight > gamma:
                    if random.random() < p:
                        G_game.add_edge(u, v)
                # 如果v不是x，且v与u的关系值大于γ，且x与v之间在关系层尚未添加边，则以概率p添加边
                if v != x and u_v_weight > gamma and not G_coop.has_edge(x, v):
                    if random.random() < p:
                        G_game.add_edge(x, v)

    # 存储合作指数、玩家类型和策略的数组
    cooperation_index = np.zeros(G_coop.number_of_nodes())
    strategies = np.random.randint(2, size=G_coop.number_of_nodes())

    # 存储历史数据
    cooperator_ratios = []

    # 计算各节点合作指数
    def compute_cooperation_index(node):
        neighbors = list(G_coop.neighbors(node))
        cooperation_values = []
        for neighbor in neighbors:
            cooperation_values.append(edge_weights[(node, neighbor)])
        cooperation_index[node] = sum(cooperation_values)  # 当前节点在合作层的所有邻居节点的合作连边权值之和

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

                Ax = cooperation_index[node]
                Ay = cooperation_index[selected_neighbor]
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
                    if edge_weights[(node1, node2)] + d <= 1:
                        edge_weights[(node1, node2)] += d  # 权值增加，但不超过1
                    else:
                        edge_weights[(node1, node2)] = 1
            # 双方都背叛
            elif strategies[node1] == 1 and strategies[node2] == 1:
                if G_coop.has_edge(node1, node2):
                    if edge_weights[(node1, node2)] - d > 0:
                        edge_weights[(node1, node2)] -= d
                    else:
                        edge_weights[(node1, node2)] = 0  # 权值减少，但不小于0

    # 计算初始合作指数
    for node in G_coop.nodes():
        compute_cooperation_index(node)

    num_runs = 10  # 设定独立运行次数
    all_cooperator_ratios = []

    for run in range(num_runs):
        print(f"开始第 {run + 1} 次独立运行")  # 打印当前运行次数
        # 初始化或重置模型，确保每次运行都是独立的
        strategies = np.random.randint(2, size=G_coop.number_of_nodes())
        edge_weights = {key: np.random.uniform(0, 1) for key in edge_weights}
        cooperator_ratios = []

        for step in range(1000):
            for node in G_coop.nodes():
                compute_cooperation_index(node)
            cooperator_count = np.sum(strategies == 0)
            cooperator_ratio = cooperator_count / G_coop.number_of_nodes()
            cooperator_ratios.append(cooperator_ratio)
            update_strategies()
            update_network()

        all_cooperator_ratios.append(cooperator_ratios)

    # 计算每个时间步的平均合作指数
    average_cooperator_ratios = np.mean(np.array(all_cooperator_ratios), axis=0)

    # 计算平稳后合作者比例的均值
    stable_cooperator_ratios = average_cooperator_ratios[800:]  # 提取平稳波动状态下的合作者比例数据
    mean_stable_cooperator_ratio = np.mean(stable_cooperator_ratios)
    print(f"Mean Cooperator Ratio in Stable State for m = {m}:", mean_stable_cooperator_ratio)

    # 记录结果
    mean_cooperator_ratios.append(mean_stable_cooperator_ratio)

# 打印最终结果
print("Mean Cooperator Ratios for each m value:", mean_cooperator_ratios)

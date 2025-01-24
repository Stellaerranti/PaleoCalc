import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from community import community_louvain
from networkx.algorithms.community import modularity
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import normalized_mutual_info_score
from infomap import Infomap

def angular_distance(lat1, lon1, lat2, lon2):
    """
    Вычисляет угловое расстояние между двумя точками на сфере по их широте и долготе.

    Аргументы:
    lat1, lon1: Координаты первой точки (в градусах).
    lat2, lon2: Координаты второй точки (в градусах).

    Возвращает:
    float: Угловое расстояние в градусах.
    """
    # Преобразуем координаты в радианы
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    # Вычисляем разницы координат
    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Формула гаверсинуса
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return c*180/np.pi

def intersection_area(r1, r2, delta):
    if delta >= r1 + r2 or delta <= abs(r1 - r2):
        return 0
    
    r1 = r1*np.pi/180
    r2 = r2*np.pi/180
    delta = delta*np.pi/180
    
    part1 = r1**2 * np.arccos((delta**2 + r1**2 - r2**2) / (2 * delta * r1))
    part2 = r2**2 * np.arccos((delta**2 + r2**2 - r1**2) / (2 * delta * r2))
    part3 = 0.5 * np.sqrt((-delta + r1 + r2) * (delta + r1 - r2) * (delta - r1 + r2) * (delta + r1 + r2))
    return part1 + part2 - part3

def calculate_average_coordinates_with_R(latitudes, longitudes, weights):
    """
    Вычисляет средние значения широты, долготы и нормализованный вектор R.

    Аргументы:
    latitudes (numpy.ndarray): Массив широт (в градусах).
    longitudes (numpy.ndarray): Массив долгот (в градусах).
    weights (numpy.ndarray): Массив весов для каждой координаты.

    Возвращает:
    tuple: Средняя широта (в градусах), средняя долгота (в градусах), нормализованный R.
    """
    if latitudes.size == 0 or longitudes.size == 0 or weights.size == 0:
        raise ValueError("Массивы широт, долгот и весов не должны быть пустыми.")

    if not (latitudes.size == longitudes.size == weights.size):
        raise ValueError("Массивы широт, долгот и весов должны быть одинакового размера.")

    # Преобразуем широты и долготы из градусов в радианы
    latitudes_rad = np.radians(latitudes)
    longitudes_rad = np.radians(longitudes)

    # Преобразуем в декартовы координаты
    x = weights * np.cos(latitudes_rad) * np.cos(longitudes_rad)
    y = weights * np.cos(latitudes_rad) * np.sin(longitudes_rad)
    z = weights * np.sin(latitudes_rad)

    # Суммируем с учётом весов и нормализуем
    total_weight = np.sum(weights)
    x_mean = np.sum(x) / total_weight
    y_mean = np.sum(y) / total_weight
    z_mean = np.sum(z) / total_weight

    # Рассчитываем нормализованный вектор R
    #R = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
    
    R = np.sqrt(np.sum((np.cos(latitudes_rad) * np.cos(longitudes_rad)))**2 + np.sum(np.cos(latitudes_rad) * np.sin(longitudes_rad))**2 + np.sum(np.sin(latitudes_rad))**2)

    # Преобразуем обратно в сферические координаты
    hyp = np.sqrt(x_mean**2 + y_mean**2)
    average_lat = np.arctan2(z_mean, hyp)
    average_lon = np.arctan2(y_mean, x_mean)

    # Преобразуем результат обратно в градусы
    average_lat = np.degrees(average_lat)
    average_lon = np.degrees(average_lon)
    
    return average_lat, average_lon, R

def calculate_node_weights(graph, subgraph_nodes):
    """
    Calculates the total weight for each node based on connected edges in the subgraph.

    Args:
        graph (networkx.Graph): The full graph.
        subgraph_nodes (list): List of nodes in the subgraph.

    Returns:
        dict: A dictionary mapping each node to its total weight.
    """
    node_weights = {node: 0 for node in subgraph_nodes}

    for node1, node2, edge_data in graph.edges(subgraph_nodes, data=True):
        weight = edge_data.get('weight', 1.0)  # Default weight is 1.0 if not specified

        # Update weights only for nodes in the subgraph
        if node1 in node_weights:
            node_weights[node1] += weight
        if node2 in node_weights:
            node_weights[node2] += weight

    return node_weights

data = np.loadtxt('data_1.txt')  # Файл с широтой, долготой и a95
latitudes = data[:,0]
longitudes = data[:,1]
a95_values = data[:,2]
num_poles = data.shape[0]

G = nx.Graph()
for i in range(num_poles):
    for j in range(i + 1, num_poles):
        delta = angular_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
        area = intersection_area(a95_values[i], a95_values[j], delta)
        if area > 0:  # Добавляем ребро, если круги пересекаются
            G.add_edge(i, j, weight=area)
            

'''          
pos = nx.spring_layout(G)  # Layout for positioning nodes
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
plt.title("Graph Representation of Circles")
plt.show()
'''

'''   
partition = community_louvain.best_partition(G, weight='weight', random_state=42)
modularity_score = community_louvain.modularity(partition, G)
#partition = nx.community.louvain_communities(G, weight='weight')

print(f"Modularity score: {modularity_score}")
pos = nx.spring_layout(G)
colors = [partition[node] for node in G.nodes()]
nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set1)
plt.show()
'''    

'''
# Map each node to its cluster index
node_to_cluster = {}
for cluster_idx, cluster in enumerate(communities):
    for node in cluster:
        node_to_cluster[node] = cluster_idx

# Visualize the graph with communities
pos = nx.spring_layout(G)
colors = [node_to_cluster[node] for node in G.nodes()]
nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set1)
plt.show()
'''

infomap = Infomap()

# Add edges from the graph to Infomap
for u, v, data in G.edges(data=True):
    weight = data.get("weight", 1.0)  # Default weight is 1.0 if not specified
    infomap.addLink(u, v, weight)

# Run the Infomap algorithm
infomap.run()

# Extract communities
node_to_cluster = {node_id: module_id for node_id, module_id in infomap.modules}

# Group nodes into communities
communities = {}
for node, cluster_id in node_to_cluster.items():
    communities.setdefault(cluster_id, []).append(node)

# Convert communities to list format
communities = list(communities.values())



# Visualize the graph with communities
colors = [node_to_cluster[node] for node in G.nodes()]
pos = nx.spring_layout(G)
nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set1)
plt.title("Graph with Infomap Communities")
plt.show()


subgraphs = []
for idx, community in enumerate(communities):
    subgraph = G.subgraph(community).copy()
    subgraphs.append(subgraph)

for idx, subgraph in enumerate(subgraphs):
    nodes = list(subgraph.nodes)

    # Extract latitudes and longitudes
    subgraph_latitudes = np.array([latitudes[node] for node in nodes])
    subgraph_longitudes = np.array([longitudes[node] for node in nodes])

    # Calculate weights for each node
    node_weights = calculate_node_weights(G, nodes)
    subgraph_weights = np.array([node_weights[node] for node in nodes])

    # Calculate weighted average coordinates
    avg_lat, avg_lon, R = calculate_average_coordinates_with_R(
        subgraph_latitudes, subgraph_longitudes, subgraph_weights)
    
    N = len(nodes)  # Число узлов в кластере
    p = 0.05  # Уровень доверия 95%
    term = (1 / p) ** (1 / (N - 1)) - 1
    alpha_95 = np.degrees(np.arccos(1 - ((N - R) / R) * term))
    
    print(f"Cluster {idx + 1}: Weighted Average Latitude = {avg_lat}, Longitude = {avg_lon}, a95 = {alpha_95}")

'''
for idx, subgraph in enumerate(subgraphs):
    print(f"Cluster {idx + 1}:")
    for node in subgraph.nodes:
        lat = latitudes[node]
        lon = longitudes[node]
        print(f"  Node {node}: Latitude = {lat}, Longitude = {lon}")
'''



for idx, subgraph in enumerate(subgraphs):
    pos = nx.spring_layout(subgraph)
    nx.draw(subgraph, pos, with_labels=True, node_color=[idx] * subgraph.number_of_nodes(), cmap=plt.cm.Set1)
    plt.title(f"Cluster {idx + 1}")
    plt.show()

#for i in G.nodes:
    #for j in G.neighbors(i):
        #weight = G[i][j]['weight']  # Вес ребра
        #total_weight += weight
        #weighted_sum += np.array(coordinates[j]) * weight




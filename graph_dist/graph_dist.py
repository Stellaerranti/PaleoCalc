import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
from infomap import Infomap

def angular_distance(lat1, lon1, lat2, lon2):
    """Compute great-circle (angular) distance between two points on a sphere."""
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = np.sin(delta_lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

   # return np.degrees(c)  # Convert back to degrees
    return c

# Load your data
file_path = "Lat_in.txt"
data = np.loadtxt(file_path)  # Load latitudes and longitudes

latitudes = data[:, 0]
longitudes = data[:, 1]
num_poles = len(latitudes)

# Create a fully connected graph with distances as edge weights
G = nx.Graph()
for i in range(num_poles):
    for j in range(i + 1, num_poles):
        distance = angular_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
        G.add_edge(i, j, weight=distance)

# Visualize the weighted graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G, weight='weight')  # Layout considering weights
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700)
edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Weighted Graph Based on Angular Distance")
plt.show()

# Apply Infomap clustering
infomap = Infomap()
for u, v, data in G.edges(data=True):
    infomap.addLink(u, v, data["weight"])

infomap.run()

# Extract communities
node_to_cluster = {node_id: module_id for node_id, module_id in infomap.modules}
communities = {}
for node, cluster_id in node_to_cluster.items():
    communities.setdefault(cluster_id, []).append(node)
communities = list(communities.values())

# Visualize clusters
colors = [node_to_cluster[node] for node in G.nodes()]
plt.figure(figsize=(8, 6))
nx.draw(G, pos, node_color=colors, with_labels=True, cmap=plt.cm.Set1)
plt.title("Graph with Infomap Communities")
plt.show()

print(f"Detected {len(communities)} clusters")
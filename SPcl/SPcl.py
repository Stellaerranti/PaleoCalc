import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import haversine_distances

def spherical_hierarchical_clustering(points, n_clusters=None, distance_threshold=None):
    """
    Cluster points on a sphere using hierarchical clustering with the great-circle distance.

    Parameters:
    - points: numpy array of shape (n, 2), where each row is [latitude, longitude] in degrees.
    - n_clusters: The number of clusters to find. If None, uses distance_threshold to determine clusters.
    - distance_threshold: The linkage distance threshold above which clusters will not be merged.
                         If None, n_clusters must be specified.

    Returns:
    - labels: Cluster labels for each point.
    """
    # Convert latitude and longitude from degrees to radians
    points_rad = np.radians(points)

    # Compute the pairwise haversine distances between points
    distance_matrix = haversine_distances(points_rad)

    # Convert the distance matrix to kilometers (Earth's radius is approximately 6371 km)
    distance_matrix_km = 6371 * distance_matrix

    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average"  # You can change this to "single", "complete", or "ward"
    )
    labels = clustering.fit_predict(distance_matrix_km)

    return labels

data = np.loadtxt('Lat_rev_only_coords.txt')

labels = spherical_hierarchical_clustering(data)

print("Cluster labels:", labels)
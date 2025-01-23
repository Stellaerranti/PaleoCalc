import numpy as np
import pandas as pd
import networkx as nx

# === Шаг 1: Загрузка данных ===
data = np.loadtxt('data_1.txt')  # Файл с широтой, долготой и a95
latitudes = data[:,0]
longitudes = data[:,1]
a95_values = data[:,2]
num_poles = data.shape[0]

# === Шаг 2: Функции для расчётов ===
def angular_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return np.degrees(np.arccos(
        np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon1 - lon2)
    ))

def intersection_area(r1, r2, delta):
    if delta >= r1 + r2 or delta <= abs(r1 - r2):
        return 0

    part1 = r1**2 * np.arccos((delta**2 + r1**2 - r2**2) / (2 * delta * r1))
    part2 = r2**2 * np.arccos((delta**2 + r2**2 - r1**2) / (2 * delta * r2))
    part3 = 0.5 * np.sqrt((-delta + r1 + r2) * (delta + r1 - r2) * (delta - r1 + r2) * (delta + r1 + r2))
    return part1 + part2 - part3

# === Шаг 3: Построение графа ===
G = nx.Graph()
for i in range(num_poles):
    for j in range(i + 1, num_poles):
        delta = angular_distance(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
        area = intersection_area(a95_values[i], a95_values[j], delta)
        if area > 0:  # Добавляем ребро, если круги пересекаются
            G.add_edge(i, j, weight=area)

# === Шаг 4: Обработка каждой связной компоненты ===
components = list(nx.connected_components(G))
results = []

for component in components:
    # Узлы, принадлежащие текущей компоненте
    nodes = list(component)
    
    # Извлечение данных узлов
    comp_latitudes = latitudes[nodes]
    comp_longitudes = longitudes[nodes]
    comp_a95 = a95_values[nodes]
    
    # Перевод в декартовы координаты
    cartesian_coords = []
    for lat, lon in zip(comp_latitudes, comp_longitudes):
        x = np.cos(np.radians(lat)) * np.cos(np.radians(lon))
        y = np.cos(np.radians(lat)) * np.sin(np.radians(lon))
        z = np.sin(np.radians(lat))
        cartesian_coords.append((x, y, z))
    cartesian_coords = np.array(cartesian_coords)
    
    # Рассчёт взвешенного среднего
    weighted_mean = np.zeros(3)
    total_weight = 0

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if G.has_edge(nodes[i], nodes[j]):
                weight = G[nodes[i]][nodes[j]]['weight']
                weighted_mean += cartesian_coords[j] * weight
                total_weight += weight

    weighted_mean /= total_weight
    R = np.linalg.norm(weighted_mean)

    # Преобразование в сферические координаты
    mean_lat = np.degrees(np.arcsin(weighted_mean[2]))
    mean_lon = np.degrees(np.arctan2(weighted_mean[1], weighted_mean[0]))
    
    # Рассчёт alpha_95 с новой формулой
    N = len(nodes)
    p = 0.05  # 95% доверительный уровень
    term = (1 / p) ** (1 / (N - 1)) - 1
    alpha_95 = np.degrees(np.arccos(1 - ((N - R) / R) * term))
    
    # Сохранение результатов
    results.append({
        'component': nodes,
        'mean_lat': mean_lat,
        'mean_lon': mean_lon,
        'R': R,
        'alpha_95': alpha_95
    })

# === Шаг 5: Вывод результатов ===
for i, res in enumerate(results):
    print(f"Компонента {i + 1}:")
    print(f"  Узлы: {res['component']}")
    print(f"  Среднее направление: широта = {res['mean_lat']:.2f}°, долгота = {res['mean_lon']:.2f}°")
    print(f"  Длина результирующего вектора (R): {res['R']:.2f}")
    print(f"  Alpha_95: {res['alpha_95']:.2f}°\n")

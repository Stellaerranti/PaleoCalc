import numpy as np

import math

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

def calculate_average_coordinates(latitudes, longitudes):
    """
    Вычисляет средние значения широты и долготы из массивов numpy с учётом сферической природы координат.

    Аргументы:
    latitudes (numpy.ndarray): Массив широт (в градусах).
    longitudes (numpy.ndarray): Массив долгот (в градусах).

    Возвращает:
    tuple: Средняя широта и средняя долгота (в градусах).
    """
    if latitudes.size == 0 or longitudes.size == 0:
        raise ValueError("Массивы широт и долгот не должны быть пустыми.")

    if latitudes.size != longitudes.size:
        raise ValueError("Массивы широт и долгот должны быть одинакового размера.")

    # Преобразуем широты и долготы из градусов в радианы
    latitudes_rad = np.radians(latitudes)
    longitudes_rad = np.radians(longitudes)

    # Преобразуем в декартовы координаты
    x = np.cos(latitudes_rad) * np.cos(longitudes_rad)
    y = np.cos(latitudes_rad) * np.sin(longitudes_rad)
    z = np.sin(latitudes_rad)

    # Находим средние значения
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    z_mean = np.mean(z)

    # Преобразуем обратно в сферические координаты
    hyp = np.sqrt(x_mean**2 + y_mean**2)
    average_lat = np.arctan2(z_mean, hyp)
    average_lon = np.arctan2(y_mean, x_mean)

    # Преобразуем результат обратно в градусы
    average_lat = np.degrees(average_lat)
    average_lon = np.degrees(average_lon)

    return average_lat, average_lon

def getS(VGP_lon, VGP_lat, NumberOfSites,cutoff):
    
    s = 0
    count = 0
    
    VGP_mean_lat, VGP_mean_lon = calculate_average_coordinates(VGP_lat, VGP_lon)

    for i in range(NumberOfSites):
        if(angular_distance(VGP_mean_lat,VGP_mean_lon,VGP_lat[i],VGP_lon[i])<cutoff):
           
            s = s + (angular_distance(VGP_mean_lat,VGP_mean_lon,VGP_lat[i],VGP_lon[i]))**2
            count+=1
     
    return math.sqrt(s/(count-1))

input_data = np.loadtxt('test S_2.txt')
cutoff = 45
S   = getS(input_data[:,0],input_data[:,1], input_data.shape[0],cutoff)

print(f'S: {S:.2f}')
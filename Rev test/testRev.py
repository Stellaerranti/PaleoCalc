import numpy as np
import math

def calculate_average_coordinates(latitudes, longitudes):

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
    
    x_sum = np.sum(x)
    y_sum = np.sum(y)
    z_sum = np.sum(z)
    
    R = np.sqrt(x_sum**2 + y_sum**2 + z_sum**2)
    
    K = (x.shape[0] - 1)/(x.shape[0] - R)

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

    return average_lat, average_lon, K, x.shape[0]

longitudes,latitudes = np.loadtxt('SW all.txt', unpack = True)

av_lat, av_lon, K, N = calculate_average_coordinates(latitudes, longitudes)

print (f'Latitude: {av_lat:.2f}')
print (f'Longitude: {av_lon:.2f}')
print (f'Pres: {K:.2f}')
print (f'N: {N:.2f}')
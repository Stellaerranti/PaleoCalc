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

def doVandamme(VGP_lon, VGP_lat, NumberOfSites,paleolat,K,N):
    
    VGP_mean_lat, VGP_mean_lon = calculate_average_coordinates(VGP_lat, VGP_lon)
    
    Theta = angular_distance(VGP_mean_lat,VGP_mean_lon,VGP_lat,VGP_lon)
    
    ASD = np.sqrt(np.sum(Theta**2)/(Theta.shape[0]-1))
    
    A = 1.8 * ASD + 5.
    
    Theta_max = Theta.max()
    
    if (Theta_max<A):
        return A, getSb(VGP_lon, VGP_lat, VGP_lon.shape[0],paleolat,K,N,A)[1], ASD, Theta.shape[0]
    
    while Theta_max > A:
        
        Theta_max = Theta.max()
        
        if (Theta_max<A):
            return A, getSb(VGP_lon, VGP_lat, VGP_lon.shape[0],paleolat,K,N,A)[1], ASD, Theta.shape[0]
        
        VGP_lon, VGP_lat = VGP_lon[Theta < Theta_max], VGP_lat[Theta < Theta_max]
        
        VGP_mean_lat, VGP_mean_lon = calculate_average_coordinates(VGP_lat, VGP_lon)
        
        Theta = angular_distance(VGP_mean_lat,VGP_mean_lon,VGP_lat,VGP_lon)
        
        ASD = np.sqrt(np.sum(Theta**2)/(Theta.shape[0]-1))
        
        A = 1.8 * ASD + 5.        


def doBootstrap(VGP_lon, VGP_lat, NumberOfSites, paleolat, K, N, cutoff, nb):
    VGP_lon = np.array(VGP_lon)
    VGP_lat = np.array(VGP_lat)
    
    Sbs = []
    
    for i in range(nb):
        indices = np.random.choice(NumberOfSites, size=NumberOfSites, replace=True)
        
        VGP_lon_bootstrap = VGP_lon[indices]
        VGP_lat_bootstrap = VGP_lat[indices]
        
        paleolat_bootstrap = paleolat[indices]
        K_bootstrap = K[indices]
        N_bootstrap = N[indices]
        
        Sb_value = getSb(VGP_lon_bootstrap, VGP_lat_bootstrap, NumberOfSites, paleolat_bootstrap, K_bootstrap, N_bootstrap, cutoff)[0]
        if Sb_value is not None:  # Avoid adding None values
            Sbs.append(Sb_value)
    
    if len(Sbs) == 0:
        print("Bootstrap produced no valid Sb values.")
        return None, None

    Sbs.sort()
    
    #print(f"Bootstrap Results (first 10 values): {Sbs[:10]}")
    #print(f"Bootstrap Results (last 10 values): {Sbs[-10:]}")
    
    return Sbs[int(.025 * nb)], Sbs[int(.975 * nb)]

def getSw(paleolat,ks):
    
    K = ks/((5 + 18*math.sin(np.radians(paleolat))**2 + 9*math.sin(np.radians(paleolat))**4)/8)
    return 81/math.sqrt(K)
    #return 0
    

def getSb(VGP_lon, VGP_lat, NumberOfSites, paleolat, K, N, cutoff):
    st = 0
    s = 0
    count = 0
    
    VGP_mean_lat, VGP_mean_lon = calculate_average_coordinates(VGP_lat, VGP_lon)

    for i in range(NumberOfSites):
        if angular_distance(VGP_mean_lat, VGP_mean_lon, VGP_lat[i], VGP_lon[i]) < cutoff:
            
            distance_sq = (angular_distance(VGP_mean_lat, VGP_mean_lon, VGP_lat[i], VGP_lon[i]))**2
            
            sw_value = getSw(paleolat[i], K[i])**2 / N[i]

            st += distance_sq - sw_value
            s += distance_sq
            count += 1

    if count <= 1:
        print("Warning: count is too small, returning zeros.")
        return 0, 0, count  # Avoid division by zero

    s_value = s / (count - 1)
    st_value = st / (count - 1)

    # Prevent sqrt of negative numbers
    s_sqrt = math.sqrt(max(s_value, 0))  # Ensures non-negative value
    st_sqrt = math.sqrt(max(st_value, 0))

    return s_sqrt, st_sqrt, count 



input_data = np.loadtxt('kupol.txt')
cutoff = 45
nb = 1000


#Порядок в файле:
#Долгота, широта, палеоширота сайта, кучность внутри сайта, число образцов в сайте

S, Sb, count_cutoff  = getSb(input_data[:,0],input_data[:,1], input_data.shape[0], input_data[:,2], input_data[:,3], input_data[:,4],cutoff)
Vandamme_cutoff, Vandamme_Sb, Vandamme_S, N_Vandamme = doVandamme(input_data[:,0],input_data[:,1], input_data.shape[0], input_data[:,2], input_data[:,3], input_data[:,4])

low, high = doBootstrap(input_data[:,0],input_data[:,1], input_data.shape[0], input_data[:,2], input_data[:,3], input_data[:,4],cutoff, nb)

Vandamme_low, Vandamme_high = doBootstrap(input_data[:,0],input_data[:,1], input_data.shape[0], input_data[:,2], input_data[:,3], input_data[:,4],Vandamme_cutoff, nb)
print(f'S: {S:.2f}')
print(f'Sb: {Sb:.2f}')
print(f'Cutoff: {cutoff:.2f}')
print(f'N cutoff: {count_cutoff}')
print(f'Low: {low:.2f}')
print(f'High: {high:.2f}')
print(f'Vandamme S: {Vandamme_S:.2f}')
print(f'Vandamme Sb: {Vandamme_Sb:.2f}')
print(f'Vandamme cutoff: {Vandamme_cutoff:.2f}')
print(f'N Vandamme: {N_Vandamme}')        
print(f'Vandamme low: {Vandamme_low:.2f}')
print(f'Vandamme high: {Vandamme_high:.2f}')

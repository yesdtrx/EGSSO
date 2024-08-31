import numpy as np

data_array = np.array([
    -57.88500646, -202.2265434, -131.1171825, -81.63117314, -97.48594047,
    -113.3645294, -135.357305, -272.7240121, -178.3330416, -137.1384771,
    -268.175862, -230.4443241, -263.9356473, -260.9727594, -231.0065054,
    -215.4105496, -535.277963, -436.9976155, -197.6337148, -636.473953,
    -395.2359848, -390.7955922, -405.4047796, -369.3238334, -499.0296056,
    -426.8780105, -380.748533, -490.0008522, -760.4911448, -1985.067257,
    -642.5873471, -842.9150079, -1076.433053, -1179.562155, -508.5726016,
    -281.3054389, -185.2782463, -187.002178, -237.4560003, -317.1423539,
    -371.0443891, -188.1432481, -249.063287, -190.5439982, -161.7200153,
    -175.370561, -151.9039676, -769.5964171, -311.8141026, -430.2055306,
    -452.5355621, -301.1060611, -353.8148477, -1495.423327, -936.7264588,
    -668.2928203, -1202.266971, -1093.062305, -1185.636227
])

def detect_outliers(data, threshold):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev for x in data]
    outliers_indices = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
    return outliers_indices

outliers_indices = detect_outliers(data_array, threshold=3)

print("Outlier：", data_array[outliers_indices])

distance_matrix = np.abs(data_array.reshape(-1, 1) - data_array)

cleaned_data = data_array.copy()
if len(outliers_indices) > 0:
    for i in outliers_indices:
        valid_indices = np.delete(np.arange(len(data_array)), outliers_indices)
        nearest_index = valid_indices[np.argmin(distance_matrix[i, valid_indices])]
        cleaned_data[i] = data_array[nearest_index]


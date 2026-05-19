# Script for programming tests

# Imports
from datetime import datetime
import os
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import glob
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import KDTree

## Tests
# 1) pandas series
serie1 = pd.Series([7.498016e+08])

# 2) 
"""dates = np.array(['2023-07-29T22:11:58' ,'2023-08-29T06:28:25', '2023-09-09T15:42:10',
 '2023-09-19T03:13:31' ,'2023-09-30T12:27:13', '2023-10-21T09:12:17',
 '2023-10-30T20:43:38'], dtype='datetime64[ns]')
target_date = np.datetime64('2023-07-29', 'D')
dates_day = dates.astype('datetime64[D]')
print(np.where(dates_day > target_date)[0]) # renvoie les indices qui correspondent aux dates voulues"""

# 3)
x = np.array([711, 712, 722,731])
y = np.array([3, 4, 5,6])
stacked_xy = np.vstack((x,y)).T # https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy-vstack
"""[[711   3]
 [712   4]
 [742   5]
 [731   6]]"""

target_lat, target_lon = 7, 750
target = np.array([target_lat,target_lon]) # (2,)

distance, index = KDTree(stacked_xy).query([target_lon, target_lat],k=1)
print(index)



"""
# chat
target_lat, target_lon = 4.2, 712.5
distances = np.sqrt((x - target_lon)**2 + (y - target_lat)**2)
best_index = np.argmin(distances)

print(f"Index de la paire la plus proche : {best_index}")
print(f"Coordonnées trouvées : lat={y[best_index]}, lon={x[best_index]}")
print(f"Distance : {distances[best_index]:.4f}")
"""
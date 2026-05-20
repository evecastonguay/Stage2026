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
from zipfile import ZipFile
import netCDF4 as nc

## Tests
# 1) pandas series
serie1 = pd.Series([7.498016e+08])

# 2) probleme soudain swot
x = np.array([711, 711, 711,711]) #lon
y = np.array([3, 4, 5,6]) # lat
stacked_xy = np.vstack((x,y)).T # https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy-vstack
#[[711   3]
 #[712   4]
# [742   5]
 #[731   6]]

target_lat, target_lon = 7, 750
target = np.array([target_lat,target_lon]) # (2,)

distance, index = KDTree(stacked_xy).query([target_lon, target_lat],k=2)
#print(index[0])

tt = 3
while tt < 10:
    print(tt)
    tt +=1
    fill = 711
    mask_fill = x != fill
    x_filtered = x[mask_fill]
    print(type(x_filtered))
    if x_filtered.size == 0:
        print("vide")
        continue
    print("nonono")
    break



"""
path = '/obs/ecastonguay/grdc_data/na/GRDC-Daily.nc'
data = xr.open_dataset(path, engine="netcdf4")
<xarray.Dataset> Size: 977kB
Dimensions:              (time: 20252, id: 10)
Coordinates:
  * time                 (time) datetime64[ns] 162kB 1970-05-01 ... 2025-10-10
  * id                   (id) int64 80B 4101200 4101250 ... 4101500 4101501
Data variables:
    runoff_mean          (time, id) float32 810kB ...
    area                 (id) float32 40B ...
    country              (id) <U2 80B ...
    geo_x                (id) float32 40B ...
    geo_y                (id) float32 40B ...
    geo_z                (id) float32 40B ...
    owneroforiginaldata  (id) <U58 2kB ...
    river_name           (id) <U19 760B ...
    station_name         (id) <U44 2kB ...
    timezone             (id) float32 40B ...
Attributes:
    title:          Mean daily discharge (Q)
    Conventions:    CF-1.7
    references:     grdc.bafg.de
    institution:    GRDC
    history:        Download from GRDC Database, 20/05/2026
    missing_value:  -999.000"""



"""
# chat
target_lat, target_lon = 4.2, 712.5
distances = np.sqrt((x - target_lon)**2 + (y - target_lat)**2)
best_index = np.argmin(distances)

print(f"Index de la paire la plus proche : {best_index}")
print(f"Coordonnées trouvées : lat={y[best_index]}, lon={x[best_index]}")
print(f"Distance : {distances[best_index]:.4f}")
"""

"""x = np.array([711, 712, 722,731]) #lon
y = np.array([3, 4, 5,6]) # lat
stacked_xy = np.vstack((x,y)).T # https://numpy.org/doc/stable/reference/generated/numpy.vstack.html#numpy-vstack
#[[711   3]
 #[712   4]
# [742   5]
 #[731   6]]

target_lat, target_lon = 7, 750
target = np.array([target_lat,target_lon]) # (2,)

distance, index = KDTree(stacked_xy).query([target_lon, target_lat],k=1)
print(index)
"""
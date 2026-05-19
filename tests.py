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

## Tests
# 1) pandas series
serie1 = pd.Series([7.498016e+08])

# 2) 
dates = np.array(['2023-07-29T22:11:58' ,'2023-08-29T06:28:25', '2023-09-09T15:42:10',
 '2023-09-19T03:13:31' ,'2023-09-30T12:27:13', '2023-10-21T09:12:17',
 '2023-10-30T20:43:38'], dtype='datetime64[ns]')
target_date = np.datetime64('2023-07-29', 'D')
dates_day = dates.astype('datetime64[D]')
print(np.where(dates_day > target_date)[0]) # renvoie les indices qui correspondent aux dates voulues



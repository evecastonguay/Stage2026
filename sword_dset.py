### Eve Castonguay, LIRA
### This scripts put all relevant SWORD data into a global dset;
### x, y, reach id, wse, width, facc, reach length, river name

# Imports
from datetime import datetime
import os
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import glob 
import numpy as np
import netCDF4 as nc
from scipy.spatial import KDTree
from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import math
from scipy import stats
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import geopy.distance
import random
import re
from rapidfuzz import fuzz

# Utils file for external functions, with auto-reload
"""%load_ext autoreload
%autoreload 2"""
import utils


version = 'v1'
## Section 1 : Setting some variables
# 1.1 Continents
continent_list = ['na', 'af', 'as', 'eu', 'sa', 'oc']
# 1.2 Period during which SWOT has data (2023-03-29 to 2025-05-02)
swot_start = '2023-03-29'
swot_end = '2025-05-02'
# 1.3 Empty dictionnaries for the DataArrays
x_darrays = {}
y_darrays = {}
wse_darrays = {}
width_darrays = {} 
facc_darrays = {}
r_len_darrays = {} 
name_darrays = {} 

## Section 2 : Loop over continents 
for i_continent in continent_list: 
    
    ## Section 3 : Sword extraction
    # 3.1 Reading
    dir_swr = "/obs/ecastonguay/sword_data/netcdf_v16" # sword v16
    file_swr = i_continent + "_sword_v16.nc"
    path_swr = os.path.join(dir_swr,file_swr)
    data_swr = nc.Dataset(path_swr) # open the netcdf file
    # 3.2 Extracting
    x_swr = data_swr["reaches"]["x"][:]
    y_swr = data_swr["reaches"]["y"][:]
    r_id_swr = data_swr["reaches"]["reach_id"][:]
    wse_swr = data_swr["reaches"]["wse"][:]
    width_swr = data_swr["reaches"]["width"][:]
    facc_swr = data_swr["reaches"]["facc"][:]
    r_len_swr = data_swr["reaches"]["reach_length"][:]
    river_swr = data_swr["reaches"]["river_name"][:]
    assert len(x_swr) == len(y_swr) == len(r_id_swr) == len(wse_swr) == len(width_swr) == len(facc_swr) == len(r_len_swr) == len(river_swr), \
    f"Lengths do not match : x={len(x_swr)}, y={len(y_swr)}, r_id={len(r_id_swr)}, wse={len(wse_swr)}, width={len(width_swr)}, facc={len(facc_swr)}, r_len={len(r_len_swr)}, river={len(river_swr)}"
    # 3.3 In DataArrays
    x_darray = xr.DataArray(
        data=x_swr,
        dims=["reach_id"],
        coords=dict(reach_id=r_id_swr,),
        name="x (lon, degrees east) per reach sword") 
    y_darray = xr.DataArray(
        data=y_swr,
        dims=["reach_id"],
        coords=dict(reach_id=r_id_swr,),
        name="y (lat, degrees north) per reach sword") 
    wse_darray = xr.DataArray(
        data=wse_swr,
        dims=["reach_id"],
        coords=dict(reach_id=r_id_swr,),
        name="wse (m) per reach sword") 
    width_darray = xr.DataArray(
        data=width_swr,
        dims=["reach_id"],
        coords=dict(reach_id=r_id_swr,),
        name="width (m) per reach sword") 
    facc_darray = xr.DataArray(
        data=facc_swr,
        dims=["reach_id"], 
        coords=dict(reach_id=r_id_swr,),
        name="facc (km^2) sword") 
    r_len_darray = xr.DataArray(
        data=r_len_swr,
        dims=["reach_id"], 
        coords=dict(reach_id=r_id_swr,),
        name="reach length (m) sword")
    river_darray = xr.DataArray(
        data=river_swr, 
        dims=["reach_id"], 
        coords=dict(reach_id=r_id_swr,),
        name="river name per reach sword") 

    ## Section 4 : Append continent DataArrays to list
    x_darrays[i_continent] = x_darray
    y_darrays[i_continent] = y_darray
    wse_darrays[i_continent] = wse_darray
    width_darrays[i_continent] = width_darray
    facc_darrays[i_continent] = facc_darray
    r_len_darrays[i_continent] = r_len_darray
    name_darrays[i_continent] = river_darray

# end of the (for i_continent in continent_list) loop 

## Section 5 : Concathenate continent DataArrays. These are individual DataArrays that contain values for all continents
x_global = xr.concat(list(x_darrays.values()), dim='reach_id') 
y_global = xr.concat(list(y_darrays.values()), dim='reach_id')
wse_global = xr.concat(list(wse_darrays.values()), dim='reach_id')
width_global = xr.concat(list(width_darrays.values()), dim='reach_id')
facc_global = xr.concat(list(facc_darrays.values()), dim='reach_id')
r_len_global = xr.concat(list(r_len_darrays.values()), dim='reach_id')
name_global = xr.concat(list(name_darrays.values()), dim='reach_id')

## Section 6 : Create the DataSet and append all the variables (DataArrays) to it
sword_dset = x_global.to_dataset(name='x_swr')
sword_dset['y_swr'] = y_global
sword_dset['wse_swr'] = wse_global
sword_dset['width_swr'] = width_global
sword_dset['facc_swr'] = facc_global
sword_dset['r_len_swr'] = r_len_global
sword_dset['name_swr'] = name_global

# Print
print(sword_dset)

# Save to netcdf
sword_dset.to_netcdf("/obs/ecastonguay/scripts/sword_dset_" + version + ".nc")

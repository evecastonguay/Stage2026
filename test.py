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
#
# 2) 
dir_l4 = "/obs/ecastonguay/swot_data/L4_discharge/"
file_suffix = "_sword_v16_SOS_results_unconstrained_20230502T204408_20250502T204408_20251219T163700.nc"
continent = "na"
single_file_name = dir_l4 + continent + file_suffix # *** change to i_continent everywhere!
data_l4 = nc.Dataset(single_file_name)
list = [0,1,2,3,4]
discharge_darray = data_l4.groups["consensus"]['consensus_q'][list] # (5,). this is <class 'numpy.ndarray'> containing other <class 'numpy.ndarray'>
time_darray = data_l4.groups["consensus"]['time_int'][list] # (5,)
print(type(discharge_darray[4][0]))
#print(time_darray)


### Goal: have a (n_station,time) ndarray
# make the time dimension
from datetime import date, timedelta
sdate = date(2023,3,29)   # start date
edate = date(2025,5,2)   # end date
time_dimension = [sdate+timedelta(days=x) for x in range((edate-sdate).days+1)] # making sure the end date is included. list of datetime.date
# OR
time2_dimension = pd.date_range(start='2023-03-29', end='2025-05-02', freq='D') # ndarray-like of datetime64 data. dtype='datetime64[us]
# make the station dimension
# *** make sure discharge and time have same size
station_dimension = np.arange(0,discharge_darray.shape[0]) # 5 stations

# make empty DataArray
Y = len(time2_dimension) #766
X = len(discharge_darray) #5
print(X)

discharge_data = np.full((X, Y), np.nan, dtype=np.float64)


# 1) mask NaN
missing_val = data_l4.groups["consensus"]['consensus_q'].missing_value # [tested]
mask_fill_values = (discharge_darray != missing_val) # [tested]
discharge_filtered = discharge_darray[mask_fill_values] 

print(discharge_filtered)

# 
print("solution 1")
goal = np.array(discharge_darray)
print(goal.shape)

print("solution 2")
goal2 = np.vstack(discharge_darray)


"""<xarray.Dataset> Size: 65MB
Dimensions:              (time: 44459, id: 360)
Coordinates:
  * time                 (time) datetime64[ns] 356kB 1904-07-31 ... 2026-04-20
  * id                   (id) int64 3kB 1159100 1159103 ... 1992900 1993401
Data variables:
    runoff_mean          (time, id) float32 64MB ...
    area                 (id) float32 1kB ...
    country              (id) <U2 3kB ...
    geo_x                (id) float32 1kB ...
    geo_y                (id) float32 1kB ...
    geo_z                (id) float32 1kB ...
    owneroforiginaldata  (id) <U99 143kB ...
    river_name           (id) <U23 33kB ...
    station_name         (id) <U41 59kB ...
    timezone             (id) float32 1kB ...
Attributes:
    title:          Mean daily discharge (Q)
    Conventions:    CF-1.7
    references:     grdc.bafg.de
    institution:    GRDC
    history:        Download from GRDC Database, 21/05/2026
    missing_value:  -999.000"""

"""
runoff_data = data['runoff_mean'].sel(id=1159100) # selectionne station. dataarray tranché en fonction de l'id
target_date_inf = "2023-08-18" # SELECT a time period for the graphs
target_date_sup = "2024-09-21"
runoff_grdc_comparison = runoff_data.sel(time=slice(target_date_inf,target_date_sup)) # selectionne période pr station

plt.figure(figsize=(12, 6))
plt.plot(runoff_grdc_comparison.time.values, runoff_grdc_comparison.values, marker='o', markersize=5, color='darkorange', markeredgecolor='white', markeredgewidth=0.4)
plt.grid(True)
#plt.show() """


"""# 2) vérifier que mes données swot sont ok [fait]
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

## Section 0 : Make sure the following variables are set correctly before running the code
continent = "na" # SELECT a continent (africa: af, asia: as, europe: eu, north_america: na, south_america: sa, oceania: oc)
target_date_inf = "2023-07-29" # SELECT a time period for the graphs
target_date_sup = "2024-08-02" # SWOT data : 2023-03-29 to 2025-05-02
use_target_date_filter = True # SELECT True if we want to use the above specified target dates, False if the goal is to display the discharge data for all time period available
# 0.2 Plot regarding all discharge data for a specific reach
selected_reach_id = 74210000201 # SELECT a reach to plot. 
                                # 74210000201 article 3, fig 2a - reach on mississippi near bâton rouge (na)
                                # 81130400011 article 3, fig 2b - the reach my code found (na)
                                # 81130400021 article 3, fig 2b - the reach they actually used (na)
                                # 21602400201 article 3, fig 2c - the reach they actually used (eu)
                                # 23229000561 article 3, fig 2d - the reach they actually used (eu)
plot_reach_discharge = 1 # SELECT 1 to plot, 0 to ignore

## Section 1 : Extracting the dataset from the SWOT continent file
# This code loads the single selected continent file
dir_l4 = "/obs/ecastonguay/swot_data/L4_discharge/"
if continent not in ['af', 'as', 'eu', 'na', 'sa', 'oc']:
    raise ValueError("Error: [1] Continent must be one of the following: 'af' (Africa), 'as' (Asia), 'eu' (Europe), 'na' (North America), 'sa' (South America), 'oc' (Oceania)")    
file_suffix = "_sword_v16_SOS_results_unconstrained_20230502T204408_20250502T204408_20251219T163700.nc"
single_file_name = dir_l4 + continent + file_suffix
data_l4 = nc.Dataset(single_file_name)

# verif de chat
# Ajoute ça AVANT ta boucle de traitement :
print("=== DIAGNOSTIC ===")
print(f"consensus_q shape: {data_l4.groups['consensus']['consensus_q'].shape}")
print(f"time_int shape: {data_l4.groups['consensus']['time_int'].shape}")
print(f"reach_id shape: {data_l4.groups['reaches']['reach_id'].shape}")

# Affiche les métadonnées temporelles
time_var = data_l4.groups['consensus']['time_int']
print(f"time_int units: {time_var.units if hasattr(time_var, 'units') else 'NOT FOUND'}")
print(f"time_int calendar: {time_var.calendar if hasattr(time_var, 'calendar') else 'NOT FOUND'}")


# 4.2 Plot all discharge data for a specific reach, without in situ data comparison
if plot_reach_discharge == 1:
    reach_id_values = data_l4.groups['reaches']['reach_id'][:]
    selected_reach_index_array = np.where(reach_id_values == selected_reach_id) # find the index of the reach i'm looking for (the array contains the index)
                                                                                # array within array; [0][0]
    if (len(selected_reach_index_array[0]) == 0):
        print("Error: [4.2] Reach id not found in the list of reach ids. Please check the reach id and try again.")
    else:
        # discharge
        selected_reach_index = selected_reach_index_array[0][0] # OK
        
        consensus_q_data = data_l4.groups["consensus"]['consensus_q'][:] # ndarray
        consensus_q_variable = data_l4.groups["consensus"]['consensus_q'] # ndarray
        
        print(consensus_q_data.size)
        discharge_selected_reach = consensus_q_data[selected_reach_index] # tableau des données par heure pour reach xxxxx
        #print('data unmasked',discharge_selected_reach) 

        # print reach's coordinates
        print(f"The coordinates of the reach are ({data_l4.groups['reaches']['y'][:][selected_reach_index]},{data_l4.groups['reaches']['x'][:][selected_reach_index]})") # (60.52684230165204,-151.13600862937432)
        
        # masking the missing discharge values
        print(consensus_q_variable.missing_value)
        mask_fill_q = discharge_selected_reach != consensus_q_variable.missing_value # *** use numpy masked array instead?
        discharge_selected_reach = discharge_selected_reach[mask_fill_q]
        
        # finding the corresponding time
        time_selected_reach = data_l4.groups["consensus"]['time_int'][:][selected_reach_index]
        time_selected_reach = time_selected_reach[mask_fill_q] # again masking the missing values

        # Après extraction, vérifie la synchronisation :
        print(f"discharge array shape after indexing: {discharge_selected_reach.shape}")
        print(f"time array shape after indexing: {time_selected_reach.shape}")
        print(f"mask True count: {mask_fill_q.sum()}")
        #print(f"discharge shape after masking: {discharge_selected_reach[mask_fill_q].shape}")
        #print(f"time shape after masking: {time_selected_reach[mask_fill_q].shape}")

        # converting time from int to datetime
        epoch = np.datetime64('2000-01-01')
        datetime_selected_reach = epoch + time_selected_reach.astype('timedelta64[s]') 
        
        yyy = 1
        if yyy == 1:
            diff = (datetime(2000,1,1) - datetime(1970,1,1)).total_seconds()
            time_plot_sp = pd.to_datetime(time_selected_reach + diff, unit='s') # *** not exact, seconds missing,
             # method1 == method2 !!
        # slicing SWOT data for the right time period
        if use_target_date_filter:
            target_datetime_inf = np.datetime64(target_date_inf, 'D')
            target_datetime_sup = np.datetime64(target_date_sup, 'D')
            index_target_datetime_period = np.where((datetime_selected_reach >= target_datetime_inf) & (datetime_selected_reach <= target_datetime_sup))
            print('index target datetime period',index_target_datetime_period)
            datetime_plot = datetime_selected_reach[index_target_datetime_period]
            print('datetime plot',datetime_plot)
            discharge_plot = discharge_selected_reach[index_target_datetime_period]  
            print('discharge plot',discharge_plot)
        else:
            datetime_plot = datetime_selected_reach
            discharge_plot = discharge_selected_reach
        
        plt.figure(figsize=(12, 6))
        plt.plot(datetime_plot,discharge_plot, marker='o', markersize=5, color='darkorange', markeredgecolor='white', markeredgewidth=0.4)

        plt.title(f"SWOT L4 consensus discharge for reach {selected_reach_id}") # SELECT title
        plt.xlabel('Time (UTC)') 
        plt.ylabel(r'Consensus discharge (m$^3$/s)')

        plt.grid(True)

        if use_target_date_filter:
            fig_name = f'discharge_r{selected_reach_id}_{target_date_inf}_{target_date_sup}.png' # SELECT file name
        else:
            fig_name = f'discharge_r{selected_reach_id}.png' # SELECT file name
        plt.savefig(f'/obs/ecastonguay/scripts/figures/{fig_name}', dpi=400, bbox_inches='tight')
        print("Figure saved in /obs/ecastonguay/scripts/figures")
        plt.show()  
"""
    

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
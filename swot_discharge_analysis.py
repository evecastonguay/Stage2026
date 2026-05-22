# 
# Author: Eve Castonguay, LIRA (CNRS)
# Creation date: 2026-05-15 [YYYY-MM-DD]
# Version 0.1: AAAA-MM-JJ

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
continent = "eu" # SELECT a continent (africa: af, asia: as, europe: eu, north_america: na, south_america: sa, oceania: oc)
target_date_inf = "2023-08-18" # SELECT a time period for the graphs
target_date_sup = "2024-09-21" # SWOT data : 2023-03-29 to 2025-05-02
use_target_date_filter = True # SELECT True if we want to use the above specified target dates, False if the goal is to display the discharge data for all time period available
# 0.1 Plot regarding discharge comparison
list_stations_plot = [3626000]  # SELECT stations in GRDC database to compare with SWOT data. Enter their stations id in int format.
                                # na
                                # 4127800 : article 3, fig 2a - Mississippi River near Baton Rouge, Louisiana, United States
                                # 4105850 : article 3, fig 2b - Kenai River near Soldotna, Alaska, United States
                                # eu
                                # 6139415 : article 3, fig 2c - Le Drac River near Grenoble, France
                                # sa
                                # 3626000 : amazonie, cours d'eau large de 3 km
plot_discharge_comparison = 0 # SELECT 1 to plot, 0 to ignore
# 0.2 Plot regarding all discharge data for a specific reach
selected_reach_id = 23229000561 # SELECT a reach to plot. 
                                # 74210000201 article 3, fig 2a - reach on mississippi near bâton rouge (na)
                                # 81130400011 article 3, fig 2b - the reach my code found (na)
                                # 81130400021 article 3, fig 2b - the reach they actually used (na)
                                # 21602400201 article 3, fig 2c - the reach they actually used (eu)
                                # 23229000561 article 3, fig 2d - the reach they actually used (eu)
plot_reach_discharge = 1 # SELECT 1 to plot, 0 to ignore

## Section 1 : Extracting the dataset from each SWOT continent file
# This code loads the single selected continent file
dir_l4 = "/obs/ecastonguay/swot_data/L4_discharge/"
continent_list = ['af', 'as', 'eu', 'na', 'sa', 'oc']
### HERE: loop over continents
if continent not in continent_list:
    raise ValueError("Error: [1] Continent must be one of the following: 'af' (Africa), 'as' (Asia), 'eu' (Europe), 'na' (North America), 'sa' (South America), 'oc' (Oceania)")    
file_suffix = "_sword_v16_SOS_results_unconstrained_20230502T204408_20250502T204408_20251219T163700.nc"
single_file_name = dir_l4 + continent + file_suffix
data_l4 = nc.Dataset(single_file_name)

## Section 2 : Reading data
# 2.1 Extracting the GRDC stations id of the continent
dir_grdc_prefix = "/obs/ecastonguay/grdc_data/"
file_nc = continent + ".nc"
path_nc = os.path.join(dir_grdc_prefix,continent,file_nc)
file_json = "stationbasins_" + continent + ".geojson"
path_json = os.path.join(dir_grdc_prefix,continent,file_json)
# open the netcdf grdc file
data_grdc = xr.open_dataset(path_nc, engine="netcdf4")
# get a list of the stations_id
list_station_id = data_grdc["id"].values # [tested]

# 2.2 Loop for reading all the GRDC stations of the continent
for station_id in list_station_id:
    station_data = data_grdc.sel(id=station_id) # DataArray sliced by station id. [tested]
    # filtering GRDC data for the period during which SWOT has data (2023-03-29 to 2025-05-02)                     
    station_data_2023_2025 = station_data.sel(time=slice('2023-03-29','2025-05-02'))

    # Section 3 : Finding a match between the GRDC gauge and the SWOT data at reach level
    # get grdc (lat,lon) coordinates of gauge
    x_found_station = station_data_2023_2025["geo_x"].values # x (lat) at the found station
    y_found_station = station_data_2023_2025["geo_y"].values
    # find the corresponding reach in the continent-level swot data (looking for the reach with the closest coordinates to the station) (consensus_q only available at reach-level)
    reach_group = data_l4.groups['reaches']
    x_coordinates_values = reach_group['x'][:] # all x coordinates in swot/l4 data for the na continent
    y_coordinates_values = reach_group['y'][:]
    # K-D tree to search nearest neighbor
    stacked_xy = np.vstack((x_coordinates_values,y_coordinates_values)).T
    k_neighbors = 10
    distance_list, index_list = KDTree(stacked_xy).query([x_found_station, y_found_station],k=k_neighbors) # we take the k nearest neighboors so that if the nearest reach doesn't contain swot data, we look at the second nearest, ...
    ii = 0
    while ii < k_neighbors:
        distance = distance_list[ii]
        index = index_list[ii] # index of the reach we will be associating with the station
        
        ## Section 4 : Fetching the data from both sources
        # 4.1 SWOT - Discharge data (& checking if the selected nearest reach contains data; if not, will move on to the next nearest reach)
        discharge_data_check = data_l4.groups["consensus"]['consensus_q'][index] # [tested]  <class 'numpy.ndarray'>
        mask_fill_values = (discharge_data_check != data_l4.groups["consensus"]['consensus_q'].missing_value) # [tested]
        consensus_q_swot_filtered = discharge_data_check[mask_fill_values]
        if consensus_q_swot_filtered.size == 0:
            ii += 1
            print("Warning: The closest reach doesn't contain valid data. Moving on to the next closest reach.")
            continue
        if (distance > 0.5):    # even if the closest reach is far, KDTree will still find a match. therefore we need to make sure the distance found isn't too big
                                # for reference, a 5,4 km distance between reach and gauge equals to a distance = 0.0557
                                # ADJUST maximum distance
            print(f"Warning: [4.1] The nearest reach associated with the station {station_id} is quite far from it's location. This might have an impact on the quality of the comparison.")
        # closest reach's id
        closest_reach_id = data_l4.groups['reaches']['reach_id'][index]
        # 4.2 SWOT - Time data
        time_swot = data_l4.groups["consensus"]['time_int'][index]
        time_swot_filtered = time_swot[mask_fill_values] # again masking the missing values
        if time_swot_filtered.size != consensus_q_swot_filtered.size:
            print("Error: [3.1] The time and discharge arrays don't have the same size after filtering the missing values. Please check the data and try again.")
        # converting time from int to datetime
        epoch = np.datetime64('2000-01-01')
        datetime_swot = epoch + time_swot_filtered.astype('timedelta64[s]')
        # 4.3 SWOT - 
        # 4.3 GRDC - Runoff data
        runoff_grdc = station_data_2023_2025["runoff_mean"]  # metadata: print(runoff)
                                                # values: print(runoff.values)
        ### *** HERE: use NaN values to fill everywhere swot doesnt contain data compared to grdc (with pandas??)
  
        ## Section 5 : Data visualization
        # 5.1 Plot comparison of in-situ with grdc 
        if plot_discharge_comparison == 1:
            if station_id in list_stations_plot:
                
                # making the graph for a specific period (or not)
                if use_target_date_filter: 
                    ### HERE: use NaN values to fill everywhere swot doesnt contain data compared to grdc (with pandas??)
                    # 5.3 Slicing SWOT data for the right time period
                    target_datetime_inf = np.datetime64(target_date_inf, 'D')
                    target_datetime_sup = np.datetime64(target_date_sup, 'D')
                    index_target_datetime_period = np.where((datetime_swot >= target_datetime_inf) & (datetime_swot <= target_datetime_sup))
                    consensus_q_comparison = consensus_q_swot_filtered[index_target_datetime_period]     
                    datetime_comparison = datetime_swot[index_target_datetime_period]
                    # 5.4 Slicing GRDC data for the right time period
                    runoff_grdc_comparison = runoff_grdc.sel(time=slice(target_date_inf,target_date_sup)) # slicing the entire data to keep values between X and Y dates
                else: 
                    datetime_comparison = datetime_swot
                    consensus_q_comparison = consensus_q_swot_filtered
                    runoff_grdc_comparison = runoff_grdc

                # printing some information related to the nearest reach found
                #print(f"Distance of the closest and data-containing reach to station {station_id} is {distance}")
                #print(f"The closest and data-containing reach's id is {closest_reach_id}")
                #print(f"The coordinates of the closest and data-containing reach are ({data_l4.groups['reaches']['y'][index]},{data_l4.groups['reaches']['x'][index]})")
                plt.figure(figsize=(12, 6))
                plt.plot(datetime_comparison,consensus_q_comparison, marker='o', markersize=5, label='SWOT discharge', color='darkorange', markeredgecolor='white', markeredgewidth=0.4)
                plt.plot(runoff_grdc_comparison.time.values, runoff_grdc_comparison.values, marker='o', markersize=5, label='In situ', color='tab:blue', alpha=1, markeredgecolor='white', markeredgewidth=0.4)

                plt.title(f"In situ runoff measurements of station {station_id} and the SWOT L4 consensus discharge \nof it's nearest corresponding reach {closest_reach_id}") # SELECT title
                plt.xlabel('Time (UTC)') 
                plt.ylabel(r'Discharge (m$^3$/s)')

                plt.legend()
                plt.grid(True)

                if use_target_date_filter:
                    fig_name = f'discharge_comparison_station{station_id}_r{closest_reach_id}_{target_date_inf}_{target_date_sup}.png' # SELECT file name
                else:
                    fig_name = f'discharge_comparison_station{station_id}_r{closest_reach_id}.png' # SELECT file name
                plt.savefig(f'/obs/ecastonguay/scripts/figures/{fig_name}', dpi=400, bbox_inches='tight')
                print(f"Figure of station {station_id} saved in /obs/ecastonguay/scripts/figures") # *** currently no error message if one of the station required to be plotted isn't in the GRDC data -> print(f"There is no correspondence in the GRDC in situ files for station id {station_id}")
                plt.show()  
        break # while ii < k_neighbors:

# 4.2 Plot all discharge data for a specific reach, without in situ data comparison
if plot_reach_discharge == 1:
    reach_id_values = data_l4.groups['reaches']['reach_id'][:]
    selected_reach_index_array = np.where(reach_id_values == selected_reach_id) # find the index of the reach i'm looking for (the array contains the index)
                                                                                # array within array; [0][0]
    if (len(selected_reach_index_array[0]) == 0):
        print("Error: [4.2] Reach id not found in the list of reach ids. Please check the reach id and try again.")
    else:
        # discharge
        selected_reach_index = selected_reach_index_array[0][0] # [tested]
        consensus_q_variable = data_l4.groups["consensus"]['consensus_q'] # <class 'netCDF4.Variable'>
        discharge_selected_reach = consensus_q_variable[selected_reach_index]
        # print reach's coordinates
        print(f"The coordinates of the reach are ({data_l4.groups['reaches']['y'][selected_reach_index]},{data_l4.groups['reaches']['x'][selected_reach_index]})")
        # masking the missing discharge values
        mask_fill_q = discharge_selected_reach != consensus_q_variable.missing_value # *** use numpy masked array instead?
        discharge_selected_reach = discharge_selected_reach[mask_fill_q]
        # finding the corresponding time
        time_selected_reach = data_l4.groups["consensus"]['time_int'][selected_reach_index]
        time_selected_reach = time_selected_reach[mask_fill_q] # again masking the missing values
        # converting time from int to datetime
        epoch = np.datetime64('2000-01-01')
        datetime_selected_reach = epoch + time_selected_reach.astype('timedelta64[s]') 
        # slicing SWOT data for the right time period
        if use_target_date_filter:
            target_datetime_inf = np.datetime64(target_date_inf, 'D')
            target_datetime_sup = np.datetime64(target_date_sup, 'D')
            index_target_datetime_period = np.where((datetime_selected_reach >= target_datetime_inf) & (datetime_selected_reach <= target_datetime_sup))
            datetime_plot = datetime_selected_reach[index_target_datetime_period]
            discharge_plot = discharge_selected_reach[index_target_datetime_period]  
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

    
# Visualizer of the SWOT discharge L4 data and GRDC in situ values for data analysis
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
continent = "na" # SELECT a continent (africa: af, asia: as, europe: eu, north_america: na, south_america: sa, oceania: oc)
list_of_station_id = [4127800] # SELECT stations in GRDC database to compare with SWOT data. Enter their stations id in int format.
target_date_inf = "2023-06-29" # SELECT a time period for the graphs
target_date_sup = "2024-09-21"

## Section 1 : Extracting the dataset from the SWOT continent file
# This code loads the single selected continent file
dir_l4 = "/obs/ecastonguay/swot_data/L4_discharge/"
if continent not in ['af', 'as', 'eu', 'na', 'sa', 'oc']:
    raise ValueError("Error: continent must be one of the following: 'af' (Africa), 'as' (Asia), 'eu' (Europe), 'na' (North America), 'sa' (South America), 'oc' (Oceania)")    
file_suffix = "_sword_v16_SOS_results_unconstrained_20230502T204408_20250502T204408_20251219T163700.nc"
single_file_name = dir_l4 + continent + file_suffix
data_l4 = nc.Dataset(single_file_name)

## Section 2 : Reading data
# 2.1 Loop for reading all the GRDC station_id 
dir_grdc_prefix = "/obs/ecastonguay/grdc_data/"
ext_grdc = "*.nc"
path_grdc = os.path.join(dir_grdc_prefix,continent,ext_grdc)
list_files_grdc = glob.glob(path_grdc)
for station_id in list_of_station_id:
    found = 0
    for grdc_files in list_files_grdc:
        data_grdc = xr.open_dataset(grdc_files, engine="netcdf4")
        found_station_id = data_grdc["id"].values[0]
        if found_station_id == station_id: 
            found = 1
            # 2.2 Finding a match between the GRDC gauge and the SWOT data at reach level
            # get grdc (lat,lon) coordinates of gauge
            x_found_station = data_grdc["geo_x"].values[0] # x (lat) at the found station
            y_found_station = data_grdc["geo_y"].values[0]
            # find the corresponding reach in the continent-level swot data (looking for the reach with the closest coordinates to the station) (consensus_q only available at reach-level)
            reach_group = data_l4.groups['reaches']
            x_coordinates_values = reach_group['x'][:] # all x coordinates in swot/l4 data for the na continent
            y_coordinates_values = reach_group['y'][:]
            # K-D tree to search nearest neighbor
            stacked_xy = np.vstack((x_coordinates_values,y_coordinates_values)).T
            distance, index = KDTree(stacked_xy).query([x_found_station, y_found_station],k=1)
            if (distance > 100):    # even if the closest reach is quite far, KDTree will still find a match. therefore we need to make sure the distance found isn't too far
                                    # ADJUST maximum distance
                print(f"Warning: The nearest reach associated with the station {station_id} is quite far from it's location. This might have an impact on the quality of the comparison.")
            ## Section 3 : Fetching the discharge data from both sources
            # 3.1 Fetching consensus_q data from the swot file
            consensus_group = data_l4.groups["consensus"] # choosing a group
            consensus_q_variable = consensus_group['consensus_q'] # choosing a variable variable
            consensus_q_values = consensus_q_variable[:] # getting all the data
            consensus_q_swot = consensus_q_variable[index] # getting the discharge data at the selected reach
            # masking the missing values
            mask_fill_values = consensus_q_swot != consensus_q_variable.missing_value
            consensus_q_swot_filtered = consensus_q_swot[mask_fill_values]
            # finding the corresponding time
            time_swot = consensus_group['time_int'][index]
            time_swot_filtered = time_swot[mask_fill_values] # again masking the missing values
            # converting time from int to datetime
            epoch = np.datetime64('2000-01-01')
            datetime_swot = epoch + time_swot_filtered.astype('timedelta64[s]')
            # 3.2 Fetching runoff/discharge data from the grdc file
            runoff_grdc = data_grdc["runoff_mean"]  # metadata: print(runoff)
                                                    # values: print(runoff.values)
            # 3.3 Slicing SWOT data for the right time period
            target_datetime_inf = np.datetime64(target_date_inf, 'D')
            target_datetime_sup = np.datetime64(target_date_sup, 'D')
            index_target_datetime_period = np.where((datetime_swot >= target_datetime_inf) & (datetime_swot <= target_datetime_sup))
            consensus_q_comparison = consensus_q_swot_filtered[index_target_datetime_period]     
            datetime_comparison = datetime_swot[index_target_datetime_period]
            # 3.4 Slicing GRDC data for the right time period
            runoff_grdc_comparison = runoff_grdc.sel(time=slice(target_date_inf,target_date_sup)) # slicing the entire data to keep values between X and Y dates
            ## Section 4 : Data visualization
            # Plot comparison of in-situ with grdc
            x = 0 # SELECT 1 to plot, 0 to ignore
            if x == 1:
                plt.figure(figsize=(12, 6))
                plt.plot(datetime_comparison,consensus_q_comparison, marker='o', label='SWOT discharge', color='orange')
                plt.plot(runoff_grdc_comparison.time.values, runoff_grdc_comparison.values, marker='o', label='in situ', color='blue')

                plt.title(f"SWOT L4 consensus discharge compared with in situ measurements for reach {selected_reach_id}") # SELECT title
                plt.xlabel('Time (UTC)') 
                plt.ylabel('Consensus discharge (m^3/s)')

                plt.legend()
                plt.grid(True)

                fig_name = f'in_situ_discharge_comparison_r{selected_reach_id}.png'
                plt.savefig(f'/obs/ecastonguay/scripts/figures/{fig_name}', dpi=400, bbox_inches='tight')
                print("Figure saved in /obs/ecastonguay/scripts/figures")
                plt.show()  

            break
    if found == 0:
        print(f"There is no correspondence in the GRDC in situ files for station id {station_id}")


## Section X : Graph of the SWOT data only
selected_reach_id = 74210000201 # 74210000201 for test - article 3, fig 2a; reach on mississippi near bâton rouge
selected_reach_index_array = np.where(reach_id_values == selected_reach_id) # find the index of the reach i'm looking for (the array contains the index)
                                                                            # array within array; [0][0]
if (len(selected_reach_index_array[0]) == 0):
    print("Error: reach id not found in the list of reach ids. Please check the reach id and try again.")
else:
    selected_reach_index = selected_reach_index_array[0][0] # [tested]
    discharge_selected_reach = consensus_q[selected_reach_index]
    # masking the missing values
    mask_fill_values = discharge_selected_reach != consensus_q.missing_value
    discharge_selected_reach = discharge_selected_reach[mask_fill_values]
    # finding the corresponding time
    time_selected_reach = consensus_group['time_int'][selected_reach_index]
    time_selected_reach = time_selected_reach[mask_fill_values] # again masking the missing values
    # converting time from int to datetime
    epoch = np.datetime64('2000-01-01')
    datetime_selected_reach = epoch + time_selected_reach.astype('timedelta64[s]') # 
    
    ## Section 4 : Data visualization
    # 4.1 In-situ of consensus_q for entire available period
    x = 0 # SELECT 1 to plot, 0 to ignore
    if x == 1:
        plt.figure(figsize=(12, 6))
        plt.plot(datetime_selected_reach,discharge_selected_reach, marker='o', label='SWOT discharge', color='orange')

        plt.title(f"SWOT L4 consensus discharge and in situ measures for reach {selected_reach_id}")
        plt.xlabel('Time (UTC)') 
        plt.ylabel('Consensus discharge (m^3/s)')

        plt.legend()
        plt.grid(True)

        fig_name = f'all_consensus_q_r{selected_reach_id}.png'
        plt.savefig(f'/obs/ecastonguay/scripts/figures/{fig_name}', dpi=400, bbox_inches='tight')
        print("Figure saved in /obs/ecastonguay/scripts/figures")
        plt.show()  

    # 4.2 Comparing in-situ with grdc # *** réduire fréquence entre les points???
    # slicing grdc data for 06-2023 to 09-2024 period
    runoff_grdc = runoff_grdc.sel(time=slice("2023-06-29","2024-09-21")) # slicing the entire data to keep values between X and Y dates
    # slicing swot data for 06-2023 to 09-2024 period
    target_date_inf = np.datetime64('2023-06-29', 'D')
    target_date_sup = np.datetime64('2024-09-21', 'D')
    index_target_period = np.where((datetime_selected_reach >= target_date_inf) & (datetime_selected_reach <= target_date_sup))
    discharge_comparison = discharge_selected_reach[index_target_period]     
    datetime_comparison = datetime_selected_reach[index_target_period]
    # plot
    x = 0 # SELECT 1 to plot, 0 to ignore
    if x == 1:
        plt.figure(figsize=(12, 6))
        plt.plot(datetime_comparison,discharge_comparison, marker='o', label='SWOT discharge', color='orange')
        plt.plot(runoff_grdc.time.values, runoff_grdc.values, marker='o', label='in situ', color='blue')

        plt.title(f"SWOT L4 consensus discharge compared with in situ measurements for reach {selected_reach_id}") # SELECT title
        plt.xlabel('Time (UTC)') 
        plt.ylabel('Consensus discharge (m^3/s)')

        plt.legend()
        plt.grid(True)

        fig_name = f'in_situ_discharge_comparison_r{selected_reach_id}.png'
        plt.savefig(f'/obs/ecastonguay/scripts/figures/{fig_name}', dpi=400, bbox_inches='tight')
        print("Figure saved in /obs/ecastonguay/scripts/figures")
        plt.show()  
    
  

## Section 5 : Opening GRDC data 


## Trash
# Old code about selecting with reach_id
"""## Section 3 : Choosing a reach
selected_reach_id = 74210000201 # 74210000201 for test - article 3, fig 2a; reach on mississippi near bâton rouge
selected_reach_index_array = np.where(reach_id_values == selected_reach_id) # find the index of the reach i'm looking for (the array contains the index)
                                                                            # array within array; [0][0]
if (len(selected_reach_index_array[0]) == 0):
    print("Error: reach id not found in the list of reach ids. Please check the reach id and try again.")
else:
    selected_reach_index = selected_reach_index_array[0][0] # [tested]
    discharge_selected_reach = consensus_q[selected_reach_index]
    # masking the missing values
    mask_fill_values = discharge_selected_reach != consensus_q.missing_value
    discharge_selected_reach = discharge_selected_reach[mask_fill_values]
    # finding the corresponding time
    time_selected_reach = consensus_group['time_int'][selected_reach_index]
    time_selected_reach = time_selected_reach[mask_fill_values] # again masking the missing values
    # converting time from int to datetime
    epoch = np.datetime64('2000-01-01')
    datetime_selected_reach = epoch + time_selected_reach.astype('timedelta64[s]') # 
    
    ## Section 4 : Data visualization
    # 4.1 In-situ of consensus_q for entire available period
    x = 0 # SELECT 1 to plot, 0 to ignore
    if x == 1:
        plt.figure(figsize=(12, 6))
        plt.plot(datetime_selected_reach,discharge_selected_reach, marker='o', label='SWOT discharge', color='orange')

        plt.title(f"SWOT L4 consensus discharge and in situ measures for reach {selected_reach_id}")
        plt.xlabel('Time (UTC)') 
        plt.ylabel('Consensus discharge (m^3/s)')

        plt.legend()
        plt.grid(True)

        fig_name = f'all_consensus_q_r{selected_reach_id}.png'
        plt.savefig(f'/obs/ecastonguay/scripts/figures/{fig_name}', dpi=400, bbox_inches='tight')
        print("Figure saved in /obs/ecastonguay/scripts/figures")
        plt.show()  

    # 4.2 Comparing in-situ with grdc # *** réduire fréquence entre les points???
    # slicing grdc data for 06-2023 to 09-2024 period
    runoff_grdc = runoff_grdc.sel(time=slice("2023-06-29","2024-09-21")) # slicing the entire data to keep values between X and Y dates
    # slicing swot data for 06-2023 to 09-2024 period
    target_date_inf = np.datetime64('2023-06-29', 'D')
    target_date_sup = np.datetime64('2024-09-21', 'D')
    index_target_period = np.where((datetime_selected_reach >= target_date_inf) & (datetime_selected_reach <= target_date_sup))
    discharge_comparison = discharge_selected_reach[index_target_period]     
    datetime_comparison = datetime_selected_reach[index_target_period]
    # plot
    x = 0 # SELECT 1 to plot, 0 to ignore
    if x == 1:
        plt.figure(figsize=(12, 6))
        plt.plot(datetime_comparison,discharge_comparison, marker='o', label='SWOT discharge', color='orange')
        plt.plot(runoff_grdc.time.values, runoff_grdc.values, marker='o', label='in situ', color='blue')

        plt.title(f"SWOT L4 consensus discharge compared with in situ measurements for reach {selected_reach_id}") # SELECT title
        plt.xlabel('Time (UTC)') 
        plt.ylabel('Consensus discharge (m^3/s)')

        plt.legend()
        plt.grid(True)

        fig_name = f'in_situ_discharge_comparison_r{selected_reach_id}.png'
        plt.savefig(f'/obs/ecastonguay/scripts/figures/{fig_name}', dpi=400, bbox_inches='tight')
        print("Figure saved in /obs/ecastonguay/scripts/figures")
        plt.show()  """

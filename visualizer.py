# Visualizer for data analysis
# Author: Eve Castonguay, LIRA (CNRS)
# Creation date: 2026-05-07 [YYYY-MM-DD]
# Version 0.1: AAAA-MM-JJ

# Imports
from datetime import datetime
import os
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import glob # because geopandas does not recognize wildcards in file paths, we use glob

## Section 1 : Data extraction (01/2024, mississippi reach)
# 1.1 River average vector product. Creation of a list of .shp files to read
dir_avg = "/obs/ecastonguay/swot_data/RiverAvg_D/"
ext_avg = "*.shp" # in theory this file only contains the coordinates of the centerline of each reach, but reading it with geopandas will extract geometry, attributes and all other info
path_avg = os.path.join(dir_avg,ext_avg)
list_files_avg = glob.glob(path_avg) # list of files to read. use of glob provides list of files corresponding to pattern. 

# 1.2 River single pass vector product - for reaches or nodes
dir_sp = "/obs/ecastonguay/swot_data/RiverSP_C/" # directory containing the C version
reach_or_node = 1 # SELECT 0 for node, 1 for reach
if reach_or_node == 0:
    file_identifier = "node_level"
    dir_sp = os.path.join(dir_avg,file_identifier)
elif reach_or_node == 1:
    file_identifier = "reach_level"
    dir_sp = os.path.join(dir_sp,file_identifier)
else: 
    raise ValueError("reach_or_node must be 0 (for node) or 1 (for reach)")
ext_sp = "*.shp" # (.shp, .shx, .dbf, .prj, .shp.xml)
path_sp = os.path.join(dir_sp,ext_sp)
list_files_sp = glob.glob(path_sp) # list of files to read. 

# 1.3 In-situ discharge data from GRDC
dir_grdc = "/obs/ecastonguay/grdc_data/"
station_grdc = "vicksburg_ms_daily" # temporary file
file_grdc = "GRDC-Daily.nc" # netcdf file
path_grdc = os.path.join(dir_grdc,station_grdc,file_grdc)
# print("Path to in-situ discharge data from GRDC: ", path_grdc) # /obs/ecastonguay/grdc_data/vicksburg_ms_daily/GRDC-Daily.nc

## Section 2 : Reading data
# SELECT a reach
selected_reach_id = '74210000201' # for test - article 3, fig 2a; reach on mississippi near bâton rouge
# Fill values
fill_value_discharge = -999999999999.0 

'''# 2.1 Reading RiverAvg
# Attribute names
time = 't_avg'
time_string = 't_str_avg'
discharge_unconstrained = 'dsg_avg'
discharge_constrained = 'dsc_avg'
# Extracting the information from each file
"""     
Example of display results:
      reach_id      p_lat      p_lon     river_name  n_passes  n_valid_p  ...    dsc_hmax_u  reach_q  reach_q_b    geoid_hght    geoid_slop                                           geometry
0  74225000323  33.859269 -96.604646        no_data         3          0  ... -1.000000e+12        3  268435456 -1.000000e+12 -1.000000e+12  LINESTRING (-96.57268 33.81966, -96.57287 33.8...
1  74226000013  33.947431 -96.589088        no_data         3          0  ... -1.000000e+12        3  268435456 -1.000000e+12 -1.000000e+12  LINESTRING (-96.60457 33.89845, -96.60424 33.8...
2  74226000023  34.021482 -96.592536        no_data         3          0  ... -1.000000e+12        3  268435456 -1.000000e+12 -1.000000e+12  LINESTRING (-96.61609 34.00173, -96.61628 34.0...
3  74226000031  34.049905 -96.565016  Washita River         3          2  ... -1.000000e+12        0          0 -2.707580e+01  4.290400e-06  LINESTRING (-96.57932 34.04564, -96.57899 34.0...
4  74226000041  34.053360 -96.539064  Washita River         3          2  ... -1.000000e+12        0          0 -2.705920e+01 -8.258400e-06  LINESTRING (-96.54943 34.05208, -96.54923 34.0...
"""
data_list_avg = [] # empty list to store the attribute values (in a dataframe)
for file_avg in list_files_avg: # *** if empty...
    gdf_avg = gpd.read_file(file_avg) # reading the files with geopandas (will contain both geometry and attributes); returns a GeoDataFrame
    gdf_avg_reach = gdf_avg[gdf_avg['reach_id'] == selected_reach_id] # selecting the reach_id 74287900061. this creates a new geodataframe. *** what if multiple identical reaches? there should be unique reaches within a basin/granule
    if len(gdf_avg_reach) > 0: # making sure the reach was found within the granule file
        df_subset = gdf_avg_reach[[time, time_string, discharge_unconstrained, discharge_constrained]].copy()    # .copy to avoid the SettingWithCopyWarning when modifying the subset dataframe. better to make an independent copy
                                                                                                    # at this point, the data isn't a geodataframe but a simple dataframe (since we haven't selected a single column, it isn't a Pandas Series)  
                                                                                                    # single discharge values (c. or unc.) for 21-day average                                                               
        if not (df_subset[discharge_unconstrained].item() == fill_value_discharge and df_subset[discharge_constrained].item() == fill_value_discharge): # making sure discharges are not a fill value. 
            # *** check later, might cause problems... how to deal with null values? (thinking about the graphs)
            data_list_avg.append(df_subset) # the copy is of the same type as the 'original'; we append it to the list
if not data_list_avg:
    print("List is empty") # *** do something with empty list or else it will raise error
df_dsc_dsg_time = pd.concat(data_list_avg, ignore_index=True) # the index is of no utility. type: DataFrame
#print(df_dsc_dsg_time) # display the dataframe with time, dsg and dsc values for the selected reach_id across all granules. should be 21 rows (one per day) if all granules contain the reach_id, but some may be missing if the reach_id is not present in some granules or if the discharge values are null.
"""
# How to select data from a geodataframe?
test = gdf_avg['river_name']    # selecting a column returns a Pandas series, with index numbers and values. simply select with [i]
                                # print(type(test)) returns <class 'pandas.core.series.Series'>
selected1 = test.iloc[0]                            # selecting the first value of the series (index 0) returns a string, which is the name of the river 
selected2 = test[test.str.contains('Mississippi')]  # returns a series with index numbers and values 
selected3 = test.values[0]                          # returns the value (str in this case); same as #1
print(selected2, type(selected2)) 
"""
'''
# 2.2 Reading RiverSP
data_list_sp = [] 
for file_sp in list_files_sp: # *** if empty... # *** also have to deal with the node files. separate code? 
    gdf_sp = gpd.read_file(file_sp) 
    gdf_sp_reach = gdf_sp[gdf_sp['reach_id'] == selected_reach_id] # selecting the reach_id 74287900061. this creates a new geodataframe. *** what if multiple identical reaches? should not happen for a single pass
    if len(gdf_sp_reach) > 0: # making sure the reach was found within the granule file
        df_subset = gdf_sp_reach[['time', 'dschg_c']].copy()    # .copy to avoid the SettingWithCopyWarning when modifying the subset dataframe. better to make an independent copy
                                                                # at this point, the data isn't a geodataframe but a simple dataframe (since we haven't selected a single column, it isn't a Pandas Series)  
                                                                #                                                                
        if not (df_subset['dschg_c'].item() == fill_value_discharge ): # making sure discharges are not a fill value.
            # *** check later, might cause problems... how to deal with fill values? (thinking about the graphs)
            data_list_sp.append(df_subset) # the copy is of the same type as the 'original'; we append it to the list
if not data_list_sp:
    print("List is empty") # *** do something with empty list or else it will raise error
df_dschg_c_time = pd.concat(data_list_sp, ignore_index=True) # the index is of no utility. type: DataFrame
#print(df_dsc_dsg_time) # display the dataframe with time, dsg and dsc values for the selected reach_id across all granules. should be 21 rows (one per day) if all granules contain the reach_id, but some may be missing if the reach_id is not present in some granules or if the discharge values are null.


# 2.3 Reading in-situ discharge data from GRDC
#ds = xr.open_dataset(complete_path_avg)




## Section 3 : Data visualization
'''# 3.1 Comparing constrained and unconstrained averaged discharge (mississippi reach)
# Plot
time_seconds = df_dsc_dsg_time[time]
diff = (datetime(2000,1,1) - datetime(1970,1,1)).total_seconds()
time_plot = pd.to_datetime(time_seconds + diff, unit='s') # *** not exact, seconds missing, check metadata. type: pandas datetime object
dsc_plot = df_dsc_dsg_time[discharge_constrained]
dsg_plot = df_dsc_dsg_time[discharge_unconstrained]

plt.figure(figsize=(12, 6))
plt.plot(time_plot, dsc_plot, marker='o', label='constrained')
plt.plot(time_plot, dsg_plot, marker='s', label='unconstrained')

plt.xlabel('Time (UTC)') # *** ajouter jour ds temps UTC
plt.ylabel('Discharge (m^3/s)')
plt.title('Comparison of unconstrained and constrained discharge data for reach 74210000201') # add: Mssissippi River, near Baton Rouge
plt.legend()
plt.grid(True)

plt.savefig('/obs/ecastonguay/scripts/figures/dsc_dsg_r74210000201.png', dpi=400, bbox_inches='tight')
print("Figure saved in /obs/ecastonguay/scripts/figures")
plt.show()'''

# 3.2 Comparing in-situ with grdc (node-level)
# Plot
time_seconds = df_dschg_c_time['time']
diff = (datetime(2000,1,1) - datetime(1970,1,1)).total_seconds()
time_plot_sp = pd.to_datetime(time_seconds + diff, unit='s') # *** not exact, seconds missing, check metadata. type: pandas datetime object
dsc_plot_sp = df_dschg_c_time['dschg_c']
plt.figure(figsize=(12, 6))
plt.plot(time_plot_sp, dsc_plot_sp, marker='o', label='SWOT single pass')
# add in situ value

plt.xlabel('Time (UTC)') # *** ajouter jour ds temps UTC
plt.ylabel('Discharge (m^3/s)')
plt.title('SWOT single pass consensus discharge and in situ measures for reach 74210000201') # add: Mssissippi River, near Baton Rouge
plt.legend()
plt.grid(True)

plt.savefig('/obs/ecastonguay/scripts/figures/dschg_c_in_situ_r74210000201.png', dpi=400, bbox_inches='tight')
print("Figure saved in /obs/ecastonguay/scripts/figures")
plt.show()

# 3.3 comparing averaged reach wih non-averaged reach (reach-level)




## Trash
"""
# Select reach id that correspond to (lon: -90.9058, lat: 32.315) (Vicksburg station, MS)
lon_vicksburg = -90.9058
lat_vicksburg = 32.315  
selected_reach_vicksburg = gdf_avg[(gdf_avg['p_lon'] == lon_vicksburg) & (gdf_avg['p_lat'] == lat_vicksburg)]
print(selected_reach_vicksburg) # empty geodatafram
"""
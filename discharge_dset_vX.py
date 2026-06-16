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


version = 'v8_1'
## Section 1 : Setting some variables
# 1.1 Continents
continent_list = ['na', 'af', 'as', 'eu', 'sa', 'oc']
# 1.2 Period during which SWOT has data (2023-03-29 to 2025-05-02)
swot_start = '2023-03-29'
swot_end = '2025-05-02'
# 1.3 Empty dictionnaries for the DataArrays
# grdc
runoff_darrays_g = {} 
geox_darrays_g = {}
geoy_darrays_g = {}
area_darrays_g = {}
river_darrays_g = {}
country_darrays_g = {}
# swot
dschg_darrays_s = {}
geox_darrays_s = {}
geoy_darrays_s = {}
id_darrays_s = {}
w_darrays_s = {} # from sword
area_darrays_s = {} # from sword
r_id_up_darrays_s = {} # from sword
r_id_dn_darrays_s = {} # from sword
river_darrays_s = {} # from sword

## Section 2 : Loop over continents 
for i_continent in continent_list: 
    
    ## Section 3 : Swot extraction
    # 3.1 Reading
    dir_l4 = "/obs/ecastonguay/swot_data/L4_discharge/"
    file_suffix = "_sword_v16_SOS_results_unconstrained_20230502T204408_20250502T204408_20251219T163700.nc"
    single_file_name = dir_l4 + i_continent + file_suffix 
    data_swot = nc.Dataset(single_file_name) # level 4 satellite data
    # 3.2 Extracting
    r_id_swt = data_swot.groups['reaches']['reach_id'][:]
    time_swt = data_swot.groups["consensus"]['time_int'][:]
    geox_swt = data_swot.groups["reaches"]['x'][:]
    geoy_swt = data_swot.groups["reaches"]['y'][:]
    mv_swot = data_swot.groups["consensus"]['consensus_q'].missing_value
    dschg_swt = data_swot.groups["consensus"]['consensus_q'][:] # shape (38048,). if no data at a reach, contains array([-1.e+12]). 
    assert dschg_swt.shape == r_id_swt.shape == time_swt.shape # dschhg is a list of arrays, same len as r_ids
    # 3.3 In DataArrays (for better access)
    geox_darray_full_s = xr.DataArray( 
        data=geox_swt,
        dims=["reach_id"], 
        coords=dict(reach_id=r_id_swt,),
        name="x coordinate swot")
    geoy_darray_full_s = xr.DataArray(
        data=geoy_swt,
        dims=["reach_id"], 
        coords=dict(reach_id=r_id_swt,),
        name="y coordinate swot")
    # 3.4 Reformatting the swot discharge data into a darray
    time_dim = pd.date_range(start=swot_start, end=swot_end, freq='D') # 766. ndarray-like of datetime64 data. dtype='datetime64[us].
    time_dim_len = len(time_dim)
    dschg_swt_rfm = utils.dschg_darray(dschg_swt, time_swt, r_id_swt, time_dim, mv_swot)
    dschg_darray_full_s = xr.DataArray( 
        data=dschg_swt_rfm,
        dims=["reach_id","time"], # name of the dimensions
        coords=dict(
            reach_id=r_id_swt,
            time=time_dim,
        ),
        attrs=dict(
            description="Consensus_q from SWOT",
            units="m3/s",
            missing_value=np.nan
        ),
        name="swot discharge"
    )

    ## Section 4 : Sword extraction
    # 4.1 Reading
    dir_swr = "/obs/ecastonguay/sword_data/netcdf_v16" # sword v16
    file_swr = i_continent + "_sword_v16.nc"
    path_swr = os.path.join(dir_swr,file_swr)
    data_swr = nc.Dataset(path_swr) # open the netcdf file
    # 4.2 Extracting
    r_id_swr = data_swr["reaches"]["reach_id"][:]
    facc_swr = data_swr["reaches"]["facc"][:]
    river_swr = data_swr["reaches"]["river_name"][:]
    width_swr = data_swr["reaches"]["width"][:]
    # 4.3 In DataArrays
    facc_darray_full_s = xr.DataArray(
        data=facc_swr,
        dims=["reach_id"], # name of the dimensions
        coords=dict(reach_id=r_id_swr,),
        name="facc sword") 
    width_darray_full_s = xr.DataArray(
        data=width_swr,
        dims=["reach_id"], # name of the dimensions
        coords=dict(reach_id=r_id_swr,),
        name="river name sword") 
    river_darray_full_s = xr.DataArray(
        data=river_swr, # contains ndarrays()
        dims=["reach_id"], # name of the dimensions
        coords=dict(reach_id=r_id_swr,),
        name="river name sword") 
    # 4.4 Pretreatment on sword names
    river_clean_full_s = []
    index_clean_full_s = [] # sword
    for idx_river, long_name in enumerate(river_darray_full_s.values):
        parts = long_name.split("; ")
        for p in parts:
            river_clean_full_s.append(re.sub(r'\s*\(.*?\)', '', p).lower().strip())
            index_clean_full_s.append(idx_river)
        
    ## Section 5 : Grdc extraction
    # 5.1 Reading
    dir_grdc_prefix = "/obs/ecastonguay/grdc_data/"
    file_nc = i_continent + ".nc"
    path_nc = os.path.join(dir_grdc_prefix,i_continent,file_nc)
    file_json = "stationbasins_" + i_continent + ".geojson"
    path_json = os.path.join(dir_grdc_prefix,i_continent,file_json)
    # 5.2 Extracting : discharge file
    data_grdc = xr.open_dataset(path_nc, engine="netcdf4") # <xarray.Dataset>
    data_23_25_g = data_grdc.sel(time=slice(swot_start,swot_end)) # DataSet. slice here includes the last day
    data_23_25_g = data_23_25_g.transpose() # swap dimensions to have (id,time) instead of (time,id)
    # 5.3 In DataArrays : discharge file
    dschg_darray_g = data_23_25_g['runoff_mean']      
    country_darray_g = data_23_25_g['country'] 
    # list of the stations_id of grdc  
    list_station_id = dschg_darray_g["id"].values 
    # 5.4 Extracting : watershed file
    data_ws = gpd.read_file(path_json) # watershed
    # pandas series
    station_id_gdf = data_ws['grdc_no'] 
    area_gdf = data_ws['area_calc']
    river_gdf = data_ws['river']
    geox_gdf = data_ws['long_pp']
    geoy_gdf = data_ws['lat_pp']
    # 5.5 In DataArrays : watershed file
    area_darray_g = xr.DataArray(
        data=area_gdf,
        dims=["id"], 
        coords=dict(id=station_id_gdf,),
        attrs=dict(description="Watershed areas (km2)"),
        name="watershed areas")
    river_darray_g = xr.DataArray(
        data=river_gdf,
        dims=["id"], 
        coords=dict(id=station_id_gdf,),
        attrs=dict(description="GRDC river names"),
        name="river names")
    geox_darray_g = xr.DataArray(
        data=geox_gdf,
        dims=["id"], 
        coords=dict(id=station_id_gdf,),
        attrs=dict(description="x coordinates of the GRDC station"),
        name="geo x grdc")
    geoy_darray_g = xr.DataArray(
        data=geoy_gdf,
        dims=["id"], 
        coords=dict(id=station_id_gdf,),
        attrs=dict(description="y coordinates of the GRDC station"),
        name="geo y grdc")  

    ## Section 6 : Empty ndarrays for swot/sword data 
    # 6.1 Swot
    # discharge
    dschg_ndarray = np.full((len(list_station_id), time_dim_len), np.nan, dtype=np.float64) # dim (id, time)
    # lon
    x_ndarray = np.full((len(list_station_id)), np.nan, dtype=np.float64) # dim (id)
    # lat
    y_ndarray = np.full((len(list_station_id)), np.nan, dtype=np.float64) # dim (id)
    # reach id
    r_id_ndarray = np.full((len(list_station_id)), np.nan, dtype=np.float64) # dim (id)
    # 6.2 Sword
    # width
    w_ndarray = np.full((len(list_station_id)), np.nan, dtype=np.float64) # dim (id)
    # area
    facc_ndarray = np.full((len(list_station_id)), np.nan, dtype=np.float64) # dim (id)
    # reach id up
    r_id_up_ndarray = np.full((len(list_station_id)), np.nan, dtype=np.float64) # dim (id)
    # reach id down
    r_id_dn_ndarray = np.full((len(list_station_id)), np.nan, dtype=np.float64) # dim (id)
    # river name
    river_ndarray = np.full((len(list_station_id)), '', dtype=object) # dim (id)

    ## Section 7 : Loop on stations
    station_pos = 0
    no_match_dist_area = 0 # n of stations that werent matched due to area diff. (corresp function)
    no_match_name = 0 # n of stations that werent matched due to name diff. (corresp function)
    no_match_dschg = 0 # n of stations that werent matched due to empty discharge data (corresp function)
    no_match_corresp = 0 # n of stations that werent matched due to corresp function
    no_match_ws = 0 # n of stations that werent matched due to (y,x) watershed data
    for station_id in list_station_id:
        
        # 7.1 Station id in watershed data?
        if station_id not in geox_darray_g.id.values:
            station_pos += 1
            no_match_ws += 1
            continue
        
        # 7.2 Coordinates of station
        x_found_station = geox_darray_g.sel(id=station_id).item() # 4977015 -> "not all values found in index 'id'. Try setting the `method` keyword argument (example: method='nearest')."
        y_found_station = geoy_darray_g.sel(id=station_id).item() # [tested]

        ## Section 8 : Corresp. based on names, then distance and area
        result, reason = utils.corresp_name_dist_area_v5(station_id, river_clean_full_s, index_clean_full_s, r_id_swr, area_darray_g, river_darray_g, facc_darray_full_s, dschg_darray_full_s, x_found_station, y_found_station, geox_darray_full_s, geoy_darray_full_s)
  
        if result is not None:
            sel_r_id, name_river_s = result 
        else: 
            if reason == 'no match on name':
                no_match_name += 1
            if reason == 'no match on distance or area':
                no_match_dist_area += 1
            if reason == 'empty dschg data':
                no_match_dschg += 1
            no_match_corresp += 1
            station_pos += 1 
            continue

        ## Section 8 : Filling empty ndarrays with sword/swot data
        # 8.1 Discharge
        dschg_ndarray[station_pos] = dschg_darray_full_s.sel(reach_id=sel_r_id).values # [tested] where there is data the space will be filled, otherwise stays NaN

        # 8.2 Coordinates
        sel_r_x = geox_darray_full_s.sel(reach_id=sel_r_id)
        sel_r_y = geoy_darray_full_s.sel(reach_id=sel_r_id)
        x_ndarray[station_pos] = sel_r_x
        y_ndarray[station_pos] = sel_r_y
        
        # 8.3 Reach id
        r_id_ndarray[station_pos] = sel_r_id

        # 8.4 Width, facc, river name (here I assume that a reach id in swot will be found in the sword database also; this assumption never caused me problems)
        w_ndarray[station_pos] = width_darray_full_s.sel(reach_id=sel_r_id)
        facc_ndarray[station_pos] = facc_darray_full_s.sel(reach_id=sel_r_id)
        river_ndarray[station_pos] = name_river_s
        
        station_pos += 1 # end of the (for station_id in list_station_id:) loop

    assert station_pos == len(list_station_id) # at the end of the station loop, these should be equal

    print(f"Number of reaches that weren't matched due to corresp. algo: {no_match_corresp} for the {i_continent} continent \n{no_match_name} for: no match on name \n{no_match_dist_area} for: no match on area/distance \n{no_match_dschg} for: empty discharge")
    print(f"Number of reaches that weren't matched due to watershed data: {no_match_ws} for the {i_continent} continent.")

    ## Section 9 : Create DataArrays for SWOT/SWORD (ndarrays ordered by station_id)
    # 9.1 Discharge
    dschg_darray_s = xr.DataArray(
        data=dschg_ndarray,
        dims=["id","time"], # name of the dimensions
        coords=dict(
            id=list_station_id,
            time=time_dim,
        ),
        attrs=dict(
            description="Consensus_q from SWOT",
            units="m3/s",
        ),
        name="swot discharge"
    )
    # 9.2 Coordinates
    if len(x_ndarray) != len(y_ndarray):
        print(f"Error: [7.2] The x and y vector of all the selected reaches' coordinates doesn't match in size")
        #break # *** activate when continent loop is added!
    # x_y_ndarray = np.hstack((x_ndarray,y_ndarray)) # doesn't work for now
    # dataarrays
    geox_darray_s = xr.DataArray(
        data=x_ndarray,
        dims=["id"], # name of the dimensions
        coords=dict(
            id=list_station_id,
        ),
        attrs=dict(
            description="X coordinate of each selected reach in the SWOT data",
            units="degrees",
        ),
        name="swot x coordinate"
    )
    geoy_darray_s = xr.DataArray(
        data=y_ndarray,
        dims=["id"], # name of the dimensions
        coords=dict(
            id=list_station_id,
        ),
        attrs=dict(
            description="Y coordinate of each selected reach in the SWOT data",
            units="degrees",
        ),
        name="swot y coordinate"
    )
    # 9.3 Reach ids
    id_darray_s = xr.DataArray(
        data=r_id_ndarray,
        dims=["id"], # name of the dimensions
        coords=dict(
            id=list_station_id,
        ),
        attrs=dict(
            description="Id's of the selected reaches in the SWOT data"
        ),
        name="swot reach ids"
    )
    # 9.4 Width
    w_darray_s = xr.DataArray(
        data=w_ndarray,
        dims=["id"], # name of the dimensions
        coords=dict(
            id=list_station_id,
        ),
        attrs=dict(
            description="Average width for a SWOT reach (units: meters)."
        ),
        name="sword widths"
    )
    # 9.5 Area
    area_darray_s = xr.DataArray(
        data=facc_ndarray,
        dims=["id"], # name of the dimensions
        coords=dict(
            id=list_station_id,
        ),
        attrs=dict(
            description="Maximum flow accumulation value for a node or reach. Flow accumulation values are extracted from the MERIT Hydro dataset (Yamazaki et al., 2019) (units: square kilometers)."
        ),
        name="sword area" 
    )
    # 9.6 River name
    river_darray_s = xr.DataArray(
        data=river_ndarray.astype(str),
        dims=["id"], # name of the dimensions
        coords=dict(
            id=list_station_id,
        ),
        attrs=dict(
            description="All river names associated with a node or reach. If there are multiple names for a node or reach they are listed in alphabetical order and separated by a semicolon."
        ),
        name="sword river name" 
    )

    ## Section 10 : Make sure all grdc DataArrays have time dim of 766 days (between 2023-03-29 & 2025-05-02)
    if len(data_23_25_g.time) != len(dschg_darray_g.time):
        print(f"Error: [8.1] Both arrays are assumed to match in size. Please verify the code.")
    if len(dschg_darray_g.time) != time_dim_len:
        dschg_darray_g = dschg_darray_g.reindex(time=time_dim) # asia doesn't -> fill the time between 2024 & 2025 with NaN
        print(f"[8.1] DataArray of continent {i_continent} was reindexed") # missing values seem to be nan [tested]

    ## Section 11 : Append continent DataArrays to list
    # grdc
    runoff_darrays_g[i_continent] = dschg_darray_g
    geox_darrays_g[i_continent] = geox_darray_g
    geoy_darrays_g[i_continent] = geoy_darray_g
    area_darrays_g[i_continent] = area_darray_g
    river_darrays_g[i_continent] = river_darray_g
    country_darrays_g[i_continent] = country_darray_g
    # swot
    dschg_darrays_s[i_continent] = dschg_darray_s
    geox_darrays_s[i_continent] = geox_darray_s
    geoy_darrays_s[i_continent] = geoy_darray_s
    id_darrays_s[i_continent] = id_darray_s
    w_darrays_s[i_continent] = w_darray_s
    area_darrays_s[i_continent] = area_darray_s
    river_darrays_s[i_continent] = river_darray_s

# end of the (for i_continent in continent_list) loop 

## Section 12 : Concathenate continent DataArrays. These are individual DataArrays that contain values for all continents
# grdc
runoff_global_g = xr.concat(list(runoff_darrays_g.values()), dim='id') 
geox_global_g = xr.concat(list(geox_darrays_g.values()), dim='id')
geoy_global_g = xr.concat(list(geoy_darrays_g.values()), dim='id')
area_global_g = xr.concat(list(area_darrays_g.values()), dim='id')
river_global_g = xr.concat(list(river_darrays_g.values()), dim='id')
country_global_g = xr.concat(list(country_darrays_g.values()), dim='id')
# swot
dschg_global_s = xr.concat(list(dschg_darrays_s.values()), dim='id') 
geox_global_s = xr.concat(list(geox_darrays_s.values()), dim='id')
geoy_global_s = xr.concat(list(geoy_darrays_s.values()), dim='id')
id_global_s = xr.concat(list(id_darrays_s.values()), dim='id')
width_global_s = xr.concat(list(w_darrays_s.values()), dim='id')
area_global_s = xr.concat(list(area_darrays_s.values()), dim='id')
river_global_s = xr.concat(list(river_darrays_s.values()), dim='id')

## Section 13 : Create the DataSet and append all the variables (DataArrays) to it
# grdc
dset_global = runoff_global_g.to_dataset(name='dschg_global_g')
dset_global['geox_global_g'] = geox_global_g
dset_global['geoy_global_g'] = geoy_global_g
dset_global['area_global_g'] = area_global_g
dset_global['river_global_g'] = river_global_g
dset_global['country_global_g'] = country_global_g
# swot
dset_global['dschg_global_s'] = dschg_global_s
dset_global['geox_global_s'] = geox_global_s
dset_global['geoy_global_s'] = geoy_global_s
dset_global['id_global_s'] = id_global_s
dset_global['width_global_s'] = width_global_s
dset_global['area_global_s'] = area_global_s
dset_global['river_global_s'] = river_global_s

# Change the data type of the reaches id which is (int)
# dset_global["id_global_s"] = dset_global["id_global_s"].astype(int) # *** actually, this can't be done since a numpy array of type int cannot contain nans (they would be transformed in -9223372036854775808)

# Print
print(dset_global)

# Save to netcdf
dset_global.to_netcdf("/obs/ecastonguay/scripts/global_dset_" + version + ".nc")
## Example script for opening discharge data from the GRDC data in netCDF format
 
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

# Open netCDF file
dir_grdc = "/obs/ecastonguay/grdc_data/vicksburg_ms_daily/GRDC-Daily.nc"
data = xr.open_dataset(dir_grdc, engine="netcdf4")
"""
<xarray.Dataset> Size: 408kB
Dimensions:              (time: 33968, id: 1)
Coordinates:
  * time                 (time) datetime64[ns] 272kB 1931-10-01 ... 2024-09-29
  * id                   (id) int64 8B 4127800
Data variables:
    runoff_mean          (time, id) float32 136kB ...
    area                 (id) float32 4B ...
    country              (id) <U2 8B ...
    geo_x                (id) float32 4B ...
    geo_y                (id) float32 4B ...
    geo_z                (id) float32 4B ...
    owneroforiginaldata  (id) <U58 232B ...
    river_name           (id) <U17 68B ...
    station_name         (id) <U13 52B ...
    timezone             (id) float32 4B ...
Attributes:
    title:          Mean daily discharge (Q)
    Conventions:    CF-1.7
    references:     grdc.bafg.de
    institution:    GRDC
    history:        Download from GRDC Database, 11/05/2026
    missing_value:  -999.000
    """
runoff = data["runoff_mean"]    # metadata: print(runoff)
                                # values: print(runoff.values)                         
"""
# Display of runoff
<xarray.DataArray 'runoff_mean' (time: 33968, id: 1)> Size: 136kB
[33968 values with dtype=float32]
Coordinates:
  * time     (time) datetime64[ns] 272kB 1931-10-01 1931-10-02 ... 2024-09-29
  * id       (id) int64 8B 4127800
Attributes:
    units:      m3/s
    long_name:  Mean daily discharge (Q)
    """
values = runoff.values # .time.values pour avoir temps
#print(values)
"""
# Display of runoff.values
[[4474.062]
 [4360.794]
 [4247.527]
 ...
 [6342.963]
 [6258.013]
 [6399.597]]
 """
# Isolate runoff for certain dates with sel
runoff_06_2023_09_2024 = runoff.sel(time=slice("2023-06-01","2024-09-01")) # slicing the entire data to keep values between X and Y dates
print(runoff_06_2023_09_2024)

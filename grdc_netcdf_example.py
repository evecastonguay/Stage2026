## Example script for opening various files in different formats

 
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
import netCDF4 as nc
import numpy as np

#### SWORD example
## Section x : Extracting SWORD data
# x.1 reading the file
continent_list = ['af', 'as', 'eu', 'na', 'sa', 'oc']
dir_swr = "/obs/ecastonguay/sword_data/netcdf_v16" # sword
file_swr = 'sa' + "_sword_v16.nc" # CONTINENT!
path_swr = os.path.join(dir_swr,file_swr)
# open the netcdf file
data_swr = nc.Dataset(path_swr)
# x.2 Finding the right reach_id
r_list_swr = data_swr["reaches"]["reach_id"][:]
selected_reach_id = 62291000181 # 74210000201 article 3, fig 2a - reach on mississippi near bâton rouge (na) [found by algo]
                                # 74230900011 reach of the good graph [found by algo]
                                # 62291000181 amazonie (1420.5)
                                # 62293400041 amazonie (3520.)
                                # tagus portugal 23160300041

                                # 81130400011 article 3, fig 2b - the reach my code found (na) [NOT found] ????? -> considérer cette option dans mon code. faire une variable qui compte le nb de reaches perdus
                                # 81130400021 article 3, fig 2b - the reach they actually used (na) [found by algo]
r_index_swr = np.where(r_list_swr==selected_reach_id) # [NOT found] ????? -> considérer cette option dans mon code. faire une variable qui compte le nb de reaches perdus
if r_index_swr[0].size != 0: # if the reach_id is found in the sword database
    print('found')
else:
    print('not found')

x = data_swr["reaches"]["x"][r_index_swr]
y = data_swr["reaches"]["y"][r_index_swr]
print(f'({y},{x})')

# x.4 Retreiving the width associated with it 
w = data_swr["reaches"]["width"][r_index_swr] # selecting group with a "." doesn't work here
print(w)

# sortie de mes données: déc 2025, jan 2026 -> v17b




"""
#### MY DATASET example
<xarray.Dataset> Size: 48MB
Dimensions:          (time: 766, id: 5241)
Coordinates:
  * time             (time) datetime64[us] 6kB 2023-03-29 ... 2025-05-02
  * id               (id) int64 42kB 1159100 1159103 1159110 ... 5870600 5870655
Data variables:
    runoff_global_g  (id, time) float32 16MB 433.0 424.6 414.8 ... 59.33 49.56
    geox_global_g    (id) float32 21kB 17.72 19.15 20.36 ... 171.9 171.7 172.4
    geoy_global_g    (id) float32 21kB -28.76 -28.96 -31.81 ... -41.83 -41.76
    area_global_g    (id) float32 21kB 8.665e+05 8.599e+05 ... 6.35e+03 1.41e+03
    dschg_global_s   (id, time) float64 32MB nan nan nan nan ... nan nan nan nan
    geox_global_s    (id) float64 42kB 17.76 19.14 19.48 ... 171.7 171.6 172.9
    geoy_global_s    (id) float64 42kB -28.75 -28.96 -32.13 ... -42.29 -41.83
    id_global_s      (id) int64 42kB 12730300031 12730700131 ... 57205200091"""

##### Section GRDC example (finir, 3 juin!)
"""continent = "eu"
dir_grdc_prefix = "/obs/ecastonguay/grdc_data/"
file_nc = continent + ".nc"
path_nc = os.path.join(dir_grdc_prefix,continent,file_nc)
file_json = "stationbasins_" + continent + ".geojson"
path_json = os.path.join(dir_grdc_prefix,continent,file_json)
# open the netcdf grdc file
data_grdc = xr.open_dataset(path_nc, engine="netcdf4") # <xarray.Dataset>
#print(data_grdc)
## check later
time_sliced = data_grdc.sel(time=slice('2023-03-29','2025-05-02'))  
# print(time_sliced)
portugal = time_sliced.sel(id=6114500)
print(type(portugal.time.values[0])) # present: 6113050, 6113110, 6111100, 6114500
                # absent: """


"""
DISPLAY: time_sliced = data_grdc.sel(time=slice('2023-03-29','2025-05-02'))
<xarray.Dataset> Size: 1MB
Dimensions:              (time: 766, id: 401)
Coordinates:
  * time                 (time) datetime64[ns] 6kB 2023-03-29 ... 2025-05-02
  * id                   (id) int64 3kB 3102010 3102450 ... 3947700 3948600
Data variables:
    runoff_mean          (time, id) float32 1MB ...
    area                 (id) float32 2kB ...
    country              (id) <U2 3kB ...
    geo_x                (id) float32 2kB ...
    geo_y                (id) float32 2kB ...
    geo_z                (id) float32 2kB ...
    owneroforiginaldata  (id) <U85 136kB ...
    river_name           (id) <U30 48kB ...
    station_name         (id) <U32 51kB ...
    timezone             (id) float32 2kB ...
Attributes:
    title:          Mean daily discharge (Q)
    Conventions:    CF-1.7
    references:     grdc.bafg.de
    institution:    GRDC
    history:        Download from GRDC Database, 21/05/2026
    missing_value:  -999.000
    """

"""
DISPLAY time_sliced = data_grdc.sel(time=slice('2023-03-29','2025-05-02')):
<xarray.DataArray 'time' (time: 766)> Size: 6kB
array(['2023-03-29T00:00:00.000000000', '2023-03-30T00:00:00.000000000',
       '2023-03-31T00:00:00.000000000', ..., '2025-04-30T00:00:00.000000000',
       '2025-05-01T00:00:00.000000000', '2025-05-02T00:00:00.000000000'],
      shape=(766,), dtype='datetime64[ns]')
Coordinates:
  * time     (time) datetime64[ns] 6kB 2023-03-29 2023-03-30 ... 2025-05-02
Attributes:
    long_name:  time"""


"""
GRDC DATASET
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
#runoff = data["runoff_mean"]    # metadata: print(runoff)
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
#values = runoff.values # .time.values pour avoir temps
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
#runoff_06_2023_09_2024 = runoff.sel(time=slice("2023-06-01","2024-09-01")) # slicing the entire data to keep values between X and Y dates
#print(data["geo_y"].values[0])

##### ---------------------------------------------- Section SWOT example
single_file_name = "/obs/ecastonguay/swot_data/L4_discharge/na_sword_v16_SOS_results_unconstrained_20230502T204408_20250502T204408_20251219T163700.nc"  
data_l4 = nc.Dataset(single_file_name)

"""
Display results (data is stored within the groups):
<class 'netCDF4.Dataset'>
root group (NETCDF4 data model, file format HDF5):
    title: SWOT discharge prior information and processing outputs
    summary: All of the outputs from all processes used to generate SWOT discharge products.
    keywords: GCMD:Rivers/Streams, GCMD:Discharge/Flow, GCMD:SWOT
    keywords_vocabulary: NASA Global Change Master Directory (GCMD) Science Keywords
    doi: 10.5067/SWOT-SOS-V1
    id: SWOT_L4_DAWG_SOS_DISCHARGE
    naming_authority: gov.nasa
    standard_name_vocabulary: CF Standard Name Table v72
    featureType: timeseries
    platform: SWOT
    platform_vocabulary: GCMD platform keywords
    instrument: KaRIn
    instrument_vocabulary: GCMD instrument keywords
    processing_level: L4
    conventions: CF-1.8, ACDD-1.3
    acknowledgement: NASA AIST Program Grant Number 80NSSC22K1487, NASA SWOT Science Team Grant Numbers: 80NSSC20K1143, 80NSSC20K1141, 80NSSC20K1340 and CNES SWOT TOSCA fund for the SWOT DAHM project. Additional support from PO.DAAC and the SWOT mission.
    references: Confluence codebase: https://github.com/SWOT-Confluence; A Framework for Estimating Global River Discharge From the Surface Water and Ocean Topography Satellite Mission: https://doi.org/10.1029/2021WR031614
    creator_name: SWOT Discharge Algorithm Working Group (DAWG)
    creator_email: coss.31@osu.edu
    creator_type: group
    creator_url: https://github.com/SWOT-Confluence
    creator_institution: SWOT Science Team
    institution: NASA Jet Propulsion Laboratory (JPL) Physical Oceanography Distributed Active Archive Center (PO.DAAC)/University of Massachusetts Amherst (UMass)/Ohio State University
    project: Surface Water And Ocean Topography Discharge Algorithms Working Group (SWOT DAWG)
    program: NASA SWOT Mission
    publisher_name: JPL PO.DAAC
    publisher_email: podaac@podaac.jpl.nasa
    publisher_url: http://podaac.jpl.nasa.gov/
    publisher_type: Institution
    publisher_institution: JPL PO.DAAC
    metadata_link: https://dx.doi.org/10.5067/SWOT-SOS-V1
    geospatial_lat_units: degree
    geospatial_lon_units: degree
    license: Issued under Creative Commons CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
    continent: NA
    run_type: constrained
    product_version: 0002
    date_created: 2025-12-19T23:16:31
    uuid: 9c18518e-5021-4d91-96c8-c045ffdc37b7
    history: 2025-12-19T23:16:31: SoS version 0002 created by Confluence version ['0.1.0']
    source: Module results: priors, hivdi, metroman, moi, momma, neobam, offline, postdiagnostics, prediagnostics, sad, sic4dvar, swot, validation, lakeflow, consensus
    comment: Constrained SoS version includes results from modules: priors, hivdi, metroman, moi, momma, neobam, offline, postdiagnostics, prediagnostics, sad, sic4dvar, swot, validation, lakeflow, consensus and cycle pass observations plus time data from SWOT shapefiles
    geospatial_lat_min: 8.090422889313631
    geospatial_lat_max: 82.31075059339722
    geospatial_lon_min: -166.39687687783314
    geospatial_lon_max: 8.090422889313631
    time_coverage_start: 2023-03-29T08:52:15
    time_coverage_end: 2025-05-03T13:40:52
    time_coverage_duration: P2Y1M4DT4H48M37S
    dimensions(sizes): num_reaches(38048), num_nodes(1683122)
    variables(dimensions): 
    groups: reaches, nodes, hivdi, metroman, moi, momma, neobam, offline, postdiagnostics, prediagnostics, sad, sic4dvar, validation, lakeflow, consensus
    """
# 2.1 Reading consensus discharge data  
# Consensus group (structure and metadata)
consensus_group = data_l4.groups["consensus"]
#print(consensus_group)
"""
# two variables with 1-d each, for the n of reaches
<class 'netCDF4.Group'>
group /consensus:
    dimensions(sizes): 
    variables(dimensions): float64 consensus_q(num_reaches), int64 time_int(num_reaches)
    groups: 
"""
# Variable 'consensus_q' (structure and metadata)
consensus_q = consensus_group['consensus_q'] 

"""
CONSENSUS Q VARIABLE
<class 'netCDF4.Variable'>
vlen consensus_q(num_reaches)
    long_name: consensus_discharge
    valid_max: 10000000
    short_name: consensus
    tag_basic_expert: Basic
    coverage_content_type: modelResult
    missing_value: -999999999999.0
    comment: Discharge from the consensus discharge algorithm.
    fill: -99999999
    units: m^3/s
    valid_min: 0
vlen data type: float64
path = /consensus
unlimited dimensions: 
current shape = (38048,)
"""
# Index/slicing the variable (looking at the values)
consensus_q_values = consensus_q[:] # <class 'numpy.ndarray'>, this is a 1-d array that itself contains arrays
                                    # tous les segments du continent sont listés ici. chaque segment a un tableau contenant la liste temporelle de toutes les valeurs de débit

"""
[array([-1.e+12]) array([-1.e+12]) array([-1.e+12]) ... array([-1.e+12])
 array([-1.e+12]) array([-1.e+12])]
 """

# 2.2 Reading reach group
reach_group = data_l4.groups['reaches']
"""
REACHES GROUP
<class 'netCDF4.Group'>
group /reaches:
    dimensions(sizes): 
    variables(dimensions): int64 reach_id(num_reaches), float64 x(num_reaches), float64 y(num_reaches), <class 'str'> river_name(num_reaches), <class 'str'> observations(num_reaches), float64 time(num_reaches)
    groups: 
"""
reach_id = reach_group['reach_id'] # NOTE: quand on sélectionne des indices, le fait de mettre [:] ne change rien
                                    # reach_id = reach_group['reach_id'][0] OU reach_id = reach_group['reach_id'][:][0]
#print(data_l4.groups["consensus"]['time_int'][:])                    
reaches_indexes_continent = np.array([0,1,2])
#print(data_l4.groups["consensus"]['time_int'][reaches_indexes_continent])

"""
REACH ID VARIABLE
<class 'netCDF4.Variable'>
int64 reach_id(num_reaches)
    format: CBBBBBRRRRT
    comment: Taken from SWORD PDD: id of each reach. The format of none the id is as follows: CBBBBBRRRRT where C = Continent (the first number of the Pfafstetter basin code), B = Remaining Pfafstetter basin codes up to level 6, R = Reach id (assigned sequentially within a level 6 basin starting at the downstream end working upstream, T = Type (1 – river, 3 – lake on river, 4 – dam or waterfall, 5 – unreliable topology, 6 – ghost reach)
    valid_min: -998
    valid_max: 1000000000000000000
    coverage_content_type: referenceInformation
    long_name: reach_identifier
path = /reaches
unlimited dimensions: 
current shape = (38048,)
filling on, default _FillValue of -9223372036854775806 used
"""
reach_id_values = reach_id[:] # len: 38048
"""selected_reach_id = 81130400011
selected_reach_index_array = np.where(reach_id_values == selected_reach_id) # find the index of the reach i'm looking for (the array contains the index)
liste=np.array([25713,25714,25715])
print(reach_id_values[liste])

print("voici le reach perdu",selected_reach_index_array)"""

"""
[71120000013 71120000043 71120000053 ... 73120001026 73120001036
 73120001046]
 """


# Offline group
offline_group = data_l4.groups["offline"]

"""
OFFLINE GROUP
<class 'netCDF4.Group'>
group /offline:
    dimensions(sizes): 
    variables(dimensions): float64 d_x_area(num_reaches), float64 d_x_area_u(num_reaches), float64 metro_q_c(num_reaches), float64 bam_q_c(num_reaches), float64 hivdi_q_c(num_reaches), float64 momma_q_c(num_reaches), float64 sads_q_c(num_reaches), float64 sic4dvar_q_c(num_reaches), float64 consensus_q_c(num_reaches), float64 metro_q_uc(num_reaches), float64 bam_q_uc(num_reaches), float64 hivdi_q_uc(num_reaches), float64 momma_q_uc(num_reaches), float64 sads_q_uc(num_reaches), float64 sic4dvar_q_uc(num_reaches), float64 consensus_q_uc(num_reaches)
    groups: """

"""
GROUP LAKEFLOW
<class 'netCDF4.Group'>
group /lakeflow:
    dimensions(sizes): lakeflow_dates(689)
    variables(dimensions): int64 lake_id(num_reaches), int32 prior_fit(num_reaches), int32 type(num_reaches), float64 q_upper(num_reaches), float64 q_lower(num_reaches), float64 n_lakeflow_sd(num_reaches), float64 a0_lakeflow(num_reaches), int64 lakeflow_date(lakeflow_dates), float64 width(num_reaches, lakeflow_dates), float64 slope2(num_reaches, lakeflow_dates), float64 da(num_reaches, lakeflow_dates), float64 wse(num_reaches, lakeflow_dates), float64 storage(num_reaches, lakeflow_dates), float64 dv(num_reaches, lakeflow_dates), float64 q_model(num_reaches, lakeflow_dates), float64 tributary(num_reaches, lakeflow_dates), float64 et(num_reaches, lakeflow_dates), float64 bayes_q(num_reaches, lakeflow_dates), float64 bayes_q_sd(num_reaches, lakeflow_dates), float64 q_lakeflow(num_reaches, lakeflow_dates), float64 n_lakeflow(num_reaches, lakeflow_dates)
    groups: 
    """

"""
OFFLINE VARIABLE
<class 'netCDF4.Variable'>
vlen d_x_area(num_reaches)
    missing_value: -999999999999.0
    long_name: change in cross-sectional area
    valid_max: 10000000
    coverage_content_type: modelResult
    comment: change in cross-sectional area
    units: m^2
    valid_min: -10000000
vlen data type: float64
path = /offline
unlimited dimensions: 
current shape = (38048,)
"""

"""
TIME VARIABLE
print(data_l4.groups["consensus"]['time_int'])
<class 'netCDF4.Variable'>
vlen time_int(num_reaches)
    long_name: integer time for consensus Q - seconds since beginning of January 1, 2000
    calendar: gregorian
    short_name: time_int
    standard_name: time (seconds)
    tag_basic_expert: Basic
    missing_value: -999999999999
    comment: seconds since beginning of January 1, 2000
    fill: -999999999999
vlen data type: int64
path = /consensus
unlimited dimensions: 
current shape = (38048,)"""
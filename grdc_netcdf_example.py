## Example script for opening discharge data from the GRDC data in netCDF format
## Also contains example of how to work with the SWOT data organized in groups
 
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

##### ----- Section GRDC example
continent = "na" # SELECT a continent (africa: af, asia: as, europe: eu, north_america: na, south_america: sa, oceania: oc)
dir_l4 = "/obs/ecastonguay/swot_data/L4_discharge/"
if continent not in ['af', 'as', 'eu', 'na', 'sa', 'oc']:
    raise ValueError("Error: continent must be one of the following: 'af' (Africa), 'as' (Asia), 'eu' (Europe), 'na' (North America), 'sa' (South America), 'oc' (Oceania)")    
file_suffix = "_sword_v16_SOS_results_unconstrained_20230502T204408_20250502T204408_20251219T163700.nc"
single_file_name = dir_l4 + continent + file_suffix
data_l4 = nc.Dataset(single_file_name)
# Open netCDF file
dir_grdc = "/obs/ecastonguay/grdc_data/na/GRDC-Daily.nc"
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
#print(data["geo_y"].values[0])

##### ----- Section SWOt example
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
print(type(consensus_q_values))
"""
[array([-1.e+12]) array([-1.e+12]) array([-1.e+12]) ... array([-1.e+12])
 array([-1.e+12]) array([-1.e+12])]
 """

# 2.2 Reading reach group
reach_group = data_l4.groups['reaches']
"""
<class 'netCDF4.Group'>
group /reaches:
    dimensions(sizes): 
    variables(dimensions): int64 reach_id(num_reaches), float64 x(num_reaches), float64 y(num_reaches), <class 'str'> river_name(num_reaches), <class 'str'> observations(num_reaches), float64 time(num_reaches)
    groups: 
"""
reach_id = reach_group['reach_id'] 
"""<class 'netCDF4.Variable'>
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
"""
[71120000013 71120000043 71120000053 ... 73120001026 73120001036
 73120001046]
 """
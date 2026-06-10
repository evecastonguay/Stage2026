## Ève Castonguay, LIRA
## Created on 10/08/2026

def corresp(station_id, r_index, dataframe_watershed):
    """Establishes the best match possible between a given GRDC station 
    and a SWOT reach, considering the distance between the two, the 
    watershed area and the name of the river."""

    # grdc (watershed) relevant data for the comparison
    sel_station = dataframe_watershed.loc[dataframe_watershed["grdc_no"] == station_id] 
    area_station = sel_station.iloc[0]['area_calc'] 
    river_station = sel_station.iloc[0]['river']
    id_station = sel_station.iloc[0]['grdc_no']



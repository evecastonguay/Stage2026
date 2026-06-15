## Ève Castonguay, LIRA
## Created on 10/08/2026

# Imports
import numpy as np
from rapidfuzz import fuzz, process
import re
from scipy.spatial import KDTree
import pandas as pd

def dschg_darray(dschg_swt, time_swt, r_id_swt, time_dim, mv_swot):
    """Takes the swot discharge data (arrays in an array which have different shapes) and 
    formats it into a (len(r_id), time) array, where time goes from swot_start to swot_end."""

    ## Section 1 : Declaring the empty array
    dschg_swt_rfm = np.full((len(r_id_swt), len(time_dim)), np.nan, dtype=np.float64) # dim (id, time)
    
    ## Section 2 : Iterating through discharge data
    for i, dschg in enumerate(dschg_swt):
        mask_mv = (dschg != mv_swot)
        dschg_flt = dschg[mask_mv]
        if dschg_flt.size != 0:
            if dschg.shape != time_swt[i].shape:
                continue # sometimes, there is no time data for the corresponding discharge time series
            else:
                time_flt = time_swt[i][mask_mv]
                # converting time from int to datetime
                epoch = np.datetime64('2000-01-01')
                datetime_flt = epoch + time_flt.astype('timedelta64[s]')
                # put hours,min,sec to 00:00:00 to ignore time and only keep date
                datetime_norm = pd.to_datetime(datetime_flt).normalize()
                # get index 
                time_index = time_dim.get_indexer(datetime_norm) # indice of everywhere where the day corresponds
                # put in empty array
                dschg_swt_rfm[i,time_index] = dschg_flt
        else:
            continue
    return dschg_swt_rfm


def corresp_area_name(station_id, r_id, r_index, area_darray_g, river_darray_g, area_darray_swr, river_darray_sword, dschg_swot):
    """Establishes the best match possible between a given GRDC station 
    and a SWOT reach, considering the watershed area and the name of the river.
    Returns the id of the reach if the match is valid, or np.nan otherwise."""

    ## Section 1 : Thresholds
    thr = [0.5, 20, 75] # [alpha, rel_err_threshold (%), str_similarity]

    ## Section 2 : Does the selected nearest reach contains discharge data?
    dschg_check_s = dschg_swot[r_index][:]
    mask_mv = (dschg_check_s != dschg_swot.missing_value) # [tested]
    dschg_flt_s = dschg_check_s[mask_mv] # [tested] filtered discharge
    if dschg_flt_s.size == 0: # for a given index, is all discharge data missing value?
        return None, 'no dschg data'

    ## Section 3 : Does the selected nearest reach contains similar area values?
    area_check_s = area_darray_swr.sel(reach_id=r_id).values 
    area_check_g = area_darray_g.sel(id=station_id).values # ex: 866486.0
    # infinity
    alpha = thr[0]
    if area_check_g < alpha:  
        return None, 'null g area' # reason of the reject
    # compute relative error on areas
    rel_err_thr = thr[1] # %
    area_rel_err = (np.absolute(area_check_s - area_check_g) / area_check_g)*100

    # Section 4 : Is the river name similar? 
    river_check_s = river_darray_sword.sel(reach_id=r_id).item() # to scalar
    river_check_g = river_darray_g.sel(id=station_id).item()
    # lower case
    river_lc_g = river_check_g.lower()
    # deal with possible multiple names in swot
    river_list_s = river_check_s.split("; ") # "apple#banana#cherry#orange" -> ['apple', 'banana', 'cherry', 'orange']
    # deal with possible problematic parenthesis (example: Nelson River (west Channel))
    river_list_lc_s = np.array([ re.sub(r'\s*\(.*?\)', '', river_list_s[k]).lower() for k in range(len(river_list_s)) ])
    # compute fuzz ratio
    ratio_list_s = np.array([fuzz.ratio(river_lc_g, river_list_lc_s[i]) for i in range(len(river_list_lc_s))])
    best_river_s = river_list_lc_s[np.argmax(ratio_list_s)]
    best_ratio = ratio_list_s[np.argmax(ratio_list_s)]
    ratio_thr = thr[2]

    ## Section 5 : Compare of area & name to thresholds
    if (best_ratio >= ratio_thr):
        if (area_rel_err <= rel_err_thr):
            return (r_id, r_index, dschg_flt_s, mask_mv, best_river_s), 'ok'
        else: 
            return None, 'area'
    else:
        return None, 'name'

def corresp_name_dist_area_v4(station_id, cleaned_river_names_swr, cleaned_river_idx_swr, r_id_swr, r_id_swt, area_darray_g, river_darray_g, area_darray_swr, dschg_swot, x_station, y_station, geox_darray_swt, geoy_darray_swt):
    """Establishes the best match possible between a given GRDC station 
    and a SWOT reach, considering first the river name, the distance and finally the watershed area.
    Returns None if no match was possible, otherwise the following: sel_r_id, sel_r_ix, sel_dschg, sel_mask, sel_river_name"""

    ## Section 1 : Set thresholds
    thr = [0.5, 20, 75, 50] # [alpha, rel_err_threshold (%), str_similarity, distance (km)]

    ## Section 2 : Compare names
    query_name_g = river_darray_g.sel(id=station_id).item().lower()
    matches = process.extract(
        query_name_g, # name of the river we are searching
        cleaned_river_names_swr, # list of sword river names 
        scorer=fuzz.ratio,
        score_cutoff=thr[2]
    )
    if not matches:
        return None, 'no match on name' 
    
    # matches: if no match, returns [].
    # else, returns [('Mississippi', 100.0, 2)]
    best_per_index = {}
    for match, score, idx in matches:
        # get original indices
        org_indice = cleaned_river_idx_swr[idx] # get the original indice of the reach name
                                            # sword_names = ["Mississippi river; Missi river", "Amazon", ...] -> sword_cleaned = ["Mississippi river", "Missi river", "Amazon", ...] -> index_cleaned = [0, 0, 1]
                                            # original_indices will get the right value in index_cleaned
                                            # The list is sorted by similarity or distance depending on the scorer used. The first element in the list has the highest similarity/smallest distance.

        # prevent duplicates in list
        if (org_indice not in best_per_index):
            best_per_index[org_indice] = match
    
    # indexes sword
    idx_list_swr = list(best_per_index.keys()) 
    # river names sword
    river_names_list = list(best_per_index.values())
    # r id sword
    r_ids_list = [r_id_swr[l] for l in idx_list_swr]
    # indexes in swot (using r_ids)
    id_to_idx = {rid: i for i, rid in enumerate(r_id_swt)} # dict of r_id: index in swot
    idx_list_swt = np.array([id_to_idx[rid] for rid in r_ids_list]) # array of indexes of the indexes
    # discharge swot
    dschg_list = [dschg_swot[x] for x in idx_list_swt]
    # mask discharge
    mask_list = []
   
    assert len(r_ids_list) == len(idx_list_swt) == len(river_names_list) == len(idx_list_swr) == len(dschg_list)

    ## Section 3 : Empty dschg data? 
    # new lists to iterate
    new_r_ids = []
    new_index_swt = []
    new_r_names = []
    new_dschg = []
    new_mask = []

    for ix, dschg in enumerate(dschg_list):
        mask_mv = (dschg != dschg_swot.missing_value)
        dschg_flt_s = dschg[mask_mv]

        if dschg_flt_s.size != 0:
            new_r_ids.append(r_ids_list[ix])
            new_index_swt.append(idx_list_swt[ix])
            new_r_names.append(river_names_list[ix])
            new_dschg.append(dschg_flt_s)
            new_mask.append(mask_mv)

    # going back to the old list names
    r_ids_list = new_r_ids
    idx_list_swt = new_index_swt
    river_names_list = new_r_names
    dschg_list = new_dschg
    mask_list = new_mask
            
    if not r_ids_list:
        return None, 'empty dschg data'
    
    assert len(r_ids_list) == len(idx_list_swt) == len(river_names_list) == len(dschg_list) == len(mask_list)

    ## Section 4 : Close distance?
    geox_list_s = []
    geoy_list_s = []
    for id in r_ids_list:
        geox_list_s.append(geox_darray_swt.sel(reach_id=id))
        geoy_list_s.append(geoy_darray_swt.sel(reach_id=id))
    stacked_xy = np.vstack((geox_list_s,geoy_list_s)).T
    k_neighbors = [stacked_xy.shape[0]] # the goal here is to compute distance, then check if below threshold
    distance_list, index_list = KDTree(stacked_xy).query([x_station, y_station],k=k_neighbors) 
    for ii in range(len(k_neighbors)): 
        if distance_list[ii] <= thr[3]: # distance_list[ii] for multiple neighbors
            
            pos = index_list[ii]
            sel_r_id = r_ids_list[pos] # index_list[ii] is in same order as r_ids_list
            sel_ix_swt = idx_list_swt[pos]
            sel_river_name = river_names_list[pos] 
            sel_dschg = dschg_list[pos] 
            sel_mask = mask_list[pos]
            
            ## Section 5 : Similar areas?
            area_check_s = area_darray_swr.sel(reach_id=sel_r_id)
            area_check_g = area_darray_g.sel(id=station_id).values 

            if area_check_g < thr[0]:  
                continue 
            
            # compute relative error on areas
            rel_err_thr = thr[1] # %
            area_rel_err = (np.absolute(area_check_s - area_check_g) / area_check_g)*100
            if (area_rel_err <= rel_err_thr):
                return (sel_r_id, sel_ix_swt, sel_dschg, sel_mask, sel_river_name), 'ok'
    return None, 'no match on distance or area'


def corresp_name_dist_area_v5(station_id, cleaned_river_names_swr, cleaned_river_idx_swr, r_id_swr, area_darray_g, river_darray_g, area_darray_swr, dschg_darray_swt, x_station, y_station, geox_darray_swt, geoy_darray_swt):
    """Establishes the best match possible between a given GRDC station 
    and a SWOT reach, considering first the river name, the distance and finally the watershed area.
    Returns None if no match was possible, otherwise the following: sel_r_id, sel_river_name.
    This function also considers the swot discharge to be a DataArray, not a simple list (novelty compared to v4)."""

    ## Section 1 : Set thresholds
    thr = [0.5, 20, 75, 50] # [alpha, rel_err_threshold (%), str_similarity (Indel index), distance (km)]

    ## Section 2 : Compare names
    query_name_g = river_darray_g.sel(id=station_id).item().lower()
    matches = process.extract(
        query_name_g, # name of the river we are searching
        cleaned_river_names_swr, # list of sword river names 
        scorer=fuzz.ratio,
        score_cutoff=thr[2]
    )
    if not matches:
        return None, 'no match on name' 
    
    # matches: if no match, returns [].
    # else, returns [('Mississippi', 100.0, 2)]
    best_per_index = {}
    for match, score, idx in matches:
        # get original indices
        org_indice = cleaned_river_idx_swr[idx] # get the original indice of the reach name
                                            # sword_names = ["Mississippi river; Missi river", "Amazon", ...] -> sword_cleaned = ["Mississippi river", "Missi river", "Amazon", ...] -> index_cleaned = [0, 0, 1]
                                            # original_indices will get the right value in index_cleaned
                                            # The list is sorted by similarity or distance depending on the scorer used. The first element in the list has the highest similarity/smallest distance.

        # prevent duplicates in list
        if (org_indice not in best_per_index):
            best_per_index[org_indice] = match
    
    # indexes sword
    idx_list_swr = list(best_per_index.keys()) 
    # river names sword
    river_names_list = list(best_per_index.values())
    # r id
    r_ids_list = [r_id_swr[l] for l in idx_list_swr]
    # discharge swot
    dschg_list = [dschg_darray_swt.sel(reach_id=r).values for r in r_ids_list]
   
    assert len(r_ids_list) == len(river_names_list) == len(idx_list_swr) == len(dschg_list)

    ## Section 3 : Empty dschg data? 
    # new lists to iterate
    new_r_ids = []
    new_r_names = []

    for ix, dschg in enumerate(dschg_list):
        mask_mv = (dschg != dschg_darray_swt.missing_value)
        dschg_flt_s = dschg[mask_mv]

        if dschg_flt_s.size != 0:
            new_r_ids.append(r_ids_list[ix])
            new_r_names.append(river_names_list[ix])

    # going back to the old list names
    r_ids_list = new_r_ids
    river_names_list = new_r_names
            
    if not r_ids_list:
        return None, 'empty dschg data'
    
    assert len(r_ids_list) == len(river_names_list)

    ## Section 4 : Close distance?
    geox_list_s = []
    geoy_list_s = []
    for id in r_ids_list:
        geox_list_s.append(geox_darray_swt.sel(reach_id=id))
        geoy_list_s.append(geoy_darray_swt.sel(reach_id=id))
    stacked_xy = np.vstack((geox_list_s,geoy_list_s)).T
    k_neighbors = [stacked_xy.shape[0]] # the goal here is to compute distance, then check if below threshold
    distance_list, index_list = KDTree(stacked_xy).query([x_station, y_station],k=k_neighbors) 
    for ii in range(len(k_neighbors)): 
        if distance_list[ii] <= thr[3]: # distance_list[ii] for multiple neighbors
            
            pos = index_list[ii]
            sel_r_id = r_ids_list[pos] # index_list[ii] is in same order as r_ids_list
            sel_river_name = river_names_list[pos]  
            
            ## Section 5 : Similar areas?
            area_check_s = area_darray_swr.sel(reach_id=sel_r_id)
            area_check_g = area_darray_g.sel(id=station_id).values 

            if area_check_g < thr[0]:  
                continue 
            
            # compute relative error on areas
            area_rel_err = (np.absolute(area_check_s - area_check_g) / area_check_g)*100
            if (area_rel_err <= thr[1]):
                return (sel_r_id, sel_river_name), 'ok' 
    return None, 'no match on distance or area'


    
     
    
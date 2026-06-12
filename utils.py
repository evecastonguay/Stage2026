## Ève Castonguay, LIRA
## Created on 10/08/2026

# Imports
import numpy as np
from rapidfuzz import fuzz
import re
from scipy.spatial import KDTree

def corresp(station_id, r_id, r_index, area_darray_g, river_darray_g, area_darray_sword, river_darray_sword, dschg_swot):
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
    area_check_s = area_darray_sword.sel(reach_id=r_id).values 
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
    
def corresp_name_dist_area(station_id, river_cleaned_s, r_id_sword, area_darray_g, river_darray_g, area_darray_sword, dschg_swot, x_station, y_station, geox_darray_s, geoy_darray_s):
    
    ## Section 1 : Set thresholds
    thr = [0.5, 20, 80, 50] # [alpha, rel_err_threshold (%), str_similarity, distance (km)]

    ## Section 2 : Compare names
    r_ids_list = []
    r_index_list = []
    r_names_list = []
    dschg_list = []
    mask_list = []
    # get station name
    river_name_g = river_darray_g.sel(id=station_id).item().lower()
    # list of all similar names in sword
    for i in range(len(river_cleaned_s)): # possible multiple names per swot reach
        river_names_s = river_cleaned_s[i]
        ratio_list = np.array([fuzz.ratio(river_name_g, r) for r in river_names_s])
        best_river = river_names_s[np.argmax(ratio_list)]
        best_ratio = ratio_list[np.argmax(ratio_list)]
        if (best_ratio >= thr[2]):
            # find related r id
            r_id_s = r_id_sword[i] 

            r_ids_list.append(r_id_s)
            r_index_list.append(i)
            r_names_list.append(best_river)
    if not r_ids_list:
        return None, 'no match name'
    
    assert len(r_ids_list) == len(r_index_list) == len(r_names_list)

    ## Section 3 : Empty dschg data? 
    new_r_ids = []
    new_r_index = []
    new_r_names = []
    new_dschg = []
    new_mask = []

    for pos, ix in enumerate(r_index_list):
        dschg_check_s = dschg_swot[ix][:]
        mask_mv = (dschg_check_s != dschg_swot.missing_value)
        dschg_flt_s = dschg_check_s[mask_mv]

        if dschg_flt_s.size != 0:
            new_r_ids.append(r_ids_list[pos])
            new_r_index.append(r_index_list[pos])
            new_r_names.append(r_names_list[pos])
            new_dschg.append(dschg_flt_s)
            new_mask.append(mask_mv)

    r_ids_list = new_r_ids
    r_index_list = new_r_index
    r_names_list = new_r_names
    dschg_list = new_dschg
    mask_list = new_mask
            
    if not r_ids_list:
        return None, 'empty dschg data'
    
    assert len(r_ids_list) == len(r_index_list) == len(r_names_list) == len(dschg_list) == len(mask_list)

    ## Section 4 : Close distance?
    geox_list_s = []
    geoy_list_s = []
    for id in r_ids_list:
        geox_list_s.append(geox_darray_s.sel(reach_id=id))
        geoy_list_s.append(geoy_darray_s.sel(reach_id=id))
    stacked_xy = np.vstack((geox_list_s,geoy_list_s)).T
    k_neighbors = [stacked_xy.shape[0]] # the goal here is to compute distance, then check if below threshold
    distance_list, index_list = KDTree(stacked_xy).query([x_station, y_station],k=k_neighbors) 
    for ii in range(len(k_neighbors)): 
        if distance_list[ii] <= thr[3]: # distance_list[ii] for multiple neighbors
            
            sel_r_id = r_ids_list[index_list[ii]] # index_list[ii]
            sel_r_ix = r_index_list[index_list[ii]] # index_list[ii]
            sel_river_name = r_names_list[index_list[ii]] # index_list[ii]
            sel_dschg = dschg_list[index_list[ii]] # idem ...
            sel_mask =mask_list[index_list[ii]]
            
            ## Section 5 : Similar areas?
            area_check_s = area_darray_sword.sel(reach_id=sel_r_id)
            area_check_g = area_darray_g.sel(id=station_id).values 
            if area_check_g < thr[0]:  
                return None, 'area g null' # reason of the reject
            # compute relative error on areas
            rel_err_thr = thr[1] # %
            area_rel_err = (np.absolute(area_check_s - area_check_g) / area_check_g)*100
            if (area_rel_err <= rel_err_thr):
                return (sel_r_id, sel_r_ix, sel_dschg, sel_mask, sel_river_name), 'ok'
            else:
                return None, 'no match areas'
    return None, 'no match distance'



    
    
     
    
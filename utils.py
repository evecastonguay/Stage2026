## Ève Castonguay, LIRA
## Created on 10/08/2026

# Imports
import numpy as np
from rapidfuzz import fuzz

def corresp(station_id, r_id, r_index, area_darray_g, river_darray_g, area_darray_sword, river_darray_sword, dschg_swot):
    """Establishes the best match possible between a given GRDC station 
    and a SWOT reach, considering the watershed area and the name of the river.
    Returns the id of the reach if the match is valid, or np.nan otherwise."""

    # Thresholds
    thr = [0.5, 20, 75] # [alpha, rel_err_threshold (%), str_similarity]

    # Test #1 : Does the selected nearest reach contains discharge data?
    dschg_check_s = dschg_swot[r_index][:]
    mask_mv = (dschg_check_s != dschg_swot.missing_value) # [tested]
    dschg_flt_s = dschg_check_s[mask_mv] # [tested] filtered discharge
    if dschg_flt_s.size == 0: # for a given index, is all discharge data missing value?
        return None

    # Test #2 : Does the selected nearest reach contains similar area values?
    area_check_s = area_darray_sword.sel(reach_id=r_id).values 
    area_check_g = area_darray_g.sel(id=station_id).values # ex: 866486.0
    # infinity
    alpha = thr[0]
    if area_check_g < alpha:  
        return None
    # compute relative error on areas
    rel_err_thr = thr[1] # %
    area_rel_err = (np.absolute(area_check_s - area_check_g) / area_check_g)*100

    # Test #3 : Is the river name similar? 
    river_check_s = river_darray_sword.sel(reach_id=r_id).values
    river_check_g = river_darray_g.sel(id=station_id).values 
    # lower case
    river_lc_s = river_check_s.lower()
    river_lc_g = river_check_g.lower()
    # deal with possible multiple names in swot
    river_list_s = river_lc_s.split(";") # "apple#banana#cherry#orange" -> ['apple', 'banana', 'cherry', 'orange']
    ratio_list_s = np.array([fuzz.ratio(river_lc_g, river_list_s[i]) for i in range(len(river_list_s))])
    best_river_s = river_list_s[np.argmax(ratio_list_s)]
    # compute fuzz ratio
    ratio = fuzz.ratio(river_lc_g, best_river_s)
    ratio_thr = thr[2]

    # Comparison of area & name to thresholds
    if (ratio >= ratio_thr) and (area_rel_err <= rel_err_thr):
        return r_id, r_index, dschg_flt_s, mask_mv
    else:
        return None
    
    
"""
A bunch of utility functions, and rather too specific to proline. Functions are used by the targeted workflow and std_finder workflow.
The key object being manipulated here is a dictionary of dfs, where each df has two columns: ppm (float) and intensity (float). This is because this is meant to be batch processed. 
"""
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
import itertools
from scipy import integrate
from scipy.signal import fftconvolve
from scipy.stats import pearsonr
from bs4 import BeautifulSoup

import sys
import os
import time
import json
import re
from io import StringIO


def find_exp_num(base_dir):
    """Find the experiment number within a given NMR run, where the acqu file contains `noesy` and `1d`.
    
    PARAMS
    ------
    base_dir: str; path containing subdirs of experiment numbers like /10, /11, /13, /9999
    
    RETURNS
    -------
    matching_dirs: list of str, usually of len=1; list containing experiment numbers matching requirements. 
    """
    matching_dirs = []

    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)

        if os.path.isdir(subdir_path):
            acqu_path = os.path.join(subdir_path, 'acqu')

            if os.path.exists(acqu_path):
                with open(acqu_path, 'r') as f:
                    for line in f:
                        if "PULPROG" in line and ("noesy" in line and "1d" in line):
                            matching_dirs.append(subdir)
                            break

    return matching_dirs


def get_unique_elements(my_list):
    """Gets the unique elements from my_list, preserving order. 
    Neither set() nor np.unique() do this natively. 
    Assumes that duplicate values in my_list already cluster together.
    """
    uq_list = []
    for val in my_list:
        if val not in uq_list:
            uq_list.append(val)
    
    return uq_list


def get_df_auc(query_df, df_dict, results_dict, multiplets_ls, set_intensity_min_to_zero_bool=False):
    """Get scaling factor by naive height matching and store in a df. Run after do_1d_std_search(). 
    keys in results_dict and df_dict must match. 
    
    PARAMS
    ------
    query_df: df; full spectra template.
    df_dict: dictionary of full spectra of samples. 
    results_dict: results from do_1d_std_search. 
    multiplets_ls: list of all multiplets. 
    set_intensity_min_to_zero_bool: bool; if true, set minimum intensity to zero. 
    
    RETURNS
    -------
    df_conc: df of auc and scaling factors, for all multiplets. 
    """
    c = []
    for k in list(results_dict.keys()):
        multiplet_idx = 0
        for multiplet_i in list(results_dict[k].keys()):
            # get coordinates of multiplet in query and target
            mcoords_query = multiplets_ls[multiplet_idx]
            mcoords_target = results_dict[k][multiplet_i]["multiplet_match_ppm"][0]

            # get multiplet regions of query and target
            dt_query = query_df.loc[(query_df["ppm"]>min(mcoords_query)) & (query_df["ppm"]<max(mcoords_query))].copy()
            dt_target = df_dict[k].copy()
            dt_target = dt_target.loc[(dt_target["ppm"]>min(mcoords_target)) & (dt_target["ppm"]<max(mcoords_target))].copy()

            # Optional: floor intensities to 0
            if set_intensity_min_to_zero_bool:
                temp_ls = dt_query["intensity"].values - min(dt_query["intensity"].values)
                dt_query["intensity"] = temp_ls
                temp_ls = dt_target["intensity"].values - min(dt_target["intensity"].values)
                dt_target["intensity"] = temp_ls

            # calc sf
            #sf = get_scaling_factor(dt_query, dt_target, [min(mcoords_query), max(mcoords_query)]) # by height
            sf = max(dt_query["intensity"].values)/max(dt_target["intensity"].values)
            auc_template_fitted = integrate.simps(dt_query.intensity.values/sf)
            auc_target = integrate.simps(dt_target.intensity.values)

            # retrieve normxcorr score for reporting purposes
            rho = max(results_dict[k][multiplet_i]["rho_ls"])

            c.append([k, multiplet_i, sf, auc_template_fitted, auc_target, rho])

            multiplet_idx += 1

    df_auc = pd.DataFrame(data=c, columns=["sample_name", "multiplet", "scaling_factor", "auc_template_fitted", "auc_target", "normxcorr"])

    return df_auc


def adjust_to_ref_peak(df, ref_coords, tolerance_window=[0, 0], adjust_ref_peak_height_bool=False, ref_peak_target_height=1E6):
    """Shifts the spectrum df horizontally s.t. the the reference peak is centred at zero.

    TODO: planned improvements:
    1. change ref_window_size to ref_end_coord instead
    2. use norm_xcorr to find a nice gaussian shape, and return a warning if no Gaussian is found with
    norm_xcorr score > 0.95, or something like that.

    Update 26-jan-2022:
    Seeing ref_pk height to a constant disabled because Brukker changed something, so the ref_pk conc 
    is no longer consistent and can no longer serve as a baseline to adjust to. Code left in for posterity, 
    but commented out. 

    PARAMS
    ------
    df: pandas df of the entire spectrum of a sample, with 2 cols: ppm (float) and intensity (float).
    ref_coords: list of 2 floats; starting and ending coords of ref peak window, in ppm. Note: since max() 
        and min() will be used, order of ref_coords doesn't matter. 
    tolerance_window: list of length 2: coordinates that contain zero. If ppm_ref_ht_max falls within this window, 
    the ref_pk will NOT be adjusted. Tolerance_window should fall within ref_coords. Set to [0, 0] to always adjust.
    adjust_ref_peak_height_bool: bool; whether or not to adjust ref peak ht. Set to False by default. !!DO NOT SET TO TRUE UNLESS EXPLICITLY INSTRUCTED TO DO SO.!!
    ref_peak_target_height: int; ref peak target height. Arbitrarily set to 1E6. Unused if adjust_ref_peak_height_bool is False. 

    RETURNS
    -------
    df: pandas df of the entire-spectrum with ref peak centred at zero.
    """
    # ===== Centre rf_pk at 0 =====
    df2 = df.copy()
    # set ppm corresponding to ref peak height to exactly 0
    dt_ref = df2.loc[(df2["ppm"] > min(ref_coords)) & (df2["ppm"] < max(ref_coords))].copy().reset_index(drop=True)
    # grab the ppm of the highest intensity in the ref window
    ppm_ref_ht_max = dt_ref.loc[dt_ref["intensity"].idxmax()].ppm
    if (ppm_ref_ht_max < min(tolerance_window)) or (ppm_ref_ht_max > max(tolerance_window)):
        # x-shift the entire spectrum in-place
        ppm_arr = df2["ppm"].values
        ppm_arr -= ppm_ref_ht_max
        df2["ppm"] = ppm_arr

    # ===== set rf_pk_height to 1E6 =====
    if adjust_ref_peak_height_bool:
        scaling_factor = ref_peak_target_height/max(dt_ref["intensity"])
        temp_array = df["intensity"].values*scaling_factor
        df2["intensity"] = temp_array

    return df2


def norm_xcorr(arr1, arr2):
    """Calc normalized cross-correlation between arrays arr1 and arr2.
    arr1 and arr2 must be of the same length.
    Used in 1d_std_search().
    """
    numerator = np.sum((arr1 - np.mean(arr1))*(arr2 - np.mean(arr2)))/len(arr1)
    denominator = np.sqrt(np.var(arr1) * np.var(arr2))

    # zero-div-error. Variance = 0 for constant arrays, which can happen in suppressed regions
    norm_xcorr = 0
    if denominator != 0:
        norm_xcorr = numerator/denominator

    return norm_xcorr


def normxcorr_1d_search(template, target):
    """Slides a shorter template (np.array), over a longer target (np.array), to find the location and value of max norm_xcorr of template in target. Error out if len(template) > len(target). 

    PARAMS
    =======
    template: 1D np array of float. 
    target: 1D np array of float. 

    RETURNS
    =======
    rho_ls: list of normxcorr scores. In practice, you'll take the max
    """
    
    rho_ls = []
    if len(template) <= len(target):
        window_size = len(template)
        for i in range(0, len(target) - window_size):
            target_window = target[i:i+window_size]
            rho_ls.append(norm_xcorr(target_window, template))
        rho_ls = np.array(rho_ls)

    else:
        print("ERROR: len(template) > len(target)!")

    return rho_ls


def normxcorr2(template, image, mode="full"):
    """
    Used in do_2d_std_search(). 
    src: https://github.com/Sabrewarrior/normxcorr2-python/blob/master/normxcorr2.py
    Input arrays should be floating point numbers. Based on matlab/octave normxcorr2 implementation.
    Does normxcorr calculations to search for `template` in `image`, where `template` and `image` can be N-D arrays, N>1.
    Not that interested in N > 2, though.
    For some reason, this only works if nrows = 257 (i.e. J-res dimension form the rows)
    WARN: there's a zero division error somewhere.
    Used in 2d_std_search()

    PARAMS
    ------
    template: N-D array of float, of template or filter you are using for cross-correlation.
    Must be of less-then-or-equal dimensions to image.
    Length of each dimension must be less than length of image.
    image: N-D array of float
    mode: Options, "full", "valid", "same"
        full (Default): The output of fftconvolve is the full discrete linear convolution of the inputs.
        Output size will be image size + 1/2 template size in each dimension.
        valid: The output consists only of those elements that do not rely on the zero-padding.
        same: The output is the same size as image, centered with respect to the ‘full’ output.

    RETURNS
    -------
    N-D array of same dimensions as image. Size depends on mode parameter.
    """
    # If this happens, it is probably a mistake
    if np.ndim(template) > np.ndim(image) or \
            len([i for i in range(np.ndim(template)) if template.shape[i] > image.shape[i]]) > 0:
        print("WARNING (in normxcorr2): TEMPLATE larger than IMG.")

    template = template - np.mean(template)
    image = image - np.mean(image)

    a1 = np.ones(template.shape)
    # Faster to flip up down and left right then use fftconvolve instead of scipy's correlate
    ar = np.flipud(np.fliplr(template))
    out = fftconvolve(image, ar.conj(), mode=mode)

    image = fftconvolve(np.square(image), a1, mode=mode) - \
            np.square(fftconvolve(image, a1, mode=mode)) / (np.prod(template.shape))

    # Remove small machine precision errors after subtraction
    image[np.where(image < 0)] = 0

    template = np.sum(np.square(template))
    out = out / np.sqrt(image * template)

    # Remove any divisions by 0 or very close to 0
    out[np.where(np.logical_not(np.isfinite(out)))] = 0

    return out


def multiplet_match_trimming(idx_ls, multiplet_match_sep):
    """Remove consecutive numbers that are not at least multiplet_match_sep apart.

    e.g.
    > my_array = [1, 2, 3, 8, 9, 10, 20, 21, 22]
    > multiplet_match_trimming(my_array, 5)
    [1, 8, 20]
    > multiplet_match_trimming(my_array, 10)
    [1, 20]
    > multiplet_match_trimming(my_array, 11)
    [1]

    PARAMS
    ------
    idx_ls: sorted list of numbers, sorted with smallest in idx_ls[0].
        Numbers can be floats, but indices are usually integers anyway.
    multiplet_match_sep: int; minimmum separation between

    RETURNS
    -------
    idx_ls2: list with consecutive numbers removed. Always includes idx_ls[0].
    """
    idx_ls2 = [idx_ls[0]]
    for i in range(1, len(idx_ls)):
        if idx_ls[i] - idx_ls[i-1] >= multiplet_match_sep:
            idx_ls2.append(idx_ls[i])

    return idx_ls2


def do_1d_std_search(query_df, target_df, multiplets_ls, query_l_dict={}, search_region_padding_size=0.1, floor_window=False):
    """
    For a single std, for each multiplet coord in multiplets_ls, search for `query` in `target`. Each multiplet will be searched for in the approximate 
    region where it's located in query_df, +/- search_region_padding_size. query_df and target_df do not need to be of the same length, though they commonly 
    are anyway, or at least off-by-one. 

    PARAMS
    ------
    query_df: df; full spectrum df of ppm and intensity of the STD. Full spectra that gets partitioned into multiplets, according to multiplets_ls.
    target_df: df; full spectrum df of ppm and intensity of the sample ("target") to be searched. Full spectra that gets partitioned into multiplets, 
    according to multiplets_ls.
    multiplets_ls: list of sublists of float, where each sublist is of length 2, i.e. a set of coordinates. List of ppm coords of the multiplets in an std. 
    Note that order of each sublist doesn't matter, because min() and max() are used anyway to get coordinate end-points.
    query_l_dict: dict of 3 dfs: 'l_params' and 'l_spectra', 'ppm_spectra'. Only 'l_params' is used in this function to get the number of lorentzians
        for each multiplet, as a proxy of how complex that multiplet is. Note that multiplet complexity is not a variable use in computation (yet). 
        To leave this unused, default to an empty dictionary `{}`.
    search_region_padding_size: float; amount to extend either side of the target_window by, in ppm.

    RETURNS
    -------
    results_dict: dict of dicts. Top-level dict has "multiplet_0", "multiplet_1"...as keys, each subdict has the keys:
        'coords': coordinates of this multiplet. 
        'num_lorentzians': number of lorentzians in this multiplet. 
        'rho_ls': full array of rho values, calculated by norm_xcorr().
        'multiplet_match_idx': list of indices of rho_ls where max(rho_ls) is found, which reports the right end-point coord on the target where the query was found. 
            Length > 1 indicates multiple matches.
        'multiplet_match_ppm': list of lists: list of ppm coords corresponding to multiplet_match_idx.
        'max_rho': list of maximum rho values, following the respective values of multiplet_match_idx
        'ppm_shift': float; amount to shift ppm by, added to std. Can be negative.
    """
    # check ppm order of query_df and target_df, that it's in NMR order
    # (i.e. the wrong way round)
    if query_df.ppm.values[0] < query_df.ppm.values[-1]:
        print("ERROR: query_df ppm values need to be in NMR order. Check input.")
    if target_df.ppm.values[0] < target_df.ppm.values[-1]:
        print("ERROR: target_df ppm values need to be in NMR order. Check input.")

    # wherein results will be stored
    results_dict = {}
    for i in range(len(multiplets_ls)):
        k = "multiplet_"+str(i)

        row_dict = {}
        row_dict["coords"] = multiplets_ls[i]
        if query_l_dict != {}:
            dt = query_l_dict["l_params"]
            row_dict["num_lorentzians"] = len(dt.loc[(dt["filtered_pk_ppm"]>multiplets_ls[i][0]) & (dt["filtered_pk_ppm"]<multiplets_ls[i][1])])

        results_dict[k] = row_dict

    # ===== run =====
    # declare params later used in multiplet_match_trimming(): q_th and multiplet_match_sep
    q_th = 99.9 # threshold of correlation to remove duplicates with
    multiplet_match_sep = 5 # discard consecutive match_indices that aren't at least this number apart

    for k in results_dict.keys():
        query_window = query_df.loc[(query_df["ppm"]>min(results_dict[k]["coords"])) &
                                       (query_df["ppm"]<max(results_dict[k]["coords"]))
                                      ].copy()
        results_dict[k]["multiplet_len_idx"] = len(query_window)
        mcoord_width = max(results_dict[k]["coords"]) - min(results_dict[k]["coords"])
        results_dict[k]["multiplet_len_ppm"] = mcoord_width
        window_size = len(query_window)
        search_target = target_df.loc[(target_df["ppm"]>min(results_dict[k]["coords"])-search_region_padding_size) &
                                              (target_df["ppm"]<max(results_dict[k]["coords"])+search_region_padding_size)
                                             ].copy().reset_index(drop=True)

        # set minimum intensity to 0 for query and target, if user-input floor_window is true
        if floor_window:
            query_intensity_arr = query_window["intensity"].values - min(query_window["intensity"].values)
            query_window["intensity"] = query_intensity_arr

            target_intensity_arr = search_target["intensity"].values - min(search_target["intensity"].values)
            search_target["intensity"] = target_intensity_arr

        rho_ls = []
        for i in range(0, len(search_target) - window_size):
            target_window = search_target.iloc[i:i+window_size]
            rho_ls.append(norm_xcorr(target_window["intensity"].values, query_window["intensity"].values))
        rho_ls = np.array(rho_ls)

        # save results in results_dict
        results_dict[k]["rho_ls"] = np.array(rho_ls)
        multiplet_match_idx_ls = np.where(rho_ls > np.percentile(rho_ls, q_th))[0]
        multiplet_match_idx_ls2 = multiplet_match_trimming(multiplet_match_idx_ls, 4)
        results_dict[k]["multiplet_match_idx"] = multiplet_match_idx_ls2
        if len(multiplet_match_idx_ls2) > 1:
            print(F"WARNING: {len(multiplet_match_idx_ls2)} matches found!")

        results_dict[k]["max_rho"] = [results_dict[k]["rho_ls"][idx] for idx in multiplet_match_idx_ls2]

        # add matching ppm coords [x0, x1] corresponding to match_indx
        multiplet_match_ppm_ls = []
        for j in range(len(results_dict[k]["multiplet_match_idx"])):
            coords_temp = [search_target.iloc[results_dict[k]["multiplet_match_idx"][j]].ppm, 
            search_target.iloc[results_dict[k]["multiplet_match_idx"][j]].ppm - mcoord_width
            ]
            multiplet_match_ppm_ls.append(coords_temp)
        results_dict[k]["multiplet_match_ppm"] = multiplet_match_ppm_ls

        # get ppm_shift
        results_dict[k]["ppm_shift"] = max(results_dict[k]["multiplet_match_ppm"][0]) - max(query_window.ppm.values)

    return results_dict


def do_2d_std_search(query_df, 
                    target_df, 
                    multiplets_ls, 
                    mask_subdict, 
                    normxcorr2_mode="same", 
                    search_region_padding_size=0.01, 
                    patch_x_padding=5, 
                    patch_y_padding=10
                    ):
    """
    Wraps normxcorr2 for convenience. This function written to mirror do_1d_std_search()
    For a single std, for each multiplet coord in multiplets_ls, search for `query` in `target`.
    If the multiplet index appears in mask_subdict for that std, use sub-patch matching instead.

    PARAMS
    ------
    query_df: df; JRES df of query (standard)
    target_df: df; JRES df of target (sample)
    multiplets_ls: list of multiplet coords
    mask_subdict: dict of subpatch coords of particular multiplets of that standard. 
        Each sub-patch coords must be in the format [x0, y0, x1, y1]
        Input an empty dictionary {} to disable this functionality.
    normxcorr2_mode: "full", "valid", "same". See docstring for normxcorr2(). 
    search_region_padding_size: padding on either side of search region, in ppm. 
    patch_x_padding: int; amount of horizontal wiggle room for subpatch matching, in indices.
    patch_y_padding: int; amount of vertical wiggle room for subpatch matching, in indices.

    RETURNS
    -------
    results_dict_2d: dictionary of results.
        coords: input multiplet coords, straight from multiplets_ls
        rho_arr: 2D corr heatmap. Can use imshow() to visualize this.
        max_rho: maximum correlation value in rho_arr
        match_coords_ls_idx: list of (x, y) coords of the points of max corr. Usually of only length 1, may be >1 if there are ties (extremely unlikely)
        match_coords_ls_ppm: list of ppm values corresponding to each y-coord in match_coords_ls_idx.
    """
    # init results_2d_dict
    results_2d_dict = {}
    # run
    idx = 0
    for multiplet in multiplets_ls:
        multiplet_str = f"multiplet_{idx}"
        results_2d_dict[multiplet_str] = {}

        query_window = query_df.loc[(query_df["ppm"] > min(multiplet)) & (query_df["ppm"] < max(multiplet))].copy()
        target_window = target_df.loc[(target_df["ppm"] > (min(multiplet)-search_region_padding_size)) & (target_df["ppm"] < (max(multiplet)+search_region_padding_size))].copy()
        query_arr = np.transpose(query_window.drop("ppm", axis=1).values)
        target_arr = np.transpose(target_window.drop("ppm", axis=1).values)

        # do subpatch matching if relevant
        if str(idx) in list(mask_subdict.keys()):
            rho_ls_ls, idx_max_ls, rho_ls_aggregate = subpatch_matching(query_window, 
                target_window, 
                mask_subdict[str(idx)], 
                patch_x_padding=patch_x_padding, 
                patch_y_padding=patch_y_padding)

            results_2d_dict[multiplet_str]["rho_arr"] = rho_ls_ls
            results_2d_dict[multiplet_str]["max_rho"] = rho_ls_aggregate
            results_2d_dict[multiplet_str]["match_coords_ls_idx"] = idx_max_ls
            results_2d_dict[multiplet_str]["match_coords_ls_ppm"] = [-1.0]
            results_2d_dict[multiplet_str]["method"] = "subpatch matching"

        # else do whole-patch normxcorr search
        else:
            # main calc: normxcorr2
            rho_arr = normxcorr2(query_arr, target_arr, mode=normxcorr2_mode)
            # get match location(s) in indices
            match_coords_ls_idx = np.transpose(np.array(np.where(rho_arr == np.amax(rho_arr))))
            # get match location(s) in ppm
            match_coords_ls_ppm = [target_window.iloc[match[1]].ppm for match in match_coords_ls_idx]

            results_2d_dict[multiplet_str]["coords"] = multiplet
            results_2d_dict[multiplet_str]["rho_arr"] = rho_arr
            results_2d_dict[multiplet_str]["max_rho"] = np.max(rho_arr)
            results_2d_dict[multiplet_str]["match_coords_ls_idx"] = match_coords_ls_idx
            results_2d_dict[multiplet_str]["match_coords_ls_ppm"] = match_coords_ls_ppm
            results_2d_dict[multiplet_str]["method"] = "whole query matching"
        
        # record final bits of results
        results_2d_dict[multiplet_str]["coords"] = multiplet
        results_2d_dict[multiplet_str]["normxcorr2_mode"] = normxcorr2_mode

        idx += 1

    return results_2d_dict


def viz_enema(query_df,
              target_df,
              query_2d_df,
              target_2d_df,
              target_name,
              results_dict,
              results_2d_dict,
              title_fontsize=30,
              search_region_padding_size_1d=0.01, 
              search_region_padding_size_2d=0.01
             ):
    """Plotting function to plot 1D and 2D normxcorr results, for a single std.
    WARNING: when referencing the highest-matching location of a multiplet, we persistently select only the first matching position. This assumes that one can never find multiplet-search will never find two highest matching locations that have the exact norm_xcorr score (not a terrible assumption to make). 

    PARAMS
    ------
    query_df: df of the query spectrum (usually the pure standard), with columns 'ppm' and 'intensity'
    target_df: df of the target spectrum (usually the sample), with columns 'ppm' and 'intensity'
    query_2d_df: df of the query JRES, typically of shape (n_row, 257). Not that one of the columns is the 'ppm' column.
    target_2d_df: df of the target JRES, typically of shape (n_row, 257). Not that one of the columns is the 'ppm' column.
    target_name: str; name of the target. Really only used for labelling purposes: plot labels, axis labels, etc.
    results_dict: output from do_1d_std_search()
    results_2d_dict: output from do_2d_std_search()
    title_fontsize: int; title font size. default = 30.
    search_region_padding_size_1d: float; 
    search_region_padding_size_2d: float; 

    RETURNS
    -------
    output_svg: figure as svg. 
    """

    ### =============== for >1 multiplets ===============
    if len(results_dict.keys()) > 1:
        fig, axarr = plt.subplots(nrows=5,
                                  ncols=len(results_dict.keys()),
                                  figsize=(8*len(results_dict.keys()), 25)
                                 )

        search_target_max_intensity = -1
        for i in range(len(results_dict.keys())):
            k = list(results_dict.keys())[i]

            # ===== plot 1d =====
            search_target = target_df.loc[(target_df["ppm"]>min(results_dict[k]["coords"])-search_region_padding_size_1d) &
                                          (target_df["ppm"]<max(results_dict[k]["coords"])+search_region_padding_size_1d)
                                         ].copy().reset_index(drop=True)
            query_window = query_df.loc[(query_df["ppm"]>min(results_dict[k]["coords"])) &
                                                  (query_df["ppm"]<max(results_dict[k]["coords"]))
                                                 ].copy().reset_index(drop=True)
            # autoscale to emulate norm_xcorr calculation behaviour
            search_target_intens_normalized = (search_target.intensity.values - np.mean(search_target.intensity.values))/np.std(search_target.intensity.values)
            query_window_intens_normalized = (query_window.intensity.values - np.mean(query_window.intensity.values))/np.std(query_window.intensity.values)

            axarr[1, i].plot(search_target.ppm.values, search_target_intens_normalized, c="steelblue")
            axarr[1, i].plot(query_window.ppm.values+max(results_dict[k]["multiplet_match_ppm"][0])-max(query_window.ppm.values), query_window_intens_normalized, c="indianred")
            axarr[1, i].set_xlim([max(search_target.ppm.values), min(search_target.ppm.values)])
            match_centre_ppm = max(results_dict[k]["multiplet_match_ppm"][0]) - results_dict[k]["multiplet_len_ppm"]/2
            subtitle = f"{k}\nMax normxcorr_1d = {round(np.max(results_dict[k]['rho_ls']), 3)}"
            subtitle += f"@{round(match_centre_ppm, 4)}"
            if "scaling_factor_1d" in results_dict[k].keys():
                subtitle += f"s.f.={round(results_dict[k]['scaling_factor_1d'], 3)}"
            axarr[0, i].set_title(subtitle, fontsize=title_fontsize)

            # draw a rectangle reflecting normxcorr score
            y0 = max(results_dict[k]["multiplet_match_ppm"][0])
            #y1 = y0 - max(results_dict[k]["multiplet_len_ppm"])
            y1 = min(results_dict[k]["multiplet_match_ppm"][0])
            selection_c = "#72EC8F" # green
            if (np.max(results_dict[k]['rho_ls']) < 0.9) and (np.max(results_dict[k]['rho_ls']) >= 0.8):
                selection_c = "#FFC01E" # orange
            elif np.max(results_dict[k]['rho_ls']) < 0.8:
                selection_c = "#F83A58" # red
            axarr[1, i].axvspan(y0, y1, alpha=0.35, color=selection_c)

            # ===== plot 2d =====
            multiplet = results_2d_dict[k]["coords"]
            query_window = query_2d_df.loc[(query_2d_df["ppm"] > min(multiplet)) & (query_2d_df["ppm"] < max(multiplet))].copy()
            target_window = target_2d_df.loc[(target_2d_df["ppm"] > (min(multiplet)-search_region_padding_size_2d)) & (target_2d_df["ppm"] < (max(multiplet)+search_region_padding_size_2d))].copy()

            query_arr = np.transpose(query_window.drop("ppm", axis=1).values)
            target_arr = np.transpose(target_window.drop("ppm", axis=1).values)

            query_border_width = int((target_arr.shape[1] - query_arr.shape[1])/2)
            #query_display = np.zeros_like(target_arr)
            query_display = np.ones(target_arr.shape)
            query_display[:, query_border_width:query_border_width+query_arr.shape[1]] = query_arr
            # Add a left- and right- border of ones
            #query_display[:, query_border_width-1] = np.ones(query_arr.shape[0])
            #query_display[:, query_border_width+query_arr.shape[1]+1] = np.ones(query_arr.shape[0])

            axarr[2, i].imshow(query_display, aspect="auto", cmap="coolwarm")
            axarr[3, i].imshow(target_arr, aspect="auto", cmap="coolwarm")
            axarr[4, i].imshow(results_2d_dict[k]["rho_arr"], aspect="auto", vmin=0, vmax=1, cmap="coolwarm")

            # Draw all subtitles
            axarr[0, 0].set_ylabel("1D un-scaled", fontsize=title_fontsize)
            axarr[1, 0].set_ylabel("1D scaled", fontsize=title_fontsize)
            axarr[2, 0].set_ylabel("multiplet J-Res", fontsize=title_fontsize)
            axarr[3, 0].set_ylabel(f"{target_name} J-Res", fontsize=title_fontsize)
            axarr[4, 0].set_ylabel(f"normxcorr2", fontsize=title_fontsize)
            subtitle = f"Max normxcorr2 = {round(np.max(results_2d_dict[k]['rho_arr']), 3)}@{round(results_2d_dict[k]['match_coords_ls_ppm'][0], 4)}"
            axarr[4, i].set_title(subtitle, fontsize=title_fontsize)

            # Draw vline where a match occurs
            for match_coord in results_2d_dict[k]["match_coords_ls_idx"]:
                axarr[3, i].axvline(match_coord[1], c="red")

            # Set font sizes
            for row_idx in np.arange(5):
                axarr[row_idx, i].xaxis.set_tick_params(labelsize=20)
                axarr[row_idx, i].yaxis.set_tick_params(labelsize=20)

            if max(search_target.intensity.values) > search_target_max_intensity:
                search_target_max_intensity = max(search_target.intensity.values)

            i += 1

        for i in range(len(results_dict.keys())):
            k = list(results_dict.keys())[i]
            search_target = target_df.loc[(target_df["ppm"]>min(results_dict[k]["coords"])-search_region_padding_size_1d) &
                                          (target_df["ppm"]<max(results_dict[k]["coords"])+search_region_padding_size_1d)
                                         ].copy().reset_index(drop=True)
            axarr[0, i].plot(search_target.ppm.values, search_target.intensity.values, c="steelblue")
            axarr[0, i].set_xlim([max(search_target.ppm.values), min(search_target.ppm.values)])
            axarr[0, i].set_ylim([0, search_target_max_intensity*1.1])

    # =============== for just 1 multiplet ===============
    else:
        fig, axarr = plt.subplots(nrows=4,
                                  ncols=len(results_dict.keys()),
                                  figsize=(8*len(results_dict.keys()), 25)
                                 )

        k = list(results_dict.keys())[0]
        # ===== plot 1d =====
        search_target = target_df.loc[(target_df["ppm"]>min(results_dict[k]["coords"])-search_region_padding_size_1d) &
                                      (target_df["ppm"]<max(results_dict[k]["coords"])+search_region_padding_size_1d)
                                     ].copy().reset_index(drop=True)
        query_window = query_df.loc[(query_df["ppm"]>min(results_dict[k]["coords"])) &
                                              (query_df["ppm"]<max(results_dict[k]["coords"]))
                                             ].copy().reset_index(drop=True)

        search_target_intens_normalized = (search_target.intensity.values - np.mean(search_target.intensity.values))/np.std(search_target.intensity.values)
        query_window_intens_normalized = (query_window.intensity.values - np.mean(query_window.intensity.values))/np.std(query_window.intensity.values)

        axarr[0].plot(search_target.ppm.values, search_target_intens_normalized, c="steelblue")
        axarr[0].plot(query_window.ppm.values+max(results_dict[k]["multiplet_match_ppm"][0])-max(query_window.ppm.values), query_window_intens_normalized, c="indianred")
        axarr[0].set_xlim([max(search_target.ppm.values), min(search_target.ppm.values)])

        match_centre_ppm = max(results_dict[k]["multiplet_match_ppm"][0]) - results_dict[k]["multiplet_len_ppm"]/2
        subtitle = f"{k}\nMax normxcorr_1d = {round(np.max(results_dict[k]['rho_ls']), 3)}"
        subtitle += f"@{round(match_centre_ppm, 4)}"
        if "scaling_factor_1d" in results_dict[k].keys():
            subtitle += f"s.f.={round(results_dict[k]['scaling_factor_1d'], 3)}"
        axarr[0].set_title(subtitle, fontsize=title_fontsize)

        y0 = max(results_dict[k]["multiplet_match_ppm"][0])
        y1 = min(results_dict[k]["multiplet_match_ppm"][0])
        axarr[0].axvspan(y0, y1, alpha=0.5, color="#72EC8F")

        # ===== plot 2d =====
        multiplet = results_2d_dict[k]["coords"]
        query_window = query_2d_df.loc[(query_2d_df["ppm"] > min(multiplet)) & (query_2d_df["ppm"] < max(multiplet))]
        target_window = target_2d_df.loc[(target_2d_df["ppm"] > min(multiplet)-search_region_padding_size_2d) & (target_2d_df["ppm"] < max(multiplet)+search_region_padding_size_2d)]

        query_arr = np.transpose(query_window.drop("ppm", axis=1).values)
        target_arr = np.transpose(target_window.drop("ppm", axis=1).values)

        query_border_width = int((target_arr.shape[1] - query_arr.shape[1])/2)
        #pad query_display array on left and right so that it's the same size as target
        query_display = np.ones(target_arr.shape)
        query_display[:, query_border_width:query_border_width+query_arr.shape[1]] = query_arr

        axarr[1].imshow(query_display, aspect="auto", cmap="coolwarm")
        axarr[2].imshow(target_arr, aspect="auto", cmap="coolwarm")
        #axarr[2].set_xticklabels(np.around(target_window.ppm.values[::5], 3))
        axarr[3].imshow(results_2d_dict[k]["rho_arr"], aspect="auto", vmin=0, vmax=1, cmap="coolwarm")

        # Draw all subtitles
        axarr[1].set_ylabel("multiplet J-Res", fontsize=title_fontsize)
        axarr[2].set_ylabel(f"{target_name} J-Res", fontsize=title_fontsize)
        subtitle = f"Max normxcorr2 = {round(np.max(results_2d_dict[k]['rho_arr']), 3)}@{round(results_2d_dict[k]['match_coords_ls_ppm'][0], 4)}"
        axarr[3].set_title(subtitle, fontsize=title_fontsize)

        # Draw vline where a match occurs
        for match_coord in results_2d_dict[k]["match_coords_ls_idx"]:
            axarr[2].axvline(match_coord[1], c="red")

        # Set font sizes
        axarr[0].xaxis.set_tick_params(labelsize=20)
        axarr[0].yaxis.set_tick_params(labelsize=20)
        axarr[1].xaxis.set_tick_params(labelsize=20)
        axarr[1].yaxis.set_tick_params(labelsize=20)
        axarr[2].xaxis.set_tick_params(labelsize=20)
        axarr[2].yaxis.set_tick_params(labelsize=20)

    plt.tight_layout()
    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # save as svg
    i = StringIO()
    fig.savefig(i, format="svg")
    output_svg = i.getvalue()
    plt.close()

    return output_svg


def resize_svg(svg_ls, resize_coeff=0.25):
    """Scale down an svg image obtained from matplotlib, because it's usually too large. Usually used after some kind of plotting function.
    
    PARAMS
    ------
    svg_ls: svg content, as a list of strings
    resize_coeff: float; coefficient to multiply width and height by. Set to <1 to shrink, >1 to expand, =1 to do nothing.
    
    RETURNS
    -------
    List of strings, with resized svg width and height (but not viewbox)
    """
    for i in range(len(svg_ls)):
        if "<svg height=" in svg_ls[i]:
            pattern = re.compile(r'width="\d{3,8}pt" xmlns')
            width_match = re.findall(pattern, svg_ls[i])[0]
            width_val = int(width_match.replace('width="', '').replace('pt" xmlns', ''))
            new_width_str = f'width="{str(int(width_val*resize_coeff))}pt" xmlns'
            
            pattern = re.compile(r'<svg height="\d{3,8}pt" ')
            height_match = re.findall(pattern, svg_ls[i])[0]
            height_val = int(height_match.replace('<svg height="', '').replace('pt"', ''))
            new_height_str = f'<svg height="{str(int(height_val*resize_coeff))}pt"'

            # replace
            new_ln = svg_ls[i].replace(height_match, new_height_str).replace(width_match, new_width_str)
            svg_ls[i] = new_ln
            
    return svg_ls


def resize_svg_bs4(svg_str, resize_coeff=0.25):
    """
    # common usage:
    # plt.show() not required
    plt.subplots_adjust(hspace=0.1, wspace=0)
    i = StringIO()
    plt.tight_layout()
    fig.savefig(i, format="svg")
    hm_svg = i.getvalue()
    hm_svg = resize_svg_bs4(hm_svg, resize_coeff=0.5)
    plt.close()

    # add to line by line list, split by '\n'
    c = []
    for line in str(hm_svg).strip().split("\n"):
        c.append(line)

    # print out the list line by line:
    with open(fn_out, "w") as f:
        for line in c:
            f.write(line)
    """
    soup = BeautifulSoup(svg_str, features="html.parser")

    # Find the SVG tags
    svg_tags = soup.find_all('svg')
    
    for svg_tag in svg_tags:
        # Get the current width and height attributes
        current_width = svg_tag.get('width')
        current_height = svg_tag.get('height')
    
        if current_width is not None and current_height is not None:
            # Convert width and height to float, multiply by 0.5
            new_width = str(float(current_width.replace("pt", "")) * resize_coeff)
            new_height = str(float(current_height.replace("pt", "")) * resize_coeff)
    
            # Update the width and height attributes
            # does replacement in-place
            svg_tag['width'] = new_width
            svg_tag['height'] = new_height
    
    return soup


def resize_and_save_svg_str(svg_str, fn_out, resize_coeff=0.25):
    """Exactly what it says on the can: splits an svg string (direct output from viz_enema()) by delimiter '\\n', resizes it with resize_svg(), and writes out to file. 

    PARAMS
    ------
    svg_str: a string of svg content. Can be direct output from viz_enema().
    fn_out: absolute/path/to/output.html
    resize_coeff: float; coefficient to multiply width and height by. Set to <1 to shrink, >1 to expand, =1 to do nothing.

    RETURNS
    -------
    Returns nothing except an SVG file written out. 
    """

    svg_ls = resize_svg(svg_str.split("\n"), resize_coeff=resize_coeff)

    with open(fn_out, "w") as f:
        for line in svg_ls:
            f.write("%s\n" % line)

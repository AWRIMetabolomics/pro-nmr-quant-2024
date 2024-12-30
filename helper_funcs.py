"""
A bunch of utility functions, and rather too specific to proline. Functions are used by the targeted workflow and std_finder workflow.
The key object being manipulated here is a dictionary of dfs, where each df has two columns: ppm (float) and intensity (float). This is because this is meant to be batch processed. 
"""
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy import integrate
from scipy.signal import fftconvolve
from scipy.stats import pearsonr
from bs4 import BeautifulSoup

import math
import re
from sklearn.linear_model import LinearRegression

import os
from io import StringIO


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

    # handling zero-div-error. 
    # Variance can be 0 for constant arrays, which can happen in suppressed regions
    norm_xcorr = 0
    if denominator != 0:
        norm_xcorr = numerator/denominator

    return norm_xcorr


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


def normxcorr_1d_search_classic(red_sum_loz, blue_sum_loz):
    """DEPRECATED, superceded by normxcorr_1d_fast_search(). Note that normxcorr_1d_fast_search with stepsize=1 is the same as normxcorr_1d_search_classic().
    A wrapper for norm_xcorr; slides a shorter template (np.array), over a longer target (np.array), running norm_xcorr() at each step, to find the location and value of max norm_xcorr of template in target. Error out if len(template) > len(target). In practice, len(target) can be set to be >= len(template) if a nonzero search_window_padding_size is used. 

    PARAMS
    =======
    red_sum_loz (template): 1D np array of float. 
    blue_sum_loz (target): 1D np array of float. 

    RETURNS
    =======
    rho_ls: list of normxcorr scores. In practice, you'll take the max
    """
    
    rho_ls = []
    # if len(template) < len(target):
    #     window_size = len(template)
    #     for i in range(0, len(target) - window_size):
    #         target_window = target[i:i+window_size]
    #         #rho_ls.append(norm_xcorr(target_window, template))

    #         # replace with built-in numpy correlation func, which returns a corr_matrix
    #         # so select the off-diagonal element to get correlation
    #         # returns nan and throws a warning if one of the arrays is an array of constants, 
    #         # which can happen in sparse regions
    #         rho_ls.append(np.corrcoef(target_window, template)[0][1])
    #     rho_ls = np.array(rho_ls)

    # elif len(template) == len(target):
    #     rho = norm_xcorr(template, target)
    #     rho_ls = np.array([rho]) # return an array of len 1

    if len(blue_sum_loz) >= len(red_sum_loz):
        # get partial rho_ls
        num_windows = len(blue_sum_loz) - len(red_sum_loz) + 1
        blue_window_matrix = np.array([blue_sum_loz[i:i+len(red_sum_loz)] for i in range(num_windows)])
        rho_ls = [np.corrcoef(blue_window_matrix[i, :], red_sum_loz)[0][1] for i in range(0, len(blue_window_matrix), 1)]
        #max_rho = np.max(rho_ls)

    elif len(red_sum_loz) > len(blue_sum_loz):
        print("ERROR: len(red_sum_loz) > len(blue_sum_loz)!")
        # leaves rho_ls unmodified, to return an empty array

    return rho_ls


def normxcorr_1d_fast_search(red_sum_loz, blue_sum_loz, stepsize=3, neighbourhood_size_padding=1):
    """Calc correlation for every other sliding window, instead of every sliding window (to reduce runtime by half). Searches for best-matching position of red_sum_loz in blue_sum_loz in a 2-step process:
    1. Do a sliding window search with a stepsize >1 (stepsize=1 is a full search) to find a first prediction of the best-matching position. 
    2. Search the immediate neighbourhood of the first prediction to see if any of the neighbours have a higher correlation. 
    
    PARAMS
    ======
    red_sum_loz: array of float. 
    blue_sum_loz: array of float. len(blue_sum_loz) must be >= len(red_sum_loz), which should be ensure by having a nonzero search_region_padding_size upstream of this function. 
    stepsize: int; number of indices to skip for rolling window. stepsize=1 means check every index, stepsize=2 means check every other index, and so on. 
    neighbourhood_size: int. Len of smaller window for full scan = neighbourhood_size*2 + 1, where: neighbourhood_size = neighbourhood_size_padding + stepsize. This is to ensure that neighbourhood_size*0.5 is always >= stepsize.

    RETURNS
    =======
    rho_ls: 
    max_rho: 
    """
    rho_ls = [] # init
    max_rho = -1
    idx_of_max = -1
    neighbourhood_size = neighbourhood_size_padding + stepsize
    if len(blue_sum_loz) >= len(red_sum_loz):
        # get partial rho_ls
        num_windows = len(blue_sum_loz) - len(red_sum_loz) + 1
        # create matrix of target windows; each row is one target window
        blue_window_matrix = np.array([blue_sum_loz[i:i+len(red_sum_loz)] for i in range(num_windows)])
        # search every other row, instead of every row. This halves runtime.
        rho_ls = [np.corrcoef(blue_window_matrix[i, :], red_sum_loz)[0][1] for i in range(0, len(blue_window_matrix), stepsize)]
        # get first prediction
        idx_pred_1 = np.argmax(rho_ls)*stepsize

        # create smaller scan window to check immediate neighbours
        start_idx = max(0, idx_pred_1-neighbourhood_size)
        end_idx = min(idx_pred_1+neighbourhood_size+1, len(blue_window_matrix))
        blue_window_matrix_xs = blue_window_matrix[start_idx:end_idx, :]
        rho_ls_small = [np.corrcoef(blue_window_matrix_xs[i, :], red_sum_loz)[0][1] for i in range(0, len(blue_window_matrix_xs))]
        max_rho = np.max(rho_ls_small)

        pred_index_offset = max(neighbourhood_size, 0)
        if idx_pred_1 <= (neighbourhood_size*2) + 1:
            idx_pred_2 = idx_pred_1 + np.argmax(rho_ls_small) - pred_index_offset
        else:
            idx_pred_2 = idx_pred_1
        
    elif len(blue_sum_loz) < len(red_sum_loz):
        print("ERROR: len(template) > len(target)!")

    return rho_ls, max_rho, idx_pred_1, idx_pred_2


def normxcorr_1d_search_UNUSED(red_sum_loz, blue_sum_loz):
    """Old normxcorr_1d_search function, superseded by normxcorr_1d_fast_search().
    """
    rho_ls = [] # init
    max_rho = -1
    if len(blue_sum_loz) >= len(red_sum_loz):
        # get partial rho_ls
        num_windows = len(blue_sum_loz) - len(red_sum_loz) + 1
        blue_window_matrix = np.array([blue_sum_loz[i:i+len(red_sum_loz)] for i in range(num_windows)])
        
        rho_ls = [np.corrcoef(blue_window_matrix[i, :], red_sum_loz)[0][1] for i in range(0, len(blue_window_matrix), 1)]
        max_rho = np.max(rho_ls)
        
    elif len(blue_sum_loz) < len(red_sum_loz):
        print("ERROR: len(template) > len(target)!")

    return rho_ls, max_rho


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
        m_k = f"multiplet_{idx}"
        results_2d_dict[m_k] = {}

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

            results_2d_dict[m_k]["rho_arr"] = rho_ls_ls
            results_2d_dict[m_k]["max_rho"] = rho_ls_aggregate
            results_2d_dict[m_k]["match_coords_ls_idx"] = idx_max_ls
            results_2d_dict[m_k]["match_coords_ls_ppm"] = [-1.0]
            results_2d_dict[m_k]["method"] = "subpatch matching"

        # else do whole-patch normxcorr search
        else:
            # main calc: normxcorr2
            rho_arr = normxcorr2(query_arr, target_arr, mode=normxcorr2_mode)
            # get match location(s) in indices
            match_coords_ls_idx = np.transpose(np.array(np.where(rho_arr == np.amax(rho_arr))))
            # get match location(s) in ppm
            match_coords_ls_ppm = [target_window.iloc[match[1]].ppm for match in match_coords_ls_idx]

            results_2d_dict[m_k]["coords"] = multiplet
            results_2d_dict[m_k]["rho_arr"] = rho_arr
            results_2d_dict[m_k]["max_rho"] = np.max(rho_arr)
            results_2d_dict[m_k]["match_coords_ls_idx"] = match_coords_ls_idx
            results_2d_dict[m_k]["match_coords_ls_ppm"] = match_coords_ls_ppm
            results_2d_dict[m_k]["method"] = "whole query matching"
        
        # record final bits of results
        results_2d_dict[m_k]["coords"] = multiplet
        results_2d_dict[m_k]["normxcorr2_mode"] = normxcorr2_mode

        idx += 1

    return results_2d_dict


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


def set_min_intensity_to_zero(df):
    """Set minimum intensity to zero. UNTESTED, because it probably shouldn't be used. 
    """
    df2 = df.copy()
    temp_ls = df2["intensity"].values - min(df2["intensity"].values)
    df2["intensity"] = temp_ls
    
    return df2


def get_starting_template(template_df, mcoords, ppm_shift):
    """
    PARAMS
    ------
    template_df: 
    mcoords: list of len 2; multiplet coords.
    ppm_shift: float; to be added to ppm to shift the template to the best-matched position, relative to some sample.
    
    RETURNS
    -------
    dt: 
    """
    intensity_init = 5E7

    dt = template_df.copy()
    dt = dt.loc[(dt["ppm"]>min(mcoords)) & (dt["ppm"]<max(mcoords))]
    sf_constant = intensity_init/max(dt["intensity"].values)
    temp_ls = dt.intensity.values
    dt["intensity"] = temp_ls * sf_constant
    temp_ls = dt["ppm"].values + ppm_shift
    dt["ppm"] = temp_ls
    
    return dt


def align_spectra_dfs(df1, df2):
    """Aligns 2 dfs by sliding the `ppm` column of df1 over the `ppm` column of df2 until their element-wise difference is minimized.
    Note the following:
    * len(df1) must be <= len(d2). 
    * Both dfs must have a `ppm` column. This will be returned as `ppm1` and `ppm2`, following input order, in the final returned df.
    * Each df can contain any other column of any name.
    * both dfs may be trimmed until they have the same length, since they must be merged into a final df.
    
    PARAMS
    ------
    df1: 
    df2: 
    
    RETURNS
    -------
    dz: merged df with aligned columns.
    """
    # sort by ppm, ascending
    df1 = df1.sort_values(by="ppm", ascending=True).reset_index(drop=True)
    df2 = df2.sort_values(by="ppm", ascending=True).reset_index(drop=True)
    ppm1 = df1["ppm"].values
    ppm2 = df2["ppm"].values

    start = max(ppm1[0], ppm2[0])
    end = min(ppm1[-1], ppm2[-1])

    dt1_out = df1.copy()
    dt1_out = dt1_out.loc[(dt1_out["ppm"]>=start) & (dt1_out["ppm"]<=end)].rename(columns={"ppm":"ppm1", "intensity":"intensity1"}).reset_index(drop=True)
    dt2_out = df2.copy()
    dt2_out = dt2_out.loc[(dt2_out["ppm"]>=start) & (dt2_out["ppm"]<=end)].rename(columns={"ppm":"ppm2", "intensity":"intensity2"}).reset_index(drop=True)

    # NaNs sometimes occur at the first or law row due to off-by-one errors
    dz = pd.concat([dt1_out, dt2_out], axis=1).dropna()
    
    return dz


def get_err_bounds(red_array, blue_array, start, end, iter_size, convergence_method="mse"):
    """Iterates over red_array*sf - blue_array, and returns the first sf where err becomes positive.
    Used to iteratively grow red_array (typically std) until it just meets blue_array (typically sample).

    PARAMS
    ------
    red_array: array of float, usually std
    blue_array: array of float, usually sample
    start: float; starting value of range of scaling factors
    end: float; ending value of range of scaling factors
    iter_size: float; interval value of scaling factor
    convergence_method: str; error type for iteration.
        'bottomup': scales red to blue such that red never exceeds blue. 
        'mse': scales red to blue such that MSE is minimized. 
        'both': error value is a sum of bottomup and mse, with a weight, `w`, on mse. 

    RETURNS
    -------
    err_positive_sf: float; converged scaling factor. 
    """
    err_is_positive_bool = False
    err_positive_sf = start
    
    if convergence_method=="bottomup":
        for sf in np.arange(start, end, iter_size):
            # red -  blue
            err = (red_array * sf) - blue_array
            # replace all negative values with 0
            err[err < 0] = 0

            if np.average(err) > 0 and not err_is_positive_bool:
                err_positive_sf = sf

            if np.average(err) > 0:
                err_is_positive_bool = True

    if convergence_method=="mse":
        # set bias factor to double error values where red > blue
        # use bias_factor = 1.0 to do nothing
        bias_factor = 2.0
        err = (red_array * start) - blue_array
        for i in range(len(err)):
            if err[i] < 0:
                new_val = err[i] * bias_factor
                err[i] = new_val

        mse_min = np.sum(np.square(err))
        err_positive_sf = start
        for sf in np.arange(start, end, iter_size):
            err = (red_array * sf) - blue_array
            for i in range(len(err)):
                if err[i] < 0:
                    new_val = err[i] * bias_factor
                    err[i] = new_val
            mse = np.sum(np.square(err))
            if mse < mse_min:
                mse_min = mse
                err_positive_sf = sf
    
    if convergence_method=="both":
        mse_err = start
        bottomup_err = start
        err_is_positive_bool = False
        err_ls = []
        w = 2.0 # coefficient to scale mse error
        mse_min = np.sum(np.square((red_array * start) - blue_array))
        for sf in np.arange(start, end, iter_size):
            # get MSE error
            mse = np.sum(np.square((red_array * sf) - blue_array))

            # red -  blue
            err = (red_array * sf) - blue_array
            # replace all negative values with 0
            err[err < 0] = 0

            if np.average(err) > 0 and not err_is_positive_bool:
                bottomup_err = sf

            if np.average(err) > 0:
                err_is_positive_bool = True            

            total_err = bottomup_err + (w*mse_err)
            
    return err_positive_sf


def fit_std_to_sample(template_df, sample_df, mcoords, ppm_shift, convergence_method="mse", set_intensity_min_to_zero_bool=False):
    """One giant wrapper for get_starting_template(), align_spectra_dfs(), get_err_bounds(). DEFUNCT?
    
    PARAMS
    ------
    template_df: original, unshifted, full template_df. 
    sample_df: original, unshifted, full sample_df.
    mcoords: list of len 2; multiplet coords.
    ppm_shift: float; added to template_df `ppm` column.
    convergence_method: str; passed to get_err_bounds()
    set_intensity_min_to_zero_bool: bool; whether or not to floor min intensity to zero
    
    RETURNS
    -------
    sf2: float; height scaling factor
    auc: float; AUC of scaled template.
    """
    # call get_starting_template() from template
    dt1 = get_starting_template(
        template_df=template_df.copy(), 
        mcoords=mcoords, 
        ppm_shift=ppm_shift
    )
    
    # extract sample
    dt2 = sample_df.copy()
    dt2 = dt2.loc[(dt2["ppm"]>min(mcoords)) & (dt2["ppm"]<max(mcoords))]

    if set_intensity_min_to_zero_bool:
        dt1 = set_min_intensity_to_zero(dt1)
        dt2 = set_min_intensity_to_zero(dt2)
    
    # align
    dz = align_spectra_dfs(dt1, dt2)
    red_arr = dz["intensity1"].values
    blue_arr = dz["intensity2"].values

    # Iterate until red grows to meet blue
    sf0 = get_err_bounds(red_arr, blue_arr, 1, 1000, 1, convergence_method=convergence_method)
    sf1 = get_err_bounds(red_arr, blue_arr, sf0-1.0, sf0+1.0, 0.1, convergence_method=convergence_method)
    sf2 = get_err_bounds(red_arr, blue_arr, sf1-0.1, sf1+0.1, 0.00001, convergence_method=convergence_method)
    
    auc = np.sum(red_arr*sf2)
    
    return sf2, auc


def get_sf_range(start, end, iter_size):
    """Generate full sf range to iterate over if start, end are float numbers which don't work well with np.arange.
    """
    total_num_iter = math.floor((end - start)/iter_size)
    sf_ls = (np.arange(total_num_iter)*iter_size) + start
    sf_ls = np.append(sf_ls, end)
    return sf_ls


def get_true_sf(template_df, mcoords, ppm_shift, bottomup_sf):
    """Calculate an sf which acts on original, un-scaled template to scale to best-fit with sample
    Intended to be used after red has already been scaled to blue in the bottom-up method.
    Wraps get_starting_template().

    PARAMS
    ------
    template_df: origin, un-scaled template_df
    mcoords: list of len 2; multiplet coords.
    ppm_shift: float;
    bottomup_sf: float;

    RETURNS
    -------
    sf_true: float
    """
    dt = get_starting_template(
        template_df=template_df, 
        mcoords=mcoords, 
        ppm_shift=ppm_shift
    )
    temp_ls = dt["intensity"].values*bottomup_sf
    dt["intensity"] = temp_ls

    dt2 = template_df.loc[(template_df["ppm"]>min(mcoords)) & (template_df["ppm"]<max(mcoords))].copy()
    temp_ls = dt2["ppm"].values + ppm_shift
    dt2["ppm"] = temp_ls
    sf_true = np.average(dt2.intensity.values/dt["intensity"].values)

    return sf_true


def get_sf_min(red_array, blue_array, sf_ls):
    """Get the minimum MSE, and associated scaling factor, sf.
    """
    sf_min = sf_ls[0]
    mse_min = np.sum((red_array * sf_ls[0]) - blue_array)
    for sf in sf_ls:
        mse = np.sum((red_array * sf) - blue_array)
        if mse < mse_min:
            mse_min = mse
            sf_min = sf
    return sf_min


def get_blue_m1_dict(results_dict, df_dict, mcoords):
    """Get pro multiplet 1 from each key (sample) in results_dict.

    PARAMS
    ------
    results_dict: direct output from do_1d_std_search().
    df_dict: dict of spectra dfs, where each key is the original, unshifted, full sample_df.
    mcoords: list of len 2; multiplet coordinates.

    RETURNS
    -------
    blue_m1_dict: dict of pro multiplet1s.
    """
    print("WARNING: shifting blues instead of reds in get_blue_m1_dict().")
    blue_m1_dict = {}
    for k in sorted(list(results_dict.keys())):
        normxcorr = results_dict[k]['multiplet_1']["max_rho"][0]
        ppm_shift = results_dict[k]["multiplet_1"]["ppm_shift"]
        #mcoords = results_dict[k]["multiplet_1"]["multiplet_match_ppm"][0]

        # get blue
        dt2 = df_dict[k].copy()
        dt2 = dt2.loc[(dt2["ppm"]>min(mcoords)) & (dt2["ppm"]<max(mcoords))].copy()
        temp_ls = dt2["ppm"].values-ppm_shift
        dt2["ppm"] = temp_ls

        # save mult1 dfs for plotting later. Not used in this function. 
        blue_m1_dict[k] = dt2

    return blue_m1_dict


def get_df_conc_lrmatching(results_dict, 
                           template_df, 
                           df_dict, 
                           mcoords, 
                           matching_coords_ls,
                           corr_series_dict,
                           min_normxcorr=0.75
                           ):
    """
    PARAMS
    ------
    results_dict: direct output from do_1d_std_search().
    template_df: original, unshifted, full template_df. 
    df_dict: dict of spectra dfs, where each key is the original, unshifted, full sample_df.
    mcoords: list of len 2; multiplet coords.
    matching_coords_ls: list of sublists, each of len 2.
    corr_series_dict: dict of corr dfs, direct. Every key in results_dict must be present in corr_series_dict.
    min_normxcorr: minimum normxcorr required to do LR matching. Set to 0.0 to always attempt LR matching; will sometimes error out due to differing blue and red lengths. 
    
    RESULT
    ------
    df_conc
    """
    # get red
    red_dt = template_df.loc[(template_df["ppm"]>min(mcoords)) & (template_df["ppm"]<max(mcoords))].copy()
    
    # run
    c = []
    for k in sorted(list(results_dict.keys())):
        normxcorr = results_dict[k]['multiplet_1']["max_rho"][0]
        ppm_shift = results_dict[k]["multiplet_1"]["ppm_shift"]

        if normxcorr >= min_normxcorr:
            # get blue
            dt2 = df_dict[k].copy()
            ppm_shift = results_dict[k]["multiplet_1"]["ppm_shift"]
            dt2 = dt2.loc[(dt2["ppm"]>min(mcoords)) & (dt2["ppm"]<max(mcoords))].copy()
            temp_ls = dt2["ppm"].values-ppm_shift
            dt2["ppm"] = temp_ls

            # get red and blue data points for LR
            red_matching_pts_ls = []
            blue_matching_pts_ls = []
            corr_weights_ls = []
            d_corr = corr_series_dict[k].copy()
            for coords in matching_coords_ls:
                vals = red_dt.loc[(red_dt["ppm"]>min(coords)) & (red_dt["ppm"]<max(coords))]["intensity"].values
                red_matching_pts_ls.append(vals)
                vals = dt2.loc[(dt2["ppm"]>min(coords)) & (dt2["ppm"]<max(coords))]["intensity"].values
                blue_matching_pts_ls.append(vals)
                vals = d_corr.loc[(d_corr["ppm"]>min(coords)) & (d_corr["ppm"]<max(coords))]["corr_series"].values
                corr_weights_ls.append(vals)

            # flatten red and blue_matching_pts_ls
            red_matching_pts_ls = [item for row in red_matching_pts_ls for item in row]
            blue_matching_pts_ls = [item for row in blue_matching_pts_ls for item in row]
            corr_weights_ls = [item for row in corr_weights_ls for item in row]

            # num of matching pts for each of blue and red must be the same
            if len(red_matching_pts_ls) != len(blue_matching_pts_ls):
                print(f"WARNING: len(red) != len(blue) for {k} - {len(red_matching_pts_ls)}, {len(blue_matching_pts_ls)}" )
                print("returning -1 instead")
                c.append([k, -1, -1, -1])

            else:
                # regress: blue = grad*red + intercept
                # coefficients = np.polyfit(red_matching_pts_ls, 
                #                         blue_matching_pts_ls, 
                #                         deg=1)
                
                # # Extracting the slope and intercept from the coefficients
                # slope = coefficients[0]
                # intercept = coefficients[1]
                
                model = LinearRegression()
                model.fit(np.array(red_matching_pts_ls).reshape(-1, 1), 
                          blue_matching_pts_ls, 
                          sample_weight=corr_weights_ls)
                
                intercept = model.intercept_
                slope = model.coef_[0]

                # get AUCs
                c.append([k, np.trapz((red_dt.intensity.values*slope)+intercept), slope, intercept])
        else:
            print(f"normxcorr for {k} too low ({normxcorr}), returning -1 instead" )
            c.append([k, -1, -1, -1])

    df_conc = pd.DataFrame(data=c, columns=["sample_name", "auc", "slope", "intercept"])
    
    return df_conc


def get_correlation_series(red_df, blue_df, 
                           min_corr=0.2, 
                           min_corr_replacement_value=0, 
                           window_size_nrows=64,
                           exponent=8
                           ):
    """
    Get a sliding window of normalized cross correlation between red and blue. 
    Red and blue arrays do not need to be of the same length, but they'll be set to the same length with align_spectra_dfs().
    
    PARAMS
    ------
    red_df: red df, normally the std
    blue_df: blue df, normally the sample
    min_corr: float; set all correlation values below `min_corr` to `min_corr_replacement_value`. Set both of these values to -1 to do nothing. 
    min_corr_replacement_value: float; set all correlation values below `min_corr` to `min_corr_replacement_value`. 
    window_size_nrows: int; window_size in terms of num_rows, not ppm.
    exponent: int; use power of correlation series to exaggerate correlation weighting effects. Set to 1 to do nothing.
    
    RETURNS
    -------
    d_corr: dataframe with columsn `ppm` and `corr_series`.
    """
    # align spectra
    dz = align_spectra_dfs(red_df, blue_df)
    #print(f"len(dz) = len")
    
    # calculate sliding window of xcorr
    correlation_series = []
    for i in range(len(dz) - window_size_nrows + 1):
        window1 = dz["intensity1"].values[i:i + window_size_nrows]
        window2 = dz["intensity2"].values[i:i + window_size_nrows]

        normxcorr = norm_xcorr(window1, window2)
        correlation_series.append(normxcorr)  # Store the correlation value

    # pad zeros to the left and right
    # I have a feeling that there'll be an off-by-one error somewhere in here
    left_zero_padding = [0]*math.floor(window_size_nrows/2)
    right_zero_padding = [0]*(math.floor(window_size_nrows/2)-1)

    correlation_series = left_zero_padding + correlation_series + right_zero_padding
    correlation_series = np.array(correlation_series)
    correlation_series[correlation_series < min_corr] = min_corr_replacement_value
    
    # create df to return
    d_corr = pd.DataFrame({"corr_series":correlation_series, 
                           "ppm":dz["ppm1"].values
                          })
    
    # take power of corr_series to exaggerate correlation weighting effects
    temp_ls = np.power(d_corr["corr_series"].values, exponent)
    d_corr["corr_series"] = temp_ls

    return d_corr
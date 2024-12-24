import numpy as np
import pandas as pd
import math
import re
from sklearn.linear_model import LinearRegression
from scipy.signal import find_peaks

import sys
sys.path.append("/Users/dteng/Documents/bin/nmr_utils/")
from nmr_targeted_utils import *

"""
Library of functions related to fitting a template to a sample.
"""

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


def get_values_from_html_report(report_fn):
    """Regex LR results from a HTML report from the LR fitting workflow.
    
    PARAMS
    ------
    report_fn: path to HTML report from LR fitting workflow.
    
    RETURNS
    -------
    lr_dict: dict with gradient, intercept and rsquared_adj values.
    """
    with open(report_fn, 'r') as file:
        c = file.read()

    lr_result_dict = {}
    # get LR result values
    pattern = re.compile(r'<li>mult1 gradient: ([\s\S]*?)</li>')
    lr_result_dict["gradient"] = float(re.findall(pattern, c)[0])
    pattern = re.compile(r'<li>mult1 intercept: ([\s\S]*?)</li>')
    lr_result_dict["intercept"] = float(re.findall(pattern, c)[0])
    pattern = re.compile(r'<li>mult1 rsquared_adj: ([\s\S]*?)</li>')
    lr_result_dict["rsquared_adj"] = float(re.findall(pattern, c)[0])
    
    return lr_result_dict


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


def get_df_conc_bottomup(template_df, df_dict, results_dict):
    """Wrapper for fit_std_to_sample() and get_true_sf(). Operates on an entire results_dict from do_1d_std_search()
    Hardcoded to operate on 'multiplet_1'. 
    
    PARAMS
    ------
    results_dict: direct output from do_1d_std_search().
    template_df: original, unshifted, full template_df. 
    df_dict: dict of spectra dfs, where each key is the original, unshifted, full sample_df.
    
    RETURNS
    -------
    df_conc2: df of resulting values, most importantly AUC. 
    """
    c = []
    for k in sorted(list(results_dict.keys())):
        mcoords = results_dict[k]["multiplet_1"]["coords"]
        ppm_shift = results_dict[k]["multiplet_1"]["ppm_shift"]
        max_rho = results_dict[k]["multiplet_1"]["max_rho"]

        bottomup_sf, auc = fit_std_to_sample(
            template_df=template_df.copy(),
            sample_df=df_dict[k],
            mcoords=mcoords,
            ppm_shift=ppm_shift,
            convergence_method='bottomup',
        )

        sf_true = get_true_sf(
            template_df.copy(), 
            mcoords, 
            ppm_shift, 
            bottomup_sf
        )

        c.append([k, bottomup_sf, sf_true, auc, max_rho])

    df_conc2 = pd.DataFrame(data=c, columns=["sample_name", "sf", "sf_true", "auc", "max_rho"])
    
    return df_conc2


def get_df_conc_bounded_lmse(template_df, df_dict, results_dict, dz_conc, sf_interval_width):
    """Get the sf with least MSE between sf_bounds. HARDCODED: Operates on multiplet_1 only. 
    
    PARAMS
    ------
    template_df: 
    df_dict: 
    results_dict: 
    dz_conc: merged df of sfs from bottom-up and naive-ht matching. Will contribute sf_lower and sf_upper.
    sf_interval_width: float;
    
    RETURNS
    -------
    df_conc_bounded_lmse: df of sf with lowest MSE. 
    """
    c = []
    for k in sorted(list(results_dict.keys())):
        ppm_shift = results_dict[k]["multiplet_1"]["ppm_shift"]
        mcoords = results_dict[k]["multiplet_1"]["coords"]

        # init reds
        red_df = template_df.copy()
        red_df_lower = red_df.loc[(red_df["ppm"]>min(mcoords)) & (red_df["ppm"]<max(mcoords))].copy()
        red_df_upper = red_df.loc[(red_df["ppm"]>min(mcoords)) & (red_df["ppm"]<max(mcoords))].copy()

        # get blue
        blue_df = df_dict[k].copy()
        blue_df = blue_df.loc[(blue_df["ppm"]>min(mcoords)) & (blue_df["ppm"]<max(mcoords))].copy()

        # get sf_lower and sf_upper from dz_conc
        dt = dz_conc.loc[dz_conc["sample_name"]==k].copy()
        sf_lower = min(dt["sf_bottomup"].values[0], dt["sf_ht_matching"].values[0])
        sf_upper = max(dt["sf_bottomup"].values[0], dt["sf_ht_matching"].values[0])

        # adjust red_lower and red_upper spectra
        temp_ls = red_df_lower["intensity"].values/sf_lower
        red_df_lower["intensity"] = temp_ls
        temp_ls = red_df_upper["intensity"].values/sf_upper
        red_df_upper["intensity"] = temp_ls
        temp_ls = red_df_lower["ppm"].values+ppm_shift
        red_df_lower["ppm"] = temp_ls
        temp_ls = red_df_upper["ppm"].values+ppm_shift
        red_df_upper["ppm"] = temp_ls

        # align
        dz = align_spectra_dfs(red_df_lower, blue_df)
        red_array = dz["intensity1"].values
        blue_array = dz["intensity2"].values

        sf_ls = get_sf_range(start=sf_lower, 
                             end=sf_upper, 
                             iter_size=sf_interval_width
                            )
        sf_min = get_sf_min(red_array, blue_array, sf_ls)

        dt = template_df.copy()
        dt = dt.loc[(dt["ppm"]>min(mcoords)) & (dt["ppm"]<max(mcoords))].copy()
        auc = np.sum(dt["intensity"].values)/sf_min
        c.append([k, sf_min, auc])

    df_conc_bounded_lmse = pd.DataFrame(data=c, columns=["sample_name", "sf", "auc"])
    
    return df_conc_bounded_lmse


def get_blue_m1_dict_new(results_dict, df_dict, mcoords=[]):
    """Get pro multiplet 1 from each key (sample) in results_dict.

    PARAMS
    ------
    results_dict: direct output from do_1d_std_search().
    df_dict: dict of spectra dfs, where each key is the original, unshifted, full sample_df.
    mcoords: list of len 2; multiplet coordinates. UNUSED.

    RETURNS
    -------
    blue_m1_dict: dict of pro multiplet1s.
    """
    print("WARNING: shifting blues instead of reds in get_blue_m1_dict().")
    blue_m1_dict = {}
    for k in sorted(list(results_dict.keys())):
        normxcorr = results_dict[k]['multiplet_1']["max_rho"][0]
        #ppm_shift = results_dict[k]["multiplet_1"]["ppm_shift"]
        blue_matching_coords = results_dict[k]["multiplet_1"]["multiplet_match_ppm"][0] # ppm location of best match on blue
        #mcoords = results_dict[k]["multiplet_1"]["multiplet_match_ppm"][0]

        # get blue
        dt2 = df_dict[k].copy()
        dt2 = dt2.loc[(dt2["ppm"]>min(blue_matching_coords)) & (dt2["ppm"]<max(blue_matching_coords))].copy()

        blue_m1_dict[k] = dt2

    return blue_m1_dict


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
                c.append([k, np.sum((red_dt.intensity.values*slope)+intercept), slope, intercept])
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
    print(f"len(dz) = len")
    
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


def get_peaks_and_troughs(df, min_diff=0.002):
    """Get peaks and troughs of an input spectra df, using scipy's find_peaks(). Works *well enough*. 

    PARAMS
    ------
    df: spectra df. For best (i.e. most checkable) results, use df of a multiplet, not a whole spectrum.
    min_diff: float; min distance in ppm between detected peak/trough to be regarded as the same point. 

    RETURNS
    -------
    d_pks: df of features, i.e. peaks and troughs. Has columns ppm and intensity.
    """
    # find peaks scratch
    pks_data = find_peaks(df["intensity"].values, 
                          height=1E7, 
                          distance=20)

    df_inverted = (df.intensity.values*-1) + max(df.intensity.values)
    troughs_data = find_peaks(df_inverted, 
                              height=0.5, 
                              distance=20)

    df_pks = df.copy()
    df_pks = df_pks.iloc[pks_data[0]]
    df_troughs = df.copy()
    df_troughs = df_troughs.iloc[troughs_data[0]]

    d_pks = pd.concat([df_pks, df_troughs], ignore_index=True)
    d_pks = d_pks.sort_values(by="ppm").reset_index(drop=True)

    # filter out overlapping peaks/troughs
    m = d_pks.values
    m1 = [m[0]]
    for i in range(1, len(m)):
        prev_val = m[i-1][0]
        current_val = m[i][0]
        if current_val - prev_val > min_diff:
            m1.append(m[i])

    d_pks2 = pd.DataFrame(data=m1, columns=d_pks.columns)

    return d_pks2


def get_matching_features(red_features_df, blue_features_df, min_match_dist=0.00075):
    """Get matching canonical features (peaks or troughs) between red_features_df and blue_features_df. Order matters, because this function searches for features in red that are in blue, but not the other way round. 
    
    PARAMS
    ------
    red_features_df: df of features from std, in ppm and intensity
    blue_features_df: df of features from sample, in ppm and intensity
    min_match_dist: if features (rows) are within `min_match_dist` of each other, they're considered to be the same.
    
    RETURNS
    -------
    d_matching_features: df of matching features, in terms of index and ppm. Index columns are in terms of the input _feature_dfs.
    """
    # get matching features (peaks or troughs)
    idx_red = 0
    c = []
    for red_ppm_val in red_features_df['ppm'].values:
        time_diffs = np.abs(blue_features_df["ppm"].values - red_ppm_val)
        idx_of_min_diff = np.argmin(time_diffs)
        min_diff = time_diffs[idx_of_min_diff]
        if min_diff < min_match_dist:
            c.append([idx_red, 
            idx_of_min_diff, 
            red_ppm_val, 
            blue_features_df["ppm"].values[idx_of_min_diff],
            ]
            )

        idx_red += 1

    d_matching_features = pd.DataFrame(data=c, 
                                       columns=["idx_red", 
                                                "idx_blue", 
                                                "ppm_red", 
                                                "ppm_blue"]
                                      )
    
    # add intensity columns
    temp_ls = red_features_df.iloc[d_matching_features["idx_red"].values]["intensity"].values
    d_matching_features["intensity_red"] = temp_ls
    temp_ls = blue_features_df.iloc[d_matching_features["idx_blue"].values]["intensity"].values
    d_matching_features["intensity_blue"] = temp_ls

    return d_matching_features
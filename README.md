# NMR Pro Quant 2024 

### In This Repo

* `mlgrad-weighted-lr-5apr2024` - dir containing workbooks that do automated fitting for training and validation sets
* `manual_integration_workflow_20240112` - a minority of samples will not be fitted well by automated fitting; use the manual integration workbook to manually fit these.

# Method

### Fitting

Fitting the standard (colloquially known as "red") to a sample ("blue") is done at many stages: getting and using a gradient. Fitting is a 3-step process:
1. Find the position of best fit of red (pro std) on blue (a sample). The entire fitting process is designed to fail if the score of the best fit (measured by *normalized cross-correlation score*, or normxcorr) is still too low (less than 0.8). 
2. Get Pearson correlation of data points of each of red and blue spectra that fall within certain manually pre-selected regions. These are usually the peaks and the troughs of the proline multiplet. This is to mitigate the presence of contaminants that persistently appear at known regions. 
3. Use linear regression to compute a multiplier that scales red, and additive constant that shifts red up or down, weighted by correlation from (2). The role of weighting linear regression by correlation is as follows:
  * If, in that manually selected region, red and blue look extremely similar (i.e. highly correlated), that region will have more weight in the LR, i.e. more influence on the final fitted gradient and intercept.
  * Conversely, if red and blue look different in that region (less correlated), that region will have less weight in the LR, i.e. less influence on the final fitted gradient and intercept.

Figure below illustrates the difference in results of OLS between not using these manually selected regions ("before"), and with ("after"). In the top row, linear regression is run (step 3) using the entire spectra. In the bottom row, linear regression is run using only the regions of the spectra which fall within the manually selected regions (gray bars). The dark line in the second row shows a sliding window of correlation of red against blue, taken across the entire spectra; this correlation series applies a weight to the linear regression. 

![Alt text](https://github.com/AWRIMetabolomics/pro-nmr-quant-2024/blob/master/figs/before_n_after.png)

### Calculating Gradient (Training Step)

* Training data: 33 AF IDs with to bootstrap a gradient with. Final output is a distro of gradients. 
* We don't use a traditional cal curve, instead we use the difference in area between a spiked sample and the unspiked version of the same sample. Spiked amount is 1115 mg/L. There are 33 such samples. 
* Use bootstrapping on AWS to generate a distribution of coefficient x, where conc = x * auc, and finally compute `np.average(x)` (bootstrapping = resampling a bootstrap sample, n = 33, with replacement. Do this 5000x to get a distribution of x) This need only be done once, or very infrequently. 
* Error: grad * (AUC(spiked) - AUC(original)) - 1115 should be 0

#### Testing Step

For each test sample:

1. Shape matching: normxcorr-based template-matching, 
2. Based on manually-pre-specified regions of high or low reliability, do linear regression to calculate scaling factor and additive constant to scale template (red) to signal (blue). Calculate the area of red. 
3. Predict the conc of red based on AUC of red from the previous step. 

# Notes

* Notably missing: various imported .py files, in particular `nmr_fitting_funcs.py` and `nmr_targeted_utils.py`. Get these from the `/bin-of-tools` repo for DRY.
* The `/manual_integration...` dir is a fairly self-contained folder containing everything you need for manual integration for when auto-integration fails, but you'll have to regenerate `blue_m1_dict.pkl` everytime as input data (that is, dict of M1 multiplets where the ref peak at 0 has been adjusted to be at 0ppm). The manual integration workbook also needs the latest working versions of `nmr_fitting_funcs.py` and `nmr_targeted_utils.py`. 
* bootstrap_results.csv as at 12Sep2023: `s3://awri-ma-archives/bootstrap_results_12sep2023.csv`
* an archival folder of old code, `/archive_old_workflows_pre2024`, is not backed up on GH. 
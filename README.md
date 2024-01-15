* Notably missing: various imported .py files, in particular `nmr_fitting_funcs.py` and `nmr_targeted_utils.py`. Get these from the `/bin-of-tools` repo for DRY.
* The `/manual_integration...` dir is a fairly self-contained folder containing everything you need for manual integration for when auto-integration fails, but you'll have to regenerate `blue_m1_dict.pkl` everytime as input data (that is, dict of M1 multiplets where the ref peak at 0 has been adjusted to be at 0ppm). The manual integration workbook also needs the latest working versions of `nmr_fitting_funcs.py` and `nmr_targeted_utils.py`. 
* bootstrap_results.csv as at 12Sep2023: `s3://awri-ma-archives/bootstrap_results_12sep2023.csv`
* an archival folder of old code, `/archive_old_workflows_pre2024`, is not backed up on GH. 

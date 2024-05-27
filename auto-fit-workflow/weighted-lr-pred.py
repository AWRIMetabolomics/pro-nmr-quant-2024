#!/usr/bin/python3
# a CLT version of weighted-lr-prod-batched.ipynb

# usage:
# open the file and manually change inputs as required, in the INPUTS block below
# within the local dir, run: 
# $ python3 find_nmr_experiment_num.py

import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.models import ColumnDataSource, HoverTool
import matplotlib.patches as patches

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
import os
import json
import time
from io import StringIO
from datetime import datetime
import pickle
import re
import math

from sklearn.linear_model import LinearRegression
from scipy import integrate

import sys
sys.path.append("/Users/dteng/Documents/bin/nmr_utils/")
from nmr_targeted_utils import *
from nmr_fitting_funcs import *

# ========== INPUTS ==========

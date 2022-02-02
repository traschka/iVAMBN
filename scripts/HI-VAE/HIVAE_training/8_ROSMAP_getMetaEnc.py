#!/usr/bin/env python
# coding: utf-8

# # Functions and imports

# Imports and the functions that call the HI-VAE, modified from the paper version only to allow inputting s_codes and z_codes manually.

# In[2]:


import time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
import os
import re
import pandas as pd
import numpy as np
from IPython.display import Audio
import seaborn as sns

import helpers

sample_size=221
# get file list
files=[i for i in os.listdir('../GridSearch/data_python/') if not '_type' in i and not '_missing' in i and not 'stalone' in i and i not in '.DS_Store']
files.sort()
print(files)

enc_vars=[pd.read_csv('Saved_Networks/'+re.sub('.csv','',f)+'_meta.csv') for f in files]
meta=helpers.merge_dat(enc_vars)
meta[meta.columns[['Unnamed' not in i for i in meta.columns]]].to_csv('metaenc.csv',index= False)

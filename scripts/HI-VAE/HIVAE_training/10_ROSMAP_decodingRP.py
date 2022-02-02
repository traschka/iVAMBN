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

def set_settings(opts,nepochs=500,modload=False,save=True): # note: modload doesnt do anything right now, hardcoded in helpers.py
    'replace setting template placeholders with file info'
    inputf=re.sub('.csv','',opts['files'].iloc[0])
    #missf=inputf+'_missing.csv'
    typef=inputf+'_types.csv'

    template = '--epochs NEPOCHS --model_name model_HIVAE_inputDropout --restore MODLOAD         --data_file ../GridSearch/data_python/INPUT_FILE.csv --types_file ../GridSearch/data_python/TYPES_FILE          --batch_size NBATCH --save NEPFILL --save_file SAVE_FILE        --dim_latent_s SDIM --dim_latent_z 1 --dim_latent_y YDIM         --learning_rate LRATE'

    # replace placeholders in template
    settings = re.sub('NEPOCHS',str(nepochs),template)
    settings = re.sub('MODLOAD','1',settings) if modload else re.sub('MODLOAD','0',settings)
    settings = re.sub('INPUT_FILE',inputf,settings)
    settings = re.sub('TYPES_FILE',typef,settings)
    settings = re.sub('NBATCH',str(opts['nbatch'].iloc[0]),settings)
    settings = re.sub('NEPFILL',str(nepochs-1),settings) if save else re.sub('NEPFILL',str(nepochs*2),settings)
    settings = re.sub('SAVE_FILE',inputf,settings)
    settings = re.sub('SDIM',str(opts['sdims'].iloc[0]),settings)
    settings = re.sub('YDIM',str(opts['ydims'].iloc[0]),settings)
    settings = re.sub('LRATE',str(opts['lrates'].iloc[0]),settings)
    #settings = re.sub('MISS_FILE',missf,settings) #if not 'medhist' in inputf else re.sub('--true_miss_file data_python/MISS_FILE','',settings)

    return settings

files=[i for i in os.listdir('../GridSearch/data_python/') if not '_type' in i and not 'stalone' in i and i not in '.DS_Store']
files.sort()

# # RP decoding (Reconstruction)

# In[14]:
best_hyper=pd.read_csv('../GridSearch/results_ROSMAP.csv',  sep = ',')
best_hyper.files = best_hyper.files.str.replace('\\_grid.*?\\_results', '')
best_hyper.files = best_hyper.files.str.replace('gridsearchresult\\\\', '')
best_hyper = best_hyper.sort_values('files')
best_hyper = best_hyper.reset_index(drop=True)
sds = best_hyper['ydims']
sdims=dict(zip(files,sds))
if any(files!=best_hyper['files']):
    print('ERROR')
else:
    best_hyper['sdims']=sds

best_hyper['nbatch'] = best_hyper['nbatch'].astype(int)
best_hyper['ydims'] = best_hyper['ydims'].astype(int)
best_hyper['sdims'] = best_hyper['sdims'].astype(int)

best_hyper = best_hyper[['lrates', 'nbatch', 'ydims', 'files', 'loss', 'sdims']]



meta = pd.read_csv('metaenc.csv')
sample_size = 221
recon=list()
recdfs=list()
for f in files:
    # replace placeholders in template
    opts=dict(best_hyper[best_hyper['files'].copy()==f])
    opts['nbatch'].iloc[0]=sample_size
    settings=set_settings(opts,nepochs=1,modload=True,save=False)

    #run
    zcodes=meta['zcode_'+re.sub('.csv','',f)]
    scodes=meta['scode_'+re.sub('.csv','',f)]
    rec=helpers.dec_network(settings,zcodes,scodes)
    recon.append(rec)

    subj=pd.read_csv('../GridSearch/python_names/'+re.sub('.csv','',f)+'_subj.csv')['x']
    names=pd.read_csv('../GridSearch/python_names/'+re.sub('.csv','',f)+'_cols.csv')['x']
    recd=pd.DataFrame(rec)
    recd.columns=names
    recd['SUBJID']=subj
    recdfs.append(recd)

recon_dic=dict(zip(files,recon))

data_recon=helpers.merge_dat(recdfs)
data_recon.Cognition_braaksc = data_recon.Cognition_braaksc + 1
data_recon.to_csv('reconRP.csv',index=False)

### get original values in one table
def merge_dat_orig(lis):
    'merge all dataframes in a list on SUBJID'
    df = lis[0]
    for x in lis[1:]:
        df=pd.concat([df, x], axis=1)
    return df

orig = pd.read_csv('../GridSearch/data_python/Cluster_1.csv', header=None)
names = pd.read_csv('../GridSearch/python_names/Cluster_1_cols.csv')
orig.columns = names['x'].values.tolist()

for f in files[1:]:
    orig_vars = pd.read_csv('../GridSearch/data_python/'+re.sub('.csv','',f)+'.csv', header=None)
    names = pd.read_csv('../GridSearch/python_names/'+re.sub('.csv','',f)+'_cols.csv')
    orig_vars.columns = names['x'].values.tolist()
    orig = pd.concat([orig, orig_vars], axis=1)

#orig=merge_dat_orig(orig_vars)
#orig.to_csv('originalVals.csv',index= False)

#orig.columns = decoded.columns
orig.head()

orig.to_csv('originalVals.csv',index= False)

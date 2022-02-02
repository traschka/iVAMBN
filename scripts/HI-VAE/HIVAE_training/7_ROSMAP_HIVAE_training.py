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
import os,operator, sys
import re
import pandas as pd
from IPython.display import Audio
import seaborn as sns
import multiprocessing as mp
from multiprocessing import set_start_method
mp.set_start_method("spawn", force=True)


import helpers # this is where the main training/decoding functions are, modified from teh original HIVAE main.py

file_index = int(sys.argv[1])

#import warnings
#warnings.filterwarnings('ignore') ########## NOTE: comment out for testing in case it's hiding problems

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


# In[3]:


os.getcwd()


# In[4]:


sample_size=221
# get file list
files=[i for i in os.listdir('../GridSearch/data_python/') if not '_type' in i and not 'stalone' in i and i not in '.DS_Store']
files.sort()
print(files)

# In[5]:
best_hyper=pd.read_csv('../GridSearch/results_ROSMAP.csv',  sep = ',')

best_hyper.files = best_hyper.files.str.replace('\\_grid.*?\\_results', '')
best_hyper.files = best_hyper.files.str.replace('gridsearchresult\\\\', '')

best_hyper = best_hyper.sort_values('files')
best_hyper = best_hyper.reset_index(drop=True)

#sds=[1,2,1,2,2,2,2,1,1,2,1,2,1,1,2,2,2,1,2,1,2,1,1,2,2,1,1,1,1]
sds = best_hyper['ydims']
#sds = best_hyper['ydims']

sdims=dict(zip(files,sds))
if any(files!=best_hyper['files']):
    print('ERROR')
else:
    best_hyper['sdims']=sds

best_hyper['nbatch'] = best_hyper['nbatch'].astype(int)
best_hyper['ydims'] = best_hyper['ydims'].astype(int)
best_hyper['sdims'] = best_hyper['sdims'].astype(int)

best_hyper = best_hyper[['lrates', 'nbatch', 'ydims', 'files', 'loss', 'sdims']]
best_hyper

# # General settings

# sds is info about which files have what dimension of the "s_codes", that determine the number of mixture components in the "zcodes", our continuous embeddings used in the Bayes Net

# # Training

# In[13]:

def wrapper_train_network(arguments):
    settings = arguments
    return helpers.train_network(settings)

def wrapper_enc_network(arguments):
    settings = arguments
    return helpers.enc_network(settings)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def main():
    f = files[file_index]

    opts=dict(best_hyper[best_hyper['files'].copy()==f])
    settings=set_settings(opts,modload=False,save=True)
    #helpers.train_network(settings)
    wrapper_train_network(settings)

    opts=dict(best_hyper[best_hyper['files'].copy()==f])
    opts['nbatch'].iloc[0]=sample_size
    settings=set_settings(opts,nepochs=1,modload=True,save=False)

    encs,encz,d = wrapper_enc_network(settings)
    # make deterministic embeddings
    subj=pd.read_csv('../GridSearch/python_names/'+re.sub('.csv','',f)+'_subj.csv')['x']
    sc=pd.DataFrame({'scode_'+re.sub('.csv','',f):pd.Series(np.array([i for i in encs])),'SUBJID':subj})
    zc=pd.DataFrame({'zcode_'+re.sub('.csv','',f):pd.Series(np.array([i[0] for i in encz])),'SUBJID':subj})
    enc=pd.merge(sc, zc, on = 'SUBJID')
    enc.to_csv('Saved_Networks/'+re.sub('.csv','',f)+'_meta.csv',index = False)


if __name__ == "__main__":
    main()

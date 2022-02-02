#!/usr/bin/env python
# coding: utf-8

# # Functions and imports

# In[1]:


# set up GPU usage
import time
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.ioff()
from pandas.plotting import scatter_matrix
import os,operator, sys
import multiprocessing as mp
from multiprocessing import set_start_method
mp.set_start_method("spawn", force=True)
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from IPython.display import Audio
import seaborn as sns
import itertools
from operator import itemgetter

import helpers # this is where the main training/decoding functions are, modified from teh original HIVAE main.py

# initialize helper variables to refer to the correct indices during parallelization
file_index = int(sys.argv[1])

def set_settings(f,opts):
    'replace setting template placeholders with file info'
    inputf=re.sub('.csv','',f)
    #missf=inputf+'_missing.csv'
    typef=inputf+'_types.csv'

    template = '--epochs 500 --model_name model_HIVAE_inputDropout --restore 0 --train 1         --data_file data_python/INPUT_FILE.csv --types_file data_python/TYPES_FILE          --batch_size NBATCH --save 499 --save_file INPUT_FILE        --dim_latent_s 1 --dim_latent_z 1 --dim_latent_y YDIM         --learning_rate LRATE --weight_decay WDECAY'

    # replace placeholders in template
    settings = re.sub('INPUT_FILE',inputf,template)
    settings = re.sub('WDECAY',str(opts['wdecay']),settings)
    settings = re.sub('NBATCH',str(opts['nbatch']),settings)
    settings = re.sub('YDIM',str(opts['ydims']),settings)
    #settings = re.sub('MISS_FILE',missf,settings) #if not 'medhist' in inputf else re.sub('--true_miss_file data_python/MISS_FILE','',settings)
    settings = re.sub('TYPES_FILE',typef,settings)
    settings = re.sub('LRATE',str(opts['lrates']),settings)

    return settings


# In[2]:

print("If the GPU is used or not:")
print(tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
))


files=[i for i in os.listdir('data_python/') if not '_type' in i and '.DS_Store' not in i and not '_missing' in i and not 'stalone' in i]


# Run training, then check "Saved Networks/train_stats/" for images of the reconstruction loss over the epochs. If training didnt converge for some files, rerun individual files below.

# In[ ]:

search_options = {
    'ydims': [1],
    'lrates': [0.001,0.01],
    'wdecay': [0,0.001,0.01],
    'nbatch': [16,32],
    'act': ['none','relu','tanh']
}
# cross val rec loss
keys, values = zip(*search_options.items())
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

n_splits=3

#l_files=[]
#for f in files:
#    l_exp=[]
#    for opt in experiments:
#        settings=set_settings(f,opt)
#        l_exp.append(helpers.run_network(settings,
#                                         'YD'+str(opt['ydims'])+'_LR'+str(opt['lrates'])+'_WD'+str(opt['wdecay'])+'_NB'+str(opt['nbatch'])+'_ACT'+str(opt['act']),
#                                         n_splits=n_splits))
#    l_files.append(l_exp)

#helper wrapper function
def wrapper_run_network(arguments):
    (opt, name, n_splits) = arguments
    return helpers.run_network(opt, name, n_splits=n_splits)

def main():
    # TODO: run different experiments across different CPUs
    f = files[file_index].split(r"/")[-1]
    # helper variable for number of experiments
    len_exps = len(experiments)

    # TODO: use python's multiprocessing library for that
    # TODO: what will run_network return? is there the experiment settings and file name in there?
    mp_pool = mp.pool.Pool(processes=len_exps)

    # set chunksize to 1 to preserve order
    # process each experiment for the specified file in a separate process (separate CPU)
    results = mp_pool.map(wrapper_run_network, [(set_settings(f,opt), 'YD'+str(opt['ydims'])+'_LR'+str(opt['lrates'])+'_WD'+str(opt['wdecay'])+'_NB'+str(opt['nbatch']), n_splits) for opt in experiments])
    # save all the processed experiments for a specific file
    result_dataframe = pd.DataFrame()
    for i in range(len_exps):
        result_dict = experiments[i]
        result_dict["loss"] = results[i]
        result_dataframe = result_dataframe.append(result_dict, ignore_index=True)

    result_dataframe.to_csv("gridsearchresult/" + re.sub(r'.csv', '', f) + "_grid_search_results.csv", index=True)

#losses=list(zip(files,l_files))
#selectexp=[np.argmin(losses[f][1]) for f in range(len(files))]
#minloss=[np.nanmin(losses[f][1]) for f in range(len(files))]
#output=itemgetter(*selectexp)(experiments)
#output=pd.DataFrame(list(output))
#output['files']=files
#output['loss']=minloss
#output.to_csv('results_Altoida.csv',index=True)


# In[ ]:

if __name__ == "__main__":
    main()

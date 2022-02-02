############# README
# This file runs the Bayesian network
############# 

############################
############################ Dependencies and helper functions
############################
rm(list=ls())
library(bnlearn)
library(bestNormalize)
# general helpers
source('load_kgAsBN.R')
# study specific helpers
source('merge_data.R')
source('make_bl_wl_ROSMAP.R')

############################
############################ Settings and preprocessing
############################

# Name output files
data_out<-'data/ROSMAP/'
scr<-"bic-cg" # BN score
mth<-"mle" # BN method

# Load data & remaining formatting of standalone
#data<-merge_data(complete=T) # merge imputed standalone and zcodes from HIVAE
load("../data/ROSMAP/data_final_ROSMAP.RData") #encoded data
load("../data/ROSMAP/data_condensed_complete.RData") #original data

orig_braak = data_all$phenotype$phenotype_braaksc
bxcx_braak = boxcox(orig_braak)$x.t

data$phenotype = bxcx_braak # set the transformed braak stages as phenotype, such that we do not have to rename the node in the net

# remove subject variable
pt<-data$SUBJID
data$SUBJID<-NULL

# Discretize & set score
discdata<-data

# save fitted network
finalBN<-readRDS(file='results/bn_ROSMAP/finalBN.rds') #load the network structure trained before
real = discdata
real$SUBJID<-NULL
set.seed(75)
fitted = bn.fit(finalBN, real, method=mth)
save(fitted,file="finalBN_onlyBraak_fitted.Rdata")

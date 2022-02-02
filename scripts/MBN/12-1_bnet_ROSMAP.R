############# README
# This file runs the Bayesian network
############# 

############################
############################ Dependencies and helper functions
############################
rm(list=ls())
library(tidyverse)
library(beepr)
library(arules)
library(mclust)
library(rpart)
library(bnlearn)
library(parallel)
# general helpers
source('helper_functions/load_kgAsBN.R')
# study specific helpers
source('ROSMAP_helper/merge_data.R')
source('ROSMAP_helper/make_bl_wl_ROSMAP.R')

############################
############################ Settings and preprocessing
############################

# Name output files
data_out<-'data/ROSMAP/'
scr<-"bic-cg" # BN score
mth<-"mle" # BN method

# Load data & remaining formatting of standalone
data<-merge_data(complete=T) # merge imputed standalone and zcodes from HIVAE



# remove subject variable
pt<-data$SUBJID
data$SUBJID<-NULL

# Discretize & set score
discdata<-data
#discdata<-addnoise(data,0.01) # add noise to the imputed/constant levels of continuous variables, prevents error in the BN due to singular data

############################
############################ Bnet
############################

# Make bl/wl/startPoints
## no and middle knowledge
blname<-'bl_noKL.csv'
make_bl_wl_ROSMAP_noKL(discdata,blname,wlname) # rm has info about "orphaned" nodes (need to be connected to visitmiss, not to AUX)
bl_no<-read.csv(blname)

##middle knowledge
### use kg as starting point
kg = load_kgAsBN()
maxpVal = max(table(kg$arcs[,"to"]))

## complete knowledge
blname<-'bl_completeKL.csv'
wlname<-'wl_completeKL.csv'
make_bl_wl_ROSMAP_completeKL(discdata,blname,wlname)
bl_complete<-read.csv(blname)
wl_complete<-read.csv(wlname)


##test the best  algorithm for each knowledge integration level
#### no Knowledge
set.seed(75)
cvres_no_hc = bn.cv(discdata, "hc", runs=10, 
                    algorithm.args = list(maxp=maxpVal, blacklist=bl_no, score=scr), 
                    fit.args=list(replace.unidentifiable=TRUE))
print(cvres_no_hc)
save(cvres_no_hc, file="../results/bn_ROSMAP/cvres_no_hc.RData")

set.seed(75)
cvres_no_tabu = bn.cv(discdata, "tabu", runs=10,
                      algorithm.args = list(maxp=maxpVal, blacklist=bl_no, score=scr),
                      fit.args=list(replace.unidentifiable=TRUE))
print(cvres_no_tabu)
save(cvres_no_tabu, file="../results/bn_ROSMAP/cvres_no_tabu.RData")

#### middle Knowledge
set.seed(75)
cvres_middle_hc = bn.cv(discdata, "hc", runs=10, 
                        algorithm.args = list(start=kg, maxp=maxpVal, blacklist=bl_no, score=scr),
                        fit.args=list(replace.unidentifiable=TRUE))
print(cvres_middle_hc)
save(cvres_middle_hc, file="../results/bn_ROSMAP/cvres_middle_hc.RData")

set.seed(75)
cvres_middle_tabu = bn.cv(discdata, "tabu", runs=10,
                          algorithm.args = list(start=kg, maxp=maxpVal, blacklist=bl_no, score=scr),
                          fit.args=list(replace.unidentifiable=TRUE))
print(cvres_middle_tabu)
save(cvres_middle_tabu, file="../results/bn_ROSMAP/cvres_middle_tabu.RData")

#### middle Knowledge2
set.seed(75)
cvres_middle2_hc = bn.cv(discdata, "hc", runs=10,
                         algorithm.args = list(maxp=maxpVal, blacklist=bl_no, whitelist=wl_complete, score=scr),
                         fit.args=list(replace.unidentifiable=TRUE))
print(cvres_middle2_hc)
save(cvres_middle2_hc, file="../results/bn_ROSMAP/cvres_middle2_hc.RData")

set.seed(75)
cvres_middle2_tabu = bn.cv(discdata, "tabu", runs=10,
                           algorithm.args = list(maxp=maxpVal, blacklist=bl_no, whitelist=wl_complete, score=scr),
                           fit.args=list(replace.unidentifiable=TRUE))
print(cvres_middle2_tabu)
save(cvres_middle2_tabu, file="../results/bn_ROSMAP/cvres_middle2_tabu.RData")

#### middle Knowledge3
set.seed(75)
cvres_middle3_hc = bn.cv(discdata, "hc", runs=10,
                         algorithm.args = list(start=kg, maxp=maxpVal, blacklist=bl_no, whitelist=wl_complete, score=scr),
                         fit.args=list(replace.unidentifiable=TRUE))
print(cvres_middle3_hc)
save(cvres_middle3_hc, file="../results/bn_ROSMAP/cvres_middle3_hc.RData")

set.seed(75)
cvres_middle3_tabu = bn.cv(discdata, "tabu", runs=10, 
                           algorithm.args = list(start=kg, maxp=maxpVal, blacklist=bl_no, whitelist=wl_complete, score=scr),
                           fit.args=list(replace.unidentifiable=TRUE))
print(cvres_middle3_tabu)
save(cvres_middle3_tabu, file="../results/bn_ROSMAP/cvres_middle3_tabu.RData")


#### complete Knowledge
set.seed(75)
cvres_complete_hc = bn.cv(discdata, "hc", runs=10,
                          algorithm.args = list(maxp=maxpVal, blacklist=bl_complete, whitelist=wl_complete, score=scr),
                          fit.args=list(replace.unidentifiable=TRUE))
print(cvres_complete_hc)
save(cvres_complete_hc, file="../results/bn_ROSMAP/cvres_complete_hc.RData")

set.seed(75)
cvres_complete_tabu = bn.cv(discdata, "tabu", runs=10,
                            algorithm.args = list(maxp=maxpVal, blacklist=bl_complete, whitelist=wl_complete, score=scr),
                            fit.args=list(replace.unidentifiable=TRUE))
print(cvres_complete_tabu)
save(cvres_complete_tabu, file="../results/bn_ROSMAP/cvres_complete_tabu.RData")


#### load bn cv results and plot
# all
plot(cvres_no_hc, cvres_no_tabu, cvres_middle_hc, cvres_middle_tabu, cvres_middle2_hc, cvres_middle2_tabu, cvres_middle3_hc, cvres_middle3_tabu, cvres_complete_hc, cvres_complete_tabu,
     xlab=c("no_hc", "no_tabu", "middle_hc", "middle_tabu", "middle2_hc", "middle2_tabu", "middle3_hc", "middle3_tabu", "complete_hc", "complete_tabu"))

# hc 
plot(cvres_no_hc, cvres_middle_hc, cvres_middle2_hc, cvres_middle3_hc, cvres_complete_hc,
     xlab=c("no", "middle1", "middle2", "middle3", "complete"))

# hc no and middle
plot(cvres_no_hc, cvres_middle_hc, cvres_middle2_hc, cvres_middle3_hc,
     xlab=c("no", "middle1", "middle2", "middle3"))



# Final bayesian network #hc_middle3
set.seed(1234)
finalBN = hc(discdata, maxp=maxpVal, blacklist=bl_no, score=scr, start=kg, whitelist=wl_complete, restart = 100)
save(finalBN,file='finalBN.Rdata')

#all(finalBN$arcs==kg$arcs)

# Bootstrapped network
set.seed(1234)
boot.stren = boot.strength(discdata, algorithm="hc", R=1000, 
                           algorithm.args = list(maxp=maxpVal, blacklist=bl_no, start=kg, whitelist=wl_complete, score=scr))
boot.strenwithThreshold = boot.stren[boot.stren$strength >= 0.5, ]
saveRDS(boot.stren,file='results/bn_ROSMAP/bootBN_no.rds')
save(boot.stren,file='results/bn_ROSMAP/bootBN_no.Rdata')

# save fitted network
real = discdata
real$SUBJID<-NULL
finalBN<-readRDS(file='results/bn_ROSMAP/finalBN.rds')
set.seed(75)
fitted = bn.fit(finalBN, real, method=mth)
save(fitted,file="finalBN_fitted.Rdata")


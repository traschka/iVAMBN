############# README
# This is to analyse the likelihood.
############# 

rm(list=ls())
library(igraph)
library(bnlearn) # hc might be overwritten by arules or some such package "bnlearn::hc" if so; not currently used though
library(bestNormalize)
library(ggplot2)
source('ROSMAP_helper/merge_data.R')

########## Name output files
data_out<-paste0('data/')
scr<-"bic-cg" # 'bic' for basic autoencoder and fully discretized
mth<-"mle" # 'bayes' for basic autoencoder and fully discretized

# load model and data
## model
load("../results/bn_ROSMAP/finalBN_onlyBraak_fitted.Rdata") #fitted, cognition = braaksc only

## ROSMAP
load("../data/ROSMAP/data_final_ROSMAP.RData") #encoded data
load("../data/ROSMAP/data_condensed_complete.RData") #original data

orig_braak = data_all$phenotype$Phenotype_braaksc
bxcx_transform = boxcox(orig_braak)
bxcx_braak = bxcx_transform$x.t

data$phenotype = bxcx_braak

rosmap<-data
rosmap$SUBJID<-NULL

rm("data", "data_all")

## Mayo
load("../data/Mayo/data_condensed_complete.RData") 
mayo_braak = data_all$phenotype$Phenotype_braaksc

data_out<-'data/Mayo/'
mayo<-merge_data(dataset="Mayo") # merge imputed standalone and zcodes from HIVAE
mayo$SUBJID<-NULL
mayo$phenotype = predict(bxcx_transform, newdata = mayo_braak)
mayo$PatDemo_educ = NA
mayo$PatDemo_educ = as.numeric(mayo$PatDemo_educ)

rm("data_all")

#### need to impute the data to be able to get a log-likelihood
### https://www.bnlearn.com/documentation/man/impute.html
### logisches Sampling
mayo_imputed = impute(fitted, mayo, method="bayes-lw")
saveRDS(mayo_imputed, file = "../data/Mayo/data_final_imputed_onlyBraak.rds")
save(mayo_imputed, file = "../data/Mayo/data_final_imputed_onlyBraak.RData")

########### Loglikelihood of ROSMAP, Mayo
rosmap_LL=logLik(fitted, rosmap, by.sample=TRUE)
mayo_LL=logLik(fitted, mayo_imputed, by.sample=TRUE)

data_lik<-data.frame(
  likelihood=c(rosmap_LL,mayo_LL),
  type=c(rep('ROSMAP',length(rosmap_LL)),rep('Mayo',length(mayo_LL)))
)
print(paste('Mean ROSMAP:',mean(rosmap_LL[is.finite(rosmap_LL)],rm.na=T),
            'Mean Mayo:',mean(mayo_LL,rm.na=T)))
pl<-ggplot(data_lik, aes(x=likelihood,fill=type))+geom_density(alpha=.2)+xlab('Likelihood')+ylab('density')
pl

write.table(data_lik, file="../results/logliks_onlyBraak/logLiksBN_allData_perSample_onlyBraak.csv", quote=F, row.names = F, sep=",")


rosmap_LL = logLik(fitted, rosmap)
mayo_LL = logLik(fitted, mayo_imputed)

data_lik<-data.frame(
  likelihood=c(rosmap_LL,mayo_LL),
  type=c(rep('ROSMAP',length(rosmap_LL)),rep('Mayo',length(mayo_LL)))
)

write.table(data_lik, file="../results/logliks_onlyBraak/logLiksBN_allData_onlyBraak.csv", quote=F, row.names = F, sep=",")

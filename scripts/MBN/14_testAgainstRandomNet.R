rm(list=ls())
library(igraph)
library(bnlearn)
library(bestNormalize)

# load needed custom function
source('load_kgAsBN.R')

# load KG for computing max degree
kg = load_kgAsBN()
maxpVal = max(table(kg$arcs[,"to"]))

# load data
load("../data/ROSMAP/data_final_ROSMAP.RData") #encoded data
load("../data/ROSMAP/data_condensed_complete.RData") #original data
load('../results/bn_ROSMAP/finalBN_complete.Rdata')

orig_braak = data_all$phenotype$Phenotype_braaksc
bxcx_transform = boxcox(orig_braak)
bxcx_braak = bxcx_transform$x.t

data$phenotype = bxcx_braak

rosmap<-data
rosmap$SUBJID<-NULL
nodeNames <- colnames(rosmap)

mayo <- readRDS('../data/Mayo/data_final_imputed_onlyBraak.rds')


# define edges from blacklist
edges_notAllowed_to = c("PatDemo_educ", "PatDemo_sex", "PatDemo_apoe", "PatDemo_brainregion")
edges_notAllowed_from = c("phenotype")

# get random nets and their LL
LL_orig = read.csv2(file="../results/logliks_onlyBraak/logLiksBN_allData_onlyBraak.csv", sep=",", stringsAsFactors = F)
LL_orig_ROSMAP = as.numeric(LL_orig$likelihood[LL_orig$type=="ROSMAP"])
LL_orig_ROSMAP_Mayo = as.numeric(LL_orig$likelihood[LL_orig$type=="Mayo"])


rdm_LL_rosmap = list()
rdm_LL_mayo = list()
diff_orig_rand_rosmap = list()
diff_orig_rand_mayo = list()
set.seed(5486)
# get random graphs and delete edges from blacklist
for (i in 1:1000) {
  #print(i)
  # create a random graph with the same maximum parents as set in original approach
  rdm_graph = random.graph(nodeNames, method = "melancon", max.in.degree = maxpVal)
  # get random graphs' edges
  rdm_graph_edges = as.data.frame(rdm_graph$arcs)
  # filter out edges from blacklist
  rdm_graph_edges_filt = subset(rdm_graph_edges, !(from%in%edges_notAllowed_from) & !(to%in%edges_notAllowed_to) )
  # create a graph with filtered arc set
  rd_graph_filt = empty.graph(nodeNames)
  arcs(rd_graph_filt) = as.matrix(rdm_graph_edges_filt)
  # fit net
  fitted = bn.fit(rd_graph_filt, rosmap, method="mle")
  # get LL
  rdm_LL_rosmap[[i]] = logLik(fitted, rosmap)#, by.sample=TRUE)
  rdm_LL_mayo[[i]] = logLik(fitted, mayo)#, by.sample=TRUE)
  
  bysample_rosmap = logLik(fitted, rosmap, by.sample=TRUE)
  bysample_mayo = logLik(fitted, mayo, by.sample=TRUE)

  #get diff to orig
  diff_orig_rand_rosmap[[i]] = LL_orig_ROSMAP - bysample_rosmap
  diff_orig_rand_mayo[[i]] = LL_orig_ROSMAP_Mayo - bysample_mayo

}

rdm_LL_rosmap = as.numeric(rdm_LL_rosmap)
rdm_LL_mayo = as.numeric(rdm_LL_mayo)

counts_origLL_greater_rdmLL_ros = sum(LL_orig_ROSMAP>rdm_LL_rosmap)
counts_origLL_greater_rdmLL_mayo = sum(LL_orig_ROSMAP_Mayo>rdm_LL_mayo)

print(counts_origLL_greater_rdmLL_ros) #1000
print(counts_origLL_greater_rdmLL_mayo) #965

p_rosmap = (1+1000-counts_origLL_greater_rdmLL_ros)/1000
p_rosmap # 0.001
p_mayo = (1000-counts_origLL_greater_rdmLL_mayo)/1000
p_mayo #0.035

diff_orig_rand_rosmap_all = LL_orig_ROSMAP - rdm_LL_rosmap
diff_orig_rand_mayo_all = LL_orig_ROSMAP_Mayo - rdm_LL_mayo

data_lik<-data.frame(
  likelihood=c(diff_orig_rand_rosmap_all,diff_orig_rand_mayo_all),
  type=c(rep('ROSMAP', 1000),rep('Mayo', 1000))
)

write.table(data_lik, file="../results/logliks_onlyBraak/logLiksBN_diff_orig_vs_rand_onlyBraak.csv", quote=F, row.names = F, sep=",")


######################per sample####################################

diff_orig_rand_rosmap_mat = do.call(rbind, diff_orig_rand_rosmap)
diff_orig_rand_rosmap_mean = colMeans(diff_orig_rand_rosmap_mat)

diff_orig_rand_mayo_mat = do.call(rbind, diff_orig_rand_mayo)
diff_orig_rand_mayo_mean = colMeans(diff_orig_rand_mayo_mat)

data_lik<-data.frame(
  likelihood=c(diff_orig_rand_rosmap_mean,diff_orig_rand_mayo_mean),
  type=c(rep('ROSMAP',NROW(rosmap)),rep('Mayo',NROW(mayo)))
)

write.table(data_lik, file="../results/logliks_onlyBraak/logLiksBN_diff_orig_vs_rand_onlyBraak_perPatient.csv", quote=F, row.names = F, sep=",")



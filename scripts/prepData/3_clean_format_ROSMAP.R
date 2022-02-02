rm(list=ls())

library("org.Hs.eg.db")

# load data
load(file="../data/ROSMAP/ROSMAP_ADdata_scaledBatch.Rdata")
dataExpr = adFinalROSMAP #expression data
dataMeta = metaROSMAP #metadata

# load clustering
mkvClust = readRDS(file="../results/markovClust/mkvClust_msFinal_adapt2Dat.rds") #cluster membership

## filter for genes in KG
# annotate genes
## map genes
mapGeneNames_h2e = function(hid) {
  neighbors_ensemble = unlist(mapIds(org.Hs.eg.db,
                                     keys=hid,
                                     column="ENSEMBL",
                                     keytype="SYMBOL",
                                     multiVals="first"))
  return(neighbors_ensemble)
}

mapGeneNames_e2h = function(eid) {
  geneNames = unlist(mapIds(org.Hs.eg.db,
                            keys=eid,
                            column="SYMBOL",
                            keytype="ENSEMBL",
                            multiVals="first"))
  return(geneNames)
}

data_kg = subset(dataExpr, select = colnames(dataExpr)%in%mapGeneNames_h2e(mkvClust$gene))
colnames(data_kg) = mapGeneNames_e2h(colnames(data_kg))

# use only genes available in dataset
mkvClust = subset(mkvClust, gene%in%colnames(data_kg))


### concat expr and meta data
if (all(rownames(data_kg)==rownames(dataMeta))) {
  dataALL = data.frame(data_kg, dataMeta)
} else {
  print("ERROR")
  break
}


# Replace INF value with NA
invisible(do.call(data.frame,lapply(dataALL, function(x) replace(x, is.infinite(x),NA))))

### extract variable groups
nClust = length(unique(mkvClust$clust)) + sum(mkvClust$clust==0) - 1 # -1 due to counting of 0 cluster in first term
mClust = table(mkvClust$clust)
# one group for every SCC that has more than one gene
whichClustLarge = names(mClust)[mClust>1]
whichClustLarge = setdiff(whichClustLarge, "0")

dataAllClust = list()
for (i in 1:length(whichClustLarge)) {
  whichClust = whichClustLarge[i]
  genesClust = subset(mkvClust, clust==whichClust)
  genesClust = genesClust$gene
  genesClust = as.character(genesClust)
  whichCols = colnames(dataALL)[colnames(dataALL)%in%genesClust]
  dataClust = subset(dataALL, select = whichCols)
  rownames(dataClust) = NULL
  colnames(dataClust) = paste0(whichClust, "_", colnames(dataClust))
  dataAllClust[[i]] = dataClust
}

### patient data
pat_demo<-dataALL[,grepl('sex|age|educ|apoe|brainregion',colnames(dataALL))]
for(col in colnames(pat_demo)){
  if (col!="educ" & col!="age") {
    pat_demo[,col]<-factor(pat_demo[,col])
  }
}
colnames(pat_demo)<-paste0('PatDemo_',colnames(pat_demo))

summary(pat_demo)
apply(pat_demo, 2, sd)

### phenotype
phenotype_dat<-dataALL[,grepl('mmse|braaksc',colnames(dataALL))]
colnames(phenotype_dat)<-paste0('Phenotype_',colnames(phenotype_dat))

summary(phenotype_dat)
apply(phenotype_dat, 2, sd)
table(phenotype_dat$Phenotype_braaksc)

#otherGeneDat
otherGenes = mkvClust$gene[!(mkvClust$clust%in%whichClustLarge)]
otherGenes = as.character(otherGenes)
whichColsSmall = colnames(dataALL)[colnames(dataALL)%in%otherGenes]
datOtherGenes = subset(dataALL, select = whichColsSmall)
rownames(datOtherGenes) = NULL

data_all = dataAllClust
data_all[[length(whichClustLarge)+1]] = phenotype_dat
data_all[[length(whichClustLarge)+2]] = cbind(pat_demo, datOtherGenes)
names(data_all) = c(paste0("Cluster_", whichClustLarge), "phenotype", "stalone")

saveRDS(data_all,"../data/ROSMAP/data_condensed_complete.rds")
save(data_all, file = "../data/ROSMAP/data_condensed_complete.RData")

for (datan in names(data_all)) {
  data = data_all[[datan]]
  write.table(data,paste0("../python/ROSMAP/GridSearch/data_python/", datan, '.csv'),sep=',',row.names = F,col.names = F,quote=F, na = "NaN")
  
  write.table(rownames(dataALL),paste0('../python/ROSMAP/GridSearch/python_names/',datan,'_subj.csv'),sep=',',row.names = F,col.names = T,quote=T, na = "NaN")
  write.table(colnames(data),paste0('../python/ROSMAP/GridSearch/python_names/',datan,'_cols.csv'),sep=',',row.names = F,col.names = T,quote=T, na = "NaN")
  
}

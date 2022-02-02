rm(list=ls())

library("org.Hs.eg.db")
library(igraph)

# load one dataset for filtering
load(file="../data/ROSMAP/ROSMAP_ADdata_scaledBatch.Rdata")
dataExpr = adFinalROSMAP #expression data

# load graph
load("../data/knowledgeGraph/kg_clusterFinal.RData")

# load clustering
mkvClust = readRDS(file="../results/markovClust/mkvClust_msFinal.rds") #cluster membership
# summarize cluster 4 and 5 into one cluster, because both are TGF-beta signalling
mkvClust$clust[mkvClust$clust == 5] = 4

##### filter KG for genes in data
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
singleGenesNotinData = mkvClust$gene[mkvClust$clust == 0][!((mkvClust$gene[mkvClust$clust == 0])%in%colnames(data_kg))]
mkvClust = subset(mkvClust, gene%in%colnames(data_kg))
genes = mkvClust$gene

##### get new knowledge graph
mClust = table(mkvClust$clust)
whichClustLarge = names(mClust)[mClust>1]
whichClustLarge = setdiff(whichClustLarge, "0")

edgeList = as_edgelist(kg_cluster)

# summarize cluster 4 and 5 into one cluster, because both are TGF-beta signalling
mkvClust$clust[mkvClust$clust == 5] = 4
edgeList[edgeList == "Cluster_5"] = "Cluster_4"

edgeList = as.data.frame(edgeList)
colnames(edgeList) = c("from", "to")
edgeList$from = as.character(edgeList$from)
edgeList$to = as.character(edgeList$to)

edgeList = subset(edgeList, subset = from!=to)
edgeList = unique(edgeList, MARGIN = 1)

# subset Clusters having only one gene with gene name
oneGeneCluster = mkvClust$gene[!(mkvClust$clust%in%whichClustLarge)]
oneGeneCluster = as.character(oneGeneCluster)

for (gene in oneGeneCluster) {
  clust_ms = paste0("Cluster_", mkvClust$clust[mkvClust$gene == gene])
  edgeList[edgeList == clust_ms] = gene
}

## filter out single genes that are not in data
for (gene in singleGenesNotinData) {
  edgeList[ edgeList==gene ] = NA
}

edgeList = edgeList[complete.cases(edgeList),]
edgeListMat = as.matrix(edgeList)
kg_cluster = graph_from_edgelist(edgeListMat)

# save collapsed knowledge graph
saveRDS(kg_cluster, file="../data/knowledgeGraph/kg_clusterFinal_adapt2Dat.rds")
save(kg_cluster, file="../data/knowledgeGraph/kg_clusterFinal_adapt2Dat.RData")

# save collapsed cluster membership
saveRDS(mkvClust, file="../results/markovClust/mkvClust_msFinal_adapt2Dat.rds")
save(mkvClust, file="../results/markovClust/mkvClust_msFinal_adapt2Dat.RData")
write.table(mkvClust, file="../results/markovClust/mkvClust_msFinal_adapt2Dat.tsv", sep="\t", quote=F, row.names = F)


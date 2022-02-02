rm(list=ls())
library(igraph)
library(stringr)
library(MCL)
library(dplyr)

### load BEL relations
HGNC_namespace = read.csv("BELgraph.csv", sep="\t")
allowedRelations = c("increases", "directly_increases", "decreases", "directly_decreases", "regulates")   # causes_no_change, association, negative/positive correlation

genePairs = subset(HGNC_namespace, select = c(gene1, gene2, relation))#, subset = relation%in%allowedRelations)
genePairs = subset(genePairs,relation%in%allowedRelations)
genePairs = unique(genePairs, MARGIN = 1) # find unique rows
genePairs = subset(genePairs, as.character(gene1)!=as.character(gene2))

genesKnowledgeGraph = unique(c(as.character(genePairs$gene1), as.character(genePairs$gene2)))

### get igraph
edgeList = as.matrix(subset(genePairs, select = c(gene2, gene1)))
edgeListUnique = as.matrix(distinct(as.data.frame(edgeList)))
knowledgeGraph = graph_from_edgelist(edgeListUnique)

save(knowledgeGraph, file="../data/knowledgeGraph.RData")

kg = knowledgeGraph

plotNets(kg, plotOption=3)

### get disconnected components of the graph and select the largest one
components <- decompose(kg, min.vertices=2)
kg = components[[1]]

## Markov Clustering
kg_adj = as_adjacency_matrix(kg, sparse = F)

checkBipartite = !bipartite_mapping(kg)$res #check if graph is bipartite, if no: addLoops needs to be TRUE

mkvClust = mcl(x = kg_adj, addLoops = checkBipartite, allow1 = FALSE, expansion = 2, inflation = 1) 
# single clusters are grouped into one cluster (the 0 one)

nClust = mkvClust$K
mkvClust_ms = data.frame(gene = rownames(kg_adj), clust = mkvClust$Cluster)

#rename clusters more intuitively
names_clust = data.frame(new=c(0:(nClust-1)), old=NA)
names_clust$old = as.integer(c(0,setdiff(names(sort(table(mkvClust_ms$clust), decreasing = T)), "0")))
mkvClust_ms_replaced = mkvClust_ms
for (i in 1:nClust) {
  mkvClust_ms_replaced$clust[mkvClust_ms$clust == names_clust$old[i]] = names_clust$new[i]
}
table(mkvClust_ms_replaced$clust)
rm("mkvClust_ms")

mkvClust_ms = mkvClust_ms_replaced
rm("mkvClust_ms_replaced")
mkvClust_ms$gene = str_replace(mkvClust_ms$gene, "-", ".")

plotNets(kg, plotOption=1)

save(mkvClust_ms, file="data/mkvClustResult.RData")


##### get new knowledge graph
edgeList = as_edgelist(kg)
edgeList[,1] = str_replace(edgeList[,1], "-", ".")
edgeList[,2] = str_replace(edgeList[,2], "-", ".")

mcluster = table(mkvClust_ms$clust)
whichClustLarge = names(mcluster)[names(mcluster)!="0"]

for (i in 1:length(whichClustLarge)) {
  whichClust = whichClustLarge[i]
  genesClust = subset(mkvClust_ms, clust==whichClust)
  genesClust = genesClust$gene
  genesClust = as.character(genesClust)
  edgeList[which(edgeList%in%genesClust, arr.ind = T)] = paste0("Cluster_", whichClust)
}

singleGenes = unique(edgeList[!startsWith(edgeList, "Cluster")])

edgeList = as.data.frame(edgeList)
colnames(edgeList) = c("from", "to")
edgeList$from = as.character(edgeList$from)
edgeList$to = as.character(edgeList$to)

edgeList = subset(edgeList, subset = from!=to)
edgeList = unique(edgeList, MARGIN = 1)
keepGenes = names(which(table(subset(edgeList, to%in%singleGenes | from%in%singleGenes)$from)>1))
#all genes which are connected to more than one other node (here: only Cluster)
edgeListMat = as.matrix(edgeList)

kg_cluster = graph_from_edgelist(edgeListMat)

plotNets(kg_cluster, plotOption=2)

### add single genes to connected cluster, if this is the only connection
verticesToCollapse = subset(singleGenes, !(singleGenes%in%keepGenes))

#posCollapse = which(mkvClust_ms$gene%in%verticesToCollapse)

for (i in 1:length(verticesToCollapse)) {
  geneToCollapse = verticesToCollapse[i]
  
  ## get the cluster a gene is connected to
  edgeList_sub = subset(edgeList, from==geneToCollapse)
  if (NROW(edgeList_sub)!=1) {
    next
  } else {
    clustCollapsed = edgeList_sub$to
    ## change the cluster membership to the cluster the gene is collapsed into
    mkvClust_ms$clust[mkvClust_ms$gene==geneToCollapse] = strsplit(clustCollapsed, "Cluster_")[[1]][2]
    ## replace the collapsed gene by the cluster it was collpased into
    edgeList[which(edgeList==geneToCollapse, arr.ind = T)] = clustCollapsed
  }
}

edgeList = subset(edgeList, subset = from!=to)
edgeList = unique(edgeList, MARGIN = 1)
edgeListMat = as.matrix(edgeList)

kg_cluster = graph_from_edgelist(edgeListMat)
#kg_cluster = delete_vertices(kg_cluster, v=verticesToCollapse)

plotNets(kg_cluster, plotOption=2)

# save collapsed knowledge graph
save(kg_cluster, file="../data/knowledgeGraph/kg_clusterFinal.RData")

# save collapsed cluster membership
save(mkvClust_ms, file="results/markovClust/mkvClust_msFinal.RData")
write.table(mkvClust_ms, file="../results/markovClust/mkvClust_msFinal.tsv", sep="\t", quote=F, row.names = F)

### enrichment analysis large SCC
library(enrichR)
library(clusterProfiler)

### NeuroMMSig Overrepresentation analysis
neuroMMSig_mechanisms = read.table(file="../data/neuroMMSig_AD/neuroMMSig_AD_onlyGeneSets.CSV", sep=";", header = T)
neuroMMSig_mechanisms$X = NULL

neuroMMSig_filtered_mech = unique(subset(neuroMMSig_mechanisms, Genes%in%mkvClust_ms$gene)$Subgraph.Name)
neuroMMSig_filtered = subset(neuroMMSig_mechanisms, Subgraph.Name%in%neuroMMSig_filtered_mech)

resSigList2 = list()

for (i in 1:length(whichClustLarge)) {
  whichClust = whichClustLarge[i]
  genesClust = subset(mkvClust_ms, clust==whichClust)
  genesClust = genesClust$gene
  genesClust = as.character(genesClust)
  if (sum(genesClust%in%neuroMMSig_filtered$Genes)==0) {
    resSigList2[[i]] = NA
    next
  } else {
    res = enricher(genesClust, TERM2GENE = neuroMMSig_filtered, minGSSize = 1)@result
    resSig = subset(res, p.adjust<0.05)
    #write.table(resSig, file=paste0("results/markovClust/enrichmentNeuroMMSig/cluster_", whichClust, "_enrichment_neuroMMSig.tsv"), quote=F, row.names = F, sep="\t")
    resSigList2[[i]] = resSig
  }
}

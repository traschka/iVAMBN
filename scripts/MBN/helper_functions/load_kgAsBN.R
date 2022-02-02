
######### merge all data into right directories
load_kgAsBN<-function(){
  library(igraph)
  load("../data/knowledgeGraph/kg_clusterFinal_adapt2Dat.RData")
  kg_edgeList = as_edgelist(kg_cluster)
  colnames(kg_edgeList) = c("from", "to")
  covariables = colnames(data)[grep("PatDemo_", colnames(data))]
  nodes = c(unique(c(kg_edgeList[,1], kg_edgeList[,2])), covariables, "phenotype")
  kg_bn = empty.graph(nodes)
  arcs(kg_bn) = kg_edgeList
  save(kg_bn, file = "../data/knowledgeGraph/kg_bn.RData")
  return(kg_bn)
}

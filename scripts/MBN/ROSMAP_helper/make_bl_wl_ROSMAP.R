make_bl_wl_ROSMAP_noKL<-function(data,blname,wlname) { #,out=F) { #,orphans){
  library(tidyverse)
  # get all possible combinations
  datan<-names(data)
  combs<-expand.grid(datan,datan)
  colnames(combs)<-c('from','to')
  
  # blacklist
  #  - all from gene (modules) to covariables, but not to ageGroup
  #  - all between covariables, but not to ageGroup (bc age is at death)
  
  combs$fromGeneToCov = ((!grepl("PatDemo_", combs$from)) & (grepl("PatDemo_", combs$to)))
  combs$btwCov = ((grepl("PatDemo_", combs$from)) & (grepl("PatDemo_", combs$to)))
  combs$toAge = grepl("age", combs$to)
  combs$fromCognition = grepl("phenotype", combs$from)
  
  bl<-subset(combs, subset = fromGeneToCov | btwCov)
  bl<-subset(bl, subset = !toAge, select = c(from,to))
  bl<-rbind(bl, subset(combs, subset = fromCognition, select = c(from, to)))
  
  # whitelist
  # no whitelist needed in that case
  
  # remove loop (shouldnt matter as bnlearn should do it automatically)
  bl<-bl[bl$from!=bl$to,]
  bl<-bl[!duplicated(bl), ]

  # write lists to file
  write.csv(bl, blname, row.names = F)
}


make_bl_wl_ROSMAP_completeKL<-function(data,blname,wlname,out=F) { #,orphans){
  library(tidyverse)
  library(igraph)
  # get all possible combinations
  datan<-names(data)
  combs<-expand.grid(datan,datan)
  colnames(combs)<-c('from','to')
  
  kg = readRDS("data/knowledgeGraph/kg_clusterFinal_adapt2Dat.rds")
  
  # whitelist
  #  - all that are in KG
  wl = as_edgelist(kg)
  colnames(wl) = c("from", "to")
  wl = as.data.frame(wl)
  
  # blacklist
  #  - all gene/cluster to gene/cluster edges that are not in KG
  #  - allowed are: Cov -> genes , genes -> Age , any -> cognition 
  combs = combs[combs$from!=combs$to,]
  combs$check = paste0(combs$from, combs$to)%in%paste0(wl$from, wl$to)
  bl = subset(combs, subset = !check, select = -check)
  
  # remove edges from bl that come from demografics or that goes into age and cognition
  combs$Cov2Gene = ((!grepl("PatDemo_", combs$to)) & (grepl("PatDemo_", combs$from)))
  combs$Gen2Age = (grepl("age", combs$to) & (!grepl("phenoypte", combs$from)))
  combs$toCognition = grepl("phenotype", combs$to)
  
  cov2gene= subset(combs, subset = Cov2Gene, select = c(from, to))
  gen2age = subset(combs, subset = Gen2Age, select = c(from, to))
  toCognition = subset(combs, subset = toCognition, select = c(from, to))
  bl = subset(bl, subset = !(from%in%cov2gene$from & to%in%cov2gene$to))
  bl = subset(bl, subset = !(from%in%gen2age$from & to%in%gen2age$to))
  bl = subset(bl, subset = !(from%in%toCognition$from & to%in%toCognition$to))
  
  # remove loop (shouldnt matter as bnlearn should do it automatically)
  bl<-bl[bl$from!=bl$to,]
  bl<-bl[!duplicated(bl), ]
  
  # write lists to file
  write.csv(bl, blname, row.names = F)
  write.csv(wl, wlname, row.names = F)
}

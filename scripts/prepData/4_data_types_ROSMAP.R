rm(list=ls())

data_all = readRDS("../data/ROSMAP/data_condensed_complete.rds")
library(ggdendro)
library(ade4)
library(PTXQC)
dataFrameNamesCombine = c()
for(i in 1:length(data_all)){
  print(i)
  dataFrameNames = names(data_all)[i]
  print(dataFrameNames)
  sv = data.frame()
  #print(sv)
  #print(str(data_all[[i]]))
  for(j in 1:ncol(data_all[[i]])){
    print(j)
    #print(data_all[[i]][[j]])
    if(class(data_all[[i]][[j]]) == "factor"){
      sv[j,"type"] = "cat"
      sv[j,"dim"] = nlevels(data_all[[i]][[j]])
      sv[j,"nclass"] = nlevels(data_all[[i]][[j]])
    }
    if(class(data_all[[i]][[j]]) == "integer"){
      sv[j,"type"]= "ordinal"
      sv[j,"dim"] = length(unique(data_all[[i]][[j]]))
      sv[j,"nclass"] = length(unique(data_all[[i]][[j]]))
    }
    if(class(data_all[[i]][[j]]) == "numeric"){
      #print("bvbvbv")
      sv[j,"type"] = "real"
      sv[j,"dim"] = 1
      sv[j,"nclass"] = ""
      if(!startsWith(dataFrameNames, "Cluster")) {
        if(all(data_all[[i]][[j]] >= 0) == TRUE){
          #print("bvbvbv")
          sv[j,"type"] = "pos"
          sv[j,"dim"] = 1
          sv[j,"nclass"] = ""
        }
        if(all(data_all[[i]][[j]] == round(data_all[[i]][[j]])) == TRUE){
          #print("bvbvbv")
          sv[j,"type"] = "count"
          sv[j,"dim"] = 1
          sv[j,"nclass"] = ""
        }
      }
    }
  }
  sv = sv[complete.cases(sv), ]
  print(sv)
  if("age" %in% colnames(data_all[[i]])){
    write.table(sv, paste0("../python/ROSMAP/GridSearch/data_python/","stalone_types.csv"),  sep=',',row.names = F,col.names = T,quote=T, na = "NaN")
  } else{
    write.table(sv,paste0('../python/ROSMAP/GridSearch/data_python/',dataFrameNames,'_types.csv'),sep=',',row.names = F,col.names = T,quote=T, na = "NaN")
  }
  dataFrameNamesCombine = c(dataFrameNamesCombine, dataFrameNames)
}

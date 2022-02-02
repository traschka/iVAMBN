
######### merge all data into right directories
merge_data<-function(complete=F, dataset="ROSMAP"){
  data_all<-readRDS(file = paste0(data_out,'data_condensed_complete.rds'))
    #(meta)
    data_meta<-read.csv(paste0('../python/', dataset, '/HIVAE_training/metaenc.csv'))
    if (dataset=="Mayo") {
      name <- 'data_final_Mayo'
    } else {
      name<-'data_final_ROSMAP'
    }
  
  #(standalone)
  data_stalone<-data_all[['stalone']]

  # merge all
  data<-cbind(data_meta,data_stalone)# %>% reduce(merge, by = 'SUBJID')
  
  #flag 0 var cols
  #print(colnames(data)[-includeVar(data)])
  whichSCode = !startsWith(prefix="scode", x=colnames(data))
  data<-subset(data, select = whichSCode)
  
  data$SUBJID<-factor(data$SUBJID)
  # refactor all factor columns (so there are no empty levels)
  for(col in colnames(data)){
    if (is.factor(data[,col])|grepl('scode_',col)){
      data[,col]<-factor(data[,col])
    }else if (is.factor(data[,col])){
      data[,col]<-as.numeric(data[,col])
    }
  }
  
  colnames(data)<-gsub('zcode_', '', colnames(data))
  save(data,file=paste0(data_out,name,'.RData'))
  saveRDS(data,paste0(data_out,name,'.rds'))
  return(data)
}

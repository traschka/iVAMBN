simulate_cd33ko = function(data, net, nodes, n=NA){
  
  library(randomForestSRC)
  library(bnlearn)
  
  n_VP=ifelse(is.na(n),NROW(data),n)
  
  VP = c()
  iter = 1
  
  # loops until we have a full dataset of VPs (overshoots so data is not always < n_ppts)
  while(NROW(VP) < n_VP){
    cat("iteration = ", iter, "\n")
    
    # generate data (until no NAs in any variables)
    generatedDF = cpdist(fitted=net, nodes=nodes, evidence = (CD33 == 0), n=n_VP)
    generatedDF = cbind(generatedDF, CD33=0)
    comp<-F
    while (!comp){ # using mixed data sometimes results in NAs in the generated VPs. These VPs are rejected.
      generatedDF<-generatedDF[complete.cases(generatedDF),]
      gen<-n_VP-dim(generatedDF)[1]
      if (gen>0){
        generatedDF_new = cpdist(net, nodes=nodes, evidence = (CD33 == 0), n=gen)
        generatedDF_new = cbind(generatedDF, CD33=0)
        generatedDF<-rbind(generatedDF,generatedDF_new) # draw virtual patients
      }else{
        comp<-T 
      }
    }
    
    VP = rbind.data.frame(VP, generatedDF)
    iter = iter + 1
    print(NROW(VP))
  }
  return(VP)
}

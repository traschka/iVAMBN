############################
############################ Dependencies and helper functions
############################
rm(list=ls())
library(bnlearn)
library("pcalg")
library(ggplot2)

finalBN = readRDS(file='../results/bn_ROSMAP/finalBN_complete.rds')
fitted = readRDS(file="../results/bn_ROSMAP/finalBN_fitted_complete.rds")

######### CD33 KO simulation through bn
data = readRDS(file="../data/ROSMAP/data_final_ROSMAP.rds")
data$SUBJID = NULL
allNodes = colnames(data)

koCD33<-fitted
koCD33$CD33 = list(coef = fitted$CD33$coefficients-0.1, sd = fitted$CD33$sd)

WT<-cpdist(fitted, nodes=allNodes,evidence=T, method = "lw",n=length(data[,1])) # draw samples from WT bayes net
KO<-cpdist(koCD33, nodes=allNodes,evidence=T, method = "lw",n=length(data[,1])) # draw samples from KO bayes net

## plot distributions
load(file = "../data/ROSMAP/simulated_cd33ko.RData")
### CD33
dat<-data.frame(dv=c(WT$CD33,KO$CD33),level=factor(c(rep('original',dim(WT)[1]),rep('CD33 underexpressed',dim(KO)[1]))))
pl<-ggplot(dat, aes(x=dv,fill=level))+geom_density(alpha=.2)+theme_bw()+theme(legend.position = "none")+xlab("CD33")
pl

pdf(file="../results/ko_cd33/ko_cd33_CD33.pdf", width = 6, height = 4)
pl
dev.off()


### phenotype
dat<-data.frame(dv=c(WT$phenotype,KO$phenotype),level=factor(c(rep('original',dim(WT)[1]),rep('CD33 underexpressed',dim(KO)[1]))))
pl<-ggplot(dat, aes(x=dv,fill=level))+geom_density(alpha=.2)+xlab('phenotype')+theme_bw()+theme(legend.position = "none")
pl

pdf(file="../results/ko_cd33/ko_cd33_phenotype.pdf", width = 6, height = 4)
pl
dev.off()


### PatDemo_age (parent of CD33)
dat<-data.frame(dv=c(WT$PatDemo_age,KO$PatDemo_age),level=factor(c(rep('original',dim(WT)[1]),rep('CD33 underexpressed',dim(KO)[1]))))
pl<-ggplot(dat, aes(x=dv,fill=level))+geom_density(alpha=.2)+xlab('PatDemo: Age')+theme_bw()+theme(legend.position = "none")
pl

### Cluster_1 (children of CD33)
dat<-data.frame(dv=c(WT$Cluster_1,KO$Cluster_1),level=factor(c(rep('original',dim(WT)[1]),rep('CD33 underexpressed',dim(KO)[1]))))
pl<-ggplot(dat, aes(x=dv,fill=level))+geom_density(alpha=.2)+xlab('GABA subgraph')+theme_bw()+theme(legend.position = "none")
pl

pdf(file="../results/ko_cd33/ko_cd33_GABA.pdf", width = 6, height = 4)
pl
dev.off()


### Cluster_18 (children of CD33)
dat<-data.frame(dv=c(WT$Cluster_18,KO$Cluster_18),level=factor(c(rep('original',dim(WT)[1]),rep('CD33 underexpressed',dim(KO)[1]))))
pl<-ggplot(dat, aes(x=dv,fill=level))+geom_density(alpha=.2)+xlab('Amyloidogenic subgraph')+theme_bw()+theme(legend.position = "none")
pl

pdf(file="../results/ko_cd33/ko_cd33_amyloidogenic.pdf", width = 6, height = 4)
pl
dev.off()

### Cluster_20 (children of CD33)
dat<-data.frame(dv=c(WT$Cluster_20,KO$Cluster_20),level=factor(c(rep('original',dim(WT)[1]),rep('CD33 underexpressed',dim(KO)[1]))))
pl<-ggplot(dat, aes(x=dv,fill=level))+geom_density(alpha=.2)+xlab('Acetylcholine signaling subgraph')+theme_bw()+theme(legend.position = "none")
pl

pdf(file="../results/ko_cd33/ko_cd33_acetylcholine.pdf", width = 6, height = 4)
pl
dev.off()

### Cluster_3 (children of CD33)
dat<-data.frame(dv=c(WT$Cluster_3,KO$Cluster_3),level=factor(c(rep('original',dim(WT)[1]),rep('CD33 underexpressed',dim(KO)[1]))))
pl<-ggplot(dat, aes(x=dv,fill=level))+geom_density(alpha=.2)+xlab('Prostaglandin subgraph')+theme_bw()+theme(legend.position = "none")
pl

pdf(file="../results/ko_cd33/ko_cd33_prostaglandin.pdf", width = 6, height = 4)
pl
dev.off()

### Cluster_9 (children of CD33)
dat<-data.frame(dv=c(WT$Cluster_9,KO$Cluster_9),level=factor(c(rep('original',dim(WT)[1]),rep('CD33 underexpressed',dim(KO)[1]))))
pl<-ggplot(dat, aes(x=dv,fill=level))+geom_density(alpha=.2)+xlab('Chaperone subgraph')+theme_bw()+theme(legend.position = "none")
pl

pdf(file="../results/ko_cd33/ko_cd33_chaperone.pdf", width = 6, height = 4)
pl
dev.off()

### TRAF1 (children of CD33)
dat<-data.frame(dv=c(WT$TRAF1,KO$TRAF1),level=factor(c(rep('original',dim(WT)[1]),rep('CD33 underexpressed',dim(KO)[1]))))
pl<-ggplot(dat, aes(x=dv,fill=level))+geom_density(alpha=.2)+xlab('TRAF1')+theme_bw()+theme(legend.position = "none")
pl

pdf(file="../results/ko_cd33/ko_cd33_TRAF1.pdf", width = 6, height = 4)
pl
dev.off()


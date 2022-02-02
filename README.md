<h1 align="center">
  iVAMBN
</h1>


## Table of Contents

* [General Info](#general-info)
* [Workflow](#workflow)
* [iVAMBN model](#ivambn-model)
* [Supplementary Data](#supplementary-data)

## General Info
This repository contains code and data for our paper, "AI reveals insights into link between CD33 and cognitive impairment in Alzheimer's Disease"), which generates a quantitative model that allowed us to simulate a down-expression of the putative drug target CD33, including potential impact on phenotype.

## Workflow
The overall workflow of iVAMBN is based on the VAMBN modelling approach described in [Variational Autoencoder Modular Bayesian Networks for Simulation of Heterogeneous Clinical Study Data](https://doi.org/10.3389/fdata.2020.00016) but additionally incorporates a Knowledge Graph into the modelling. Therefore the workflow is defined as following and the corresponding [scripts](/scripts) need to be run in the given order:

1. Clustering the knowledge graph
  - [cluster knowledge graph](scripts/knowledgeGraph/1_getMarkovClusters.R)
  - [adapt to data](scripts/knowledgeGraph/2_adaptKGtoData.R)
2. Train HI-VAEs based on clustered modules
  - [preparation of data for HI-VAE](scripts/prepData/)
  - [Grid Search](scripts/HI-VAE/GridSearch/5_GridSearch_ROSMAP.py)
  - [Find best hyperparameters](scripts/HI-VAE/GridSearch/6_extractBestHyperparameters.ipynb)
  - [train HI-VAEs](HI-VAE/HIVAE_training/7_ROSMAP_HIVAE_training.py)
  - [get embeddings](HI-VAE/HIVAE_training/8_ROSMAP_getMetaEnc.py)
  - [plot embeddings](HI-VAE/HIVAE_training/9_ROSMAP_plotMetaEnc.ipynb)
  - [get decoded values](HI-VAE/HIVAE_training/10_ROSMAP_decodingRP.py)
  - [get log-likelihoods](HI-VAE/HIVAE_training/11_HIVAE_logLikelihoods.py)
3. Train the MBN with autoencoded values
  - [train modular bayesian network](MBN/12-1_bnet_ROSMAP.R)
4. Apply on Mayo dataset
  - [apply HI-VAEs](HI-VAE/Mayo/M1_Mayo_HIVAE_apply.py)
  - [get embeddings](HI-VAE/Mayo/M2_Mayo_getMetaEnc.py)
  - [get decoded values](HI-VAE/Mayo/M3_Mayo_decodingRP.py)
  - [get log-likelihoods](HI-VAE/Mayo/M4_HIVAE_logLikelihoods.py)
  - [train modular bayesian network with only Braak stages](MBN/12-2_bnet_ROSMAP_onlyBraak.R)
5. Get statistics
  - [get log-likelihoods of MBN](MBN/13_bnet_likelihoods_allAD.R)
  - [test MBN against random nets](MBN/14_testAgainstRandomNet.R)
  - [decode simulated WT samples](HI-VAE/HIVAE_training/15_decoding_simulatedCD33WT.py)
4. Simulate CD33 down-expression
  - [simulate CD33 down-expression samples](MBN/16_simulateCD33KO.R)
  - [decode simulated CD33 down-expression samples](HI-VAE/HIVAE_training/17_decoding_CD33KO.py)

## iVAMBN model
The final trained iVAMBN model, as well as the BN structure object, the iVAMBN model with Braak stages only, and the HI-VAEs for each module can be found in the [models](/models) section.
- [final fitted MBN](models/MBN/finalBN_fitted.Rdata)
- [BN structure](models/MBN/finalBN.Rdata)
- [MBN with Braak stages only](models/MBN/finalBN_onlyBraak_fitted.Rdata)
- [HI-VAES](models/HI-VAE/)

## Supplementary Data
Additionally to the scripts and models, one can find also the supplementary information for the manuscript [here](/data).
- [enrichment analysis of clusters of KG](/data/enrichment_KG_NeuroMMSig)
- [comparisoan of real and drawn samples](/data/realVSdrawn)
- [Knowledge Graph](/data/BELgraph.csv)
- Black and white lists used, e.g. [black list](/data/bl_noKL.csv)
- [Markov Cluster Assignment](/data/mkvClust_msFinal.tsv)

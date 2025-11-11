# SIMLINK
# 1. Description

SIMLINK (Simultaneous Modeling of Linear and Nonlinear Components of Knowledge Graph) is a method using a knowledge graph–based framework to model gene–variant–feature associations while integrating a linear model to isolate the linear component of variant pathogenicity.

## 2. Requirements

python >= 3.6.0      

tensorflow = 1.15.0    

scipy = 1.4.1  


## 3. Data preparation

Before using UPPER, all data need be converted to knowledge graph. There are already processed data in the kgdata/class, but it requires decompression: 
```
tar -zxvf ./kgdata/class/kgdata.tar.gz
```
## 4. Train model

To train models for pathogenicity prediction of single nucleotide variants, run these commands:
```
./run_class.sh 1 train_clinvar_2022_all_test_clinvar_20230326 QuatE --save --rel_update
```
QuatE is the name of knowledge graph embedding method. feel free to replace it with: RotatE, TransE, TransD, TransH, DistMult.   

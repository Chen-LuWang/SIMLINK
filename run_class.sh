#!/bin/bash
# method: TransE, RotatE, DistMult, QuatE...

GPU=$1
DATASET=$2
MODEL=$3
ARGS="$4 $5 $6 $7"
LR=(0.01)
TRAIN_CSV_PATH="./csv_data/merged_test_clinvar_data.csv"
A=(0.3) 
MODE="missense"
SEED=(12345)



if [ ${DATASET} == "train_clinvar_2022_all_test_clinvar_20230326" ]
then
    DIM=(128)
    ALPHA=(0.23)
    LAYER=(2)
elif [ ${DATASET} == "train_clinvar_2022_test_clinvar_20230326" ]
then
    DIM=(128)
    ALPHA=(0.23)
    LAYER=(2)
    TEST_CSV_PATH="./csv_data/clinvar_20230326.csv"
fi





for a in "${A[@]}"
do
    for alpha in "${ALPHA[@]}"
    do
        for layer in "${LAYER[@]}"
        do
            for dim in "${DIM[@]}"
            do
                for lr in "${LR[@]}"
                do
                    for seed in "${SEED[@]}"
                    do
                        option="--dataset ${DATASET} --dim ${dim} --mode ${MODEL} --learning_rate ${lr} --train_csv ${TRAIN_CSV_PATH} --test_csv ${TEST_CSV_PATH} --missense_or_synonymous ${MODE} --a ${a} --alpha ${alpha} --beta ${alpha} --layer ${layer} --rel_update --epochs 1000 --randomseed ${seed} ${ARGS} "
                        cmd="CUDA_VISIBLE_DEVICES=${GPU} python train_class.py ${option}"
                        echo $cmd
                        eval $cmd
                    done
                done
            done
        done
    done
done
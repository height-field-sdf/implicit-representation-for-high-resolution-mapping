#!/bin/bash
# Example training script for HF-SDF (Height-Field SDF) shape space
# Adjust parameters below for your experiment

python shapespace/train.py \
    --batch_size 16 \
    --points_batch 4096 \
    --nepoch 10000 \
    --gpu 0 \
    --conf shapespace/dfaust_setup.conf \
    --split splits/your_train_split.json \
    --expname my_shapespace_exp

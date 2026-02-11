#!/bin/bash

# gpu 0 or 1
export CUDA_VISIBLE_DEVICES=0

python src/train.py \
    --config_path configs/config.yaml \
    --batch_size 16 \
    --lr 0.0001 \
    --num_epochs 100 \
    --backbone resnet18 \
    --num_classes 2
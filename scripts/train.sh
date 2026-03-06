#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train.py \
    --config_path configs/config.yaml \
    --batch_size 16 \
    --lr 0.0001 \
    --num_epochs 50 \
    --backbone ResNet50 \
    --num_classes 2
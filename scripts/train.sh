#!/bin/bash

python src/train.py \
    --config_path configs/config.yaml \
    --batch_size 16 \
    --lr 0.0001 \
    --num_epochs 100
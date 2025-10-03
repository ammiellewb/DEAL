#!/bin/bash
# filepath: /home/ammiellewb/DEAL/scripts/train.sh
# exectutable: chmod +x train.sh
# run: bash scripts/train.sh

python3 train_active.py \
    --dataset Cityscapes \
    --base-size 512,512 \
    --crop-size 512,512 \
    --workers 8 \
    --epochs 5 \
    --eval-interval 1 \
    --batch-size 4 \
    --gpu-ids 0 \
    --lr 0.01 \
    --lr-scheduler poly \
    --checkname demo_active_fast_scoring \
    --backbone mobilenet \
    --init-percent 10 \
    --select-num 20 \
    --with-pam \
    --with-mask \
    --max-percent 20 \
    --percent-step 5 \
    --active_selection_mode deal \
    --strategy DS \
    --seed 42 \
    --use-wandb --wandb-project "deal-implementation"
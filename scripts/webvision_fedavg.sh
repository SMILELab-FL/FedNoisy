#!/bin/bash
python fednoisy/algorithms/fedavg/main.py --dataset webvision \
    --model InceptionResNetV2 \
    --partition noniid-labeldir \
    --dir_alpha 0.6 \
    --num_clients 10 \
    --data_dir ../fedNLLdata/webvision \
    --out_dir ../Fed-Noisy-checkpoint/webvision/ \
    --com_round 180 \
    --epochs 5 \
    --sample_ratio 1.0 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --batch_size 64 \  # 128 can be too large for some GPU
    --seed 1

#!/bin/bash
python fednoisy/algorithms/fedavg/main.py --dataset clothing1m \
    --model ResNet50 \
    --partition iid \
    --num_clients 10 \
    --globalize \
    --noise_mode real \
    --data_dir ../fedNLLdata/clothing1m \
    --out_dir ../Fed-Noisy-checkpoint/clothing1m/ \
    --com_round 150 \
    --epochs 5 \
    --sample_ratio 1.0 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --batch_size 32 \
    --seed 1


python fednoisy/algorithms/fedavg/main.py --dataset clothing1m \
    --model ResNet50 \
    --partition noniid-labeldir \
    --dir_alpha 0.1 \
    --num_clients 10 \
    --globalize \
    --noise_mode real \
    --data_dir ../fedNLLdata/clothing1m \
    --out_dir ../Fed-Noisy-checkpoint/clothing1m/ \
    --com_round 150 \
    --epochs 5 \
    --sample_ratio 1.0 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --batch_size 32 \
    --seed 1


python fednoisy/algorithms/fedavg/main.py --dataset clothing1m \
    --model ResNet50 \
    --partition noniid-quantity \
    --dir_alpha 0.1 \
    --num_clients 10 \
    --globalize \
    --noise_mode real \
    --data_dir ../fedNLLdata/clothing1m \
    --out_dir ../Fed-Noisy-checkpoint/clothing1m/ \
    --com_round 150 \
    --epochs 5 \
    --sample_ratio 1.0 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --batch_size 32 \
    --seed 1 

python fednoisy/algorithms/fedavg/main.py --dataset clothing1m \
    --model ResNet50 \
    --partition noniid-#label \
    --major_classes_num 5 \
    --num_clients 10 \
    --globalize \
    --noise_mode real \
    --data_dir ../fedNLLdata/clothing1m \
    --out_dir ../Fed-Noisy-checkpoint/clothing1m/ \
    --com_round 150 \
    --epochs 5 \
    --sample_ratio 1.0 \
    --lr 0.01 \
    --momentum 0.9 \
    --weight_decay 5e-4 \
    --batch_size 32 \
    --seed 1

#!/bin/bash
python build_dataset_fed.py --dataset clothing1m \
    --partition iid \
    --num_clients 10 \
    --globalize \
    --noise_mode real \
    --raw_data_dir ../rawdata/clothing1M/ \
    --data_dir ../fedNLLdata/clothing1m \
    --seed 1

python build_dataset_fed.py --dataset clothing1m \
    --partition noniid-labeldir \
    --dir_alpha 0.1 \
    --num_clients 10 \
    --globalize \
    --noise_mode real \
    --raw_data_dir ../rawdata/clothing1M/ \
    --data_dir ../fedNLLdata/clothing1m \
    --seed 1

python build_dataset_fed.py --dataset clothing1m \
    --partition noniid-quantity \
    --dir_alpha 0.1 \
    --num_clients 10 \
    --globalize \
    --noise_mode real \
    --raw_data_dir ../rawdata/clothing1M/ \
    --data_dir ../fedNLLdata/clothing1m \
    --seed 1

python build_dataset_fed.py --dataset clothing1m \
    --partition noniid-#label \
    --major_classes_num 5 \
    --num_clients 10 \
    --globalize \
    --noise_mode real \
    --raw_data_dir ../rawdata/clothing1M/ \
    --data_dir ../fedNLLdata/clothing1m \
    --seed 1

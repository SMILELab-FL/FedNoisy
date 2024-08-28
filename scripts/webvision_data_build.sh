#!/bin/bash
python build_dataset_fed.py --dataset webvision \
    --partition iid \
    --num_clients 10 \
    --raw_data_dir ../rawdata/webvision1.0 \
    --raw_imagenet_dir ../rawdata/imagenet_data/ILSVRC2012 \
    --data_dir ../fedNLLdata/webvision \
    --seed 1

python build_dataset_fed.py --dataset webvision \
    --partition noniid-labeldir \
    --dir_alpha 0.6 \
    --num_clients 10 \
    --raw_data_dir ../rawdata/webvision1.0 \
    --raw_imagenet_dir ../rawdata/imagenet_data/ILSVRC2012 \
    --data_dir ../fedNLLdata/webvision \
    --seed 1

python build_dataset_fed.py --dataset webvision \
    --partition noniid-quantity \
    --dir_alpha 0.1 \
    --num_clients 10 \
    --raw_data_dir ../rawdata/webvision1.0 \
    --raw_imagenet_dir ../rawdata/imagenet_data/ILSVRC2012 \
    --data_dir ../fedNLLdata/webvision \
    --seed 1

python build_dataset_fed.py --dataset webvision \
    --partition noniid-#label \
    --major_classes_num 20 \
    --num_clients 10 \
    --raw_data_dir ../rawdata/webvision1.0 \
    --raw_imagenet_dir ../rawdata/imagenet_data/ILSVRC2012 \
    --data_dir ../fedNLLdata/webvision \
    --seed 1
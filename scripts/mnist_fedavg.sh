#!/bin/bash
python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition iid --num_clients 10 --globalize --noise_mode clean --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition iid --num_clients 10 --globalize --noise_mode sym --noise_ratio 0.4 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition iid --num_clients 10 --noise_mode sym --max_noise_ratio 0.50 --min_noise_ratio 0.30 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition iid --num_clients 10 --globalize --noise_mode asym --noise_ratio 0.4 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition iid --num_clients 10 --noise_mode asym --max_noise_ratio 0.50 --min_noise_ratio 0.30 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode clean --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode sym --noise_ratio 0.4 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --noise_mode sym --max_noise_ratio 0.50 --min_noise_ratio 0.30 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode asym --noise_ratio 0.4 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-labeldir --num_clients 10 --dir_alpha 0.1 --noise_mode asym --max_noise_ratio 0.50 --min_noise_ratio 0.30 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-#label --num_clients 10 --major_classes_num 3 --globalize --noise_mode clean --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-#label --num_clients 10 --major_classes_num 3 --globalize --noise_mode sym --noise_ratio 0.4 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-#label --num_clients 10 --major_classes_num 3 --noise_mode sym --max_noise_ratio 0.50 --min_noise_ratio 0.30 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-#label --num_clients 10 --major_classes_num 3 --globalize --noise_mode asym --noise_ratio 0.4 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-#label --num_clients 10 --major_classes_num 3 --noise_mode asym --max_noise_ratio 0.50 --min_noise_ratio 0.30 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-quantity --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode clean --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-quantity --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode sym --noise_ratio 0.4 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-quantity --num_clients 10 --dir_alpha 0.1 --noise_mode sym --max_noise_ratio 0.50 --min_noise_ratio 0.30 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-quantity --num_clients 10 --dir_alpha 0.1 --globalize --noise_mode asym --noise_ratio 0.4 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

python fednoisy/algorithms/fedavg/main.py --dataset mnist --partition noniid-quantity --num_clients 10 --dir_alpha 0.1 --noise_mode asym --max_noise_ratio 0.50 --min_noise_ratio 0.30 --seed 1 --sample_ratio 1.0 --com_round 500 --epochs 5 --model SimpleCNN --momentum 0.9 --lr 0.01 --weight_decay 0.0005 --data_dir ../fedNLLdata/mnist --out_dir ../Fed-Noisy-checkpoint/mnist/

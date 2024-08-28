import torch
import argparse
import sys
import os
from typing import Dict, Tuple, List, Optional

from fednoisy.data.NLLData import functional as nllF


def read_singlenll_args():
    parser = argparse.ArgumentParser(description='Federated Noisy Labels Preparation')

    # ==== Pipeline args ====

    parser.add_argument(
        '--num_clients',
        default=10,
        type=int,
        help="Number for clients in federated setting.",
    )
    parser.add_argument("--client_id", default=0, type=int)
    parser.add_argument("--model", type=str, default='ResNet18')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.5)
    # parser.add_argument("--lr_decay_per_round", type=float, default=1)

    # ==== FedNLL data args ====
    parser.add_argument(
        '--centralized',
        default=False,
        help="Centralized setting or federated setting. True for centralized "
        "setting, while False for federated setting.",
    )
    # ----Federated Partition----
    parser.add_argument(
        '--partition',
        default='iid',
        type=str,
        choices=['iid', 'noniid-#label', 'noniid-labeldir', 'noniid-quantity'],
        help="Data partition scheme for federated setting.",
    )

    parser.add_argument(
        '--dir_alpha',
        default=0.1,
        type=float,
        help="Parameter for Dirichlet distribution.",
    )
    parser.add_argument(
        '--major_classes_num',
        default=2,
        type=int,
        help="Major class number for 'noniid-#label' partition.",
    )
    parser.add_argument(
        '--min_require_size',
        default=10,
        type=int,
        help="Minimum sample size for each client.",
    )

    # ----Noise setting options----
    parser.add_argument(
        '--noise_mode',
        default=None,
        type=str,
        choices=['clean', 'sym', 'asym'],
        help="Noise type for centralized setting: 'sym' for symmetric noise; "
        "'asym' for asymmetric noise; 'real' for real-world noise. Only works "
        "if --centralized=True.",
    )
    parser.add_argument(
        '--globalize',
        action='store_true',
        help="Federated noisy label setting, globalized noise or localized noise.",
    )

    parser.add_argument(
        '--noise_ratio',
        default=0.0,
        type=float,
        help="Noise ratio for symmetric noise or asymmetric noise.",
    )
    parser.add_argument(
        '--min_noise_ratio',
        default=0.0,
        type=float,
        help="Minimum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is Flase",
    )
    parser.add_argument(
        '--max_noise_ratio',
        default=1.0,
        type=float,
        help="Maximum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is Flase",
    )

    # ----Path options----
    parser.add_argument(
        '--dataset',
        default='cifar10',
        type=str,
        choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'clothing1m', 'webvision'],
        help="Dataset for experiment. Current support: ['mnist', 'cifar10', "
        "'cifar100', 'svhn', 'clothing1m', 'webvision']",
    )
    parser.add_argument(
        '--raw_data_dir',
        default='../data',
        type=str,
        help="Directory for raw dataset download",
    )
    parser.add_argument(
        '--data_dir',
        default='../noisy_label_data',
        type=str,
        help="Directory to save the dataset with noisy labels.",
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        default='../checkponit/',
        help="Checkpoint path for log files and report files.",
    )

    # ----Miscs options----
    parser.add_argument(
        "--save_best", action='store_true', help="Whether to save the best model."
    )
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument("--device", default='cuda:0', type=str)

    args = parser.parse_args()
    return args

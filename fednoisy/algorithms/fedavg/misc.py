import torch
import argparse
import sys
import os
from typing import Dict, Tuple, List, Optional

from fednoisy.data.NLLData import functional as nllF


def read_fednll_args():
    parser = argparse.ArgumentParser(description="Federated Noisy Labels Preparation")

    # ==== Pipeline args ====

    parser.add_argument(
        "--num_clients",
        default=10,
        type=int,
        help="Number for clients in federated setting.",
    )
    parser.add_argument("--com_round", type=int, default=3)
    parser.add_argument(
        "--model",
        type=str,
        default="ResNet18",
        help="Currently only support 'TwoLayerLinear', 'Cifar10Net', 'SimpleCNN',  'LeNet', 'VGG11', 'VGG13', 'VGG16', 'VGG19', 'ToyModel', 'PreResNet18', 'ResNet18', 'ResNet20', 'WRN28_10', 'WRN40_2', 'ResNet32' and 'ResNet34'.",
    )
    parser.add_argument("--sample_ratio", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)

    # ==== FedNLL data args ====
    parser.add_argument(
        "--centralized",
        default=False,
        help="Centralized setting or federated setting. True for centralized "
        "setting, while False for federated setting.",
    )
    parser.add_argument(
        "--preload", action="store_true", help="Whether to preload dataset into memory."
    )
    # ----Federated Partition----
    parser.add_argument(
        "--partition",
        default="iid",
        type=str,
        choices=["iid", "noniid-#label", "noniid-labeldir", "noniid-quantity"],
        help="Data partition scheme for federated setting.",
    )

    parser.add_argument(
        "--dir_alpha",
        default=0.1,
        type=float,
        help="Parameter for Dirichlet distribution.",
    )
    parser.add_argument(
        "--major_classes_num",
        default=2,
        type=int,
        help="Major class number for 'noniid-#label' partition.",
    )
    parser.add_argument(
        "--min_require_size",
        default=10,
        type=int,
        help="Minimum sample size for each client.",
    )

    # ----Noise setting options----
    parser.add_argument(
        "--noise_mode",
        default=None,
        type=str,
        choices=["clean", "sym", "asym", "real"],
        help="Noise type for centralized setting: 'sym' for symmetric noise; "
        "'asym' for asymmetric noise; 'real' for real-world noise. ",
    )
    parser.add_argument(
        "--globalize",
        action="store_true",
        help="Federated noisy label setting, globalized noise or localized noise.",
    )

    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="Noise ratio for symmetric noise or asymmetric noise.",
    )
    parser.add_argument(
        "--min_noise_ratio",
        default=0.0,
        type=float,
        help="Minimum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is Flase",
    )
    parser.add_argument(
        "--max_noise_ratio",
        default=1.0,
        type=float,
        help="Maximum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is Flase",
    )
    parser.add_argument(
        "--num_samples",
        default=32 * 2 * 1000,
        type=int,
        help="Number of samples used for Clothing1M training. Defaults as 64000.",
    )
    parser.add_argument(
        "--feature_dim",
        default=20,
        type=int,
        help="Feature dimension for synthetic dataset",
    )

    # ----Robust Loss Function options----
    parser.add_argument(
        "--criterion", type=str, default="ce"
    )  # for robust loss function
    parser.add_argument(
        "--sce_alpha",
        type=float,
        default=0.1,
        help="Symmetric cross entropy loss: alpha * CE + beta * RCE",
    )
    parser.add_argument(
        "--sce_beta",
        type=float,
        default=1.0,
        help="Symmetric cross entropy loss: alpha * CE + beta * RCE",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=1.0,
        help="scale parameter for loss, for example, scale * RCE, scale * NCE, scale * normalizer * RCE.",
    )
    parser.add_argument(
        "--gce_q",
        type=float,
        default=0.7,
        help="q parametor for Generalized-Cross-Entropy, Normalized-Generalized-Cross-Entropy.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=None,
        help="alpha parameter for Focal loss and Normalzied Focal loss.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help="gamma parameter for Focal loss and Normalzied Focal loss.",
    )

    # ----Mixup options----
    parser.add_argument("--mixup", action="store_true", help="Whether to use mixup.")
    parser.add_argument(
        "--mixup_alpha", type=float, default=1.0, help="Hyperparameter alpha for mixup."
    )

    # ----Co-teaching options----
    parser.add_argument(
        "--coteaching", action="store_true", help="Whether to use co-teahcing."
    )
    parser.add_argument(
        "--coteaching_forget_rate",
        type=float,
        default=None,
        help="Forget rate for co-teaching.",
    )
    parser.add_argument(
        "--coteaching_exponent",
        type=float,
        default=1,
        help="exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.",
    )
    parser.add_argument(
        "--coteaching_num_gradual",
        type=int,
        default=25,
        help="how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.",
    )

    # ----Dynamic Bootstrapping options----
    parser.add_argument(
        "--dynboot",
        action="store_true",
        help="Whether to use Dynamic Bootstrapping. Original paper is 'Unsupervised Label Noise Modeling and Loss Correction'.",
    )
    parser.add_argument(
        "--dynboot_mixup",
        type=str,
        default="static",
        choices=["static", "dynamic"],
        help="Dynamic Bootstrapping: Type of bootstrapping. Available: 'static' (as in the paper, default), 'dynamic' (BMM to mix the smaples, will use decreasing softmax). Default: 'static'",
    )
    # parser.add_argument("--debug", action="store_true")
    # parser.add_argument(
    #     "--dynboot_M",
    #     nargs="+",
    #     type=int,
    #     default=[167, 417],
    #     help="Milestones for the LR sheduler, default 100 250",
    # )
    parser.add_argument(
        "--dynboot_alpha",
        type=float,
        default=32,
        help="Dynamic Bootstrapping: alpha parameter for the mixup distribution, default: 32",
    )
    parser.add_argument(
        "--dynboot_bootbeta",
        type=str,
        default="hard",
        choices=[None, "hard", "soft"],
        help="Dynamic Bootstrapping: Type of Bootstrapping guided with the BMM. Available: \
                        None (deactivated)(default), 'hard' (Hard bootstrapping), 'soft' (Soft bootstrapping), default: 'hard'",
    )
    parser.add_argument(
        "--dynboot_reg",
        type=float,
        default=0.0,
        help="Dynamic Bootstrapping: Parameter of the regularization term, default: 0.",
    )

    # ----Path options----
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=[
            "synthetic",
            "mnist",
            "cifar10",
            "cifar100",
            "svhn",
            "clothing1m",
            "webvision",
        ],
        help="Dataset for experiment. Current support: ['synthetic', 'mnist', 'cifar10', "
        "'cifar100', 'svhn', 'clothing1m', 'webvision']",
    )
    parser.add_argument(
        "--data_dir",
        default="../noisy_label_data",
        type=str,
        help="Directory to save the dataset with noisy labels.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="../checkponit/",
        help="Checkpoint path for log files and report files.",
    )

    # ----Miscs options----
    parser.add_argument("--proj_name", default="FedNoisy-debug", type=str)
    parser.add_argument(
        "--save_best", action="store_true", help="Whether to save the best model."
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed")

    args = parser.parse_args()

    # default setting
    args.personalize = False
    return args

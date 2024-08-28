"""
Prepare dataset for noisy label setting, including centralized setting and federated setting.

Centralized setting:
- Symmetric noise
- Asymmetric noise
- Real-world noise

Federated setting:
- Globalized noise
    - IID
        - clean
        - sym
        - asym
    - noniid-#label
        - clean
        - sym
        - asym
    - noniid-labeldir
        - clean
        - sym
        - asym
    - noniid-quantity
        - clean
        - sym
        - asym
- Localized noise
    - IID
        - sym
        - asym
    - noniid-#label
        - sym
        - asym
    - noniid-labeldir
        - sym
        - asym
    - noniid-quantity
        - sym
        - asym
- Real-world noise
  - Data partition
    - IID
    - Non-IID-xxx
    - Non-IID-xxx
    - Non-IID-xxx
"""

import argparse

# from progress.bar import Bar as Bar

from fednoisy.data.NLLData.CentrNLL.cifar import CentrNLLCIFAR10, CentrNLLCIFAR100
from fednoisy.data.NLLData.CentrNLL.clothing1m import CentrNLLClothing1M
from fednoisy.data.NLLData.CentrNLL.webvision import CentrNLLWebVision
from fednoisy.data.NLLData.BaseNLL.cifar import NLLCIFAR100
from fednoisy.data.NLLData.FedNLL import (
    FedNLLCIFAR10,
    FedNLLCIFAR100,
    FedNLLMNIST,
    FedNLLSVHN,
    FedNLLClothing1M,
    FedNLLWebVision,
    FedNLLSynthetic,
)
from fednoisy.data.NLLData import functional as nllF


def read_args():
    parser = argparse.ArgumentParser(description="Federated Noisy Labels Preparation")
    parser.add_argument(
        "--centralized",
        default=False,
        help="Centralized setting or federated setting. True for centralized "
        "setting, while False for federated setting.",
    )
    # ----Federated Partition----
    parser.add_argument(
        "--partition",
        default="iid",
        type=str,
        choices=[
            "iid",
            "noniid",
            "noniid-#label",
            "noniid-labeldir",
            "noniid-quantity",
        ],
        help="Data partition scheme for federated setting.",
    )
    parser.add_argument(
        "--personalize",
        action="store_true",
        help="Whether use personalized local test set for each client. If True, then each client's class ratio of local test set is same as the training set",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="whether use balance partition for Synthetic dataset.",
    )
    parser.add_argument(
        "--num_clients",
        default=10,
        type=int,
        help="Number for clients in federated setting.",
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
        "'asym' for asymmetric noise; 'real' for real-world noise. Only works "
        "if --centralized=True.",
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
        help="Number of samples used for Clothing1M/Synthetic data training. Defaults as 64000.",
    )

    parser.add_argument(
        "--num_test_samples",
        default=1000,
        type=int,
        help="Number of test samples for synthetic dataset.",
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=100,
        help="Feature dimension for synthetic dataset.",
    )
    parser.add_argument(
        "--use_bias",
        action="store_true",
        help="Whether to use bias in synthetic data generation. If True, Y = Xw + b + ε; otherwise Y = Xw + ε.",
    )

    # ----Dataset path options----
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=[
            "mnist",
            "cifar10",
            "cifar100",
            "svhn",
            "clothing1m",
            "webvision",
            "synthetic",
        ],
        help="Dataset for experiment. Current support: ['mnist', 'cifar10', "
        "'cifar100', 'svhn', 'clothing1m', 'webvision', 'synthetic]",
    )
    parser.add_argument(
        "--raw_data_dir",
        default="../data",
        type=str,
        help="Directory for raw dataset download",
    )
    parser.add_argument(
        "--raw_imagenet_dir",
        default="../rawdata/imagenet",
        type=str,
        help="Directory for raw dataset download",
    )
    parser.add_argument(
        "--data_dir",
        default="../noisy_label_data",
        type=str,
        help="Directory to save the dataset with noisy labels.",
    )

    # ----Miscs options----
    parser.add_argument("--seed", default=0, type=int, help="Random seed")

    args = parser.parse_args()
    return args


# def read_args_centr():
#     parser = argparse.ArgumentParser(
#         description='Centralized Noisy Labels data preparation'
#     )
#     # ----Noise setting options----
#     parser.add_argument(
#         '--noise_mode',
#         default=None,
#         type=str,
#         choices=['clean', 'sym', 'asym', 'real'],
#         help="Noise type for centralized setting: 'clean' for clean dataset; 'sym' for symmetric noise; 'asym' for asymmetric noise; 'real' for real-world noise.",
#     )

#     parser.add_argument(
#         '--noise_ratio',
#         default=0.1,
#         type=float,
#         help="Noise ratio for symmetric noise or asymmetric noise.",
#     )

#     # ----Dataset path options----
#     parser.add_argument(
#         '--dataset',
#         default='cifar10',
#         type=str,
#         choices=['mnist', 'cifar10', 'cifar100', 'svhn', 'clothing1m', 'webvision'],
#         help="Dataset for experiment. Current support: ['mnist', 'cifar10', "
#         "'cifar100', 'svhn', 'clothing1m', 'webvision']",
#     )
#     parser.add_argument(
#         '--raw_data_dir',
#         default='../rawdata/cifar10',
#         type=str,
#         help="Directory for raw dataset download",
#     )
#     parser.add_argument(
#         '--raw_imagenet_dir',
#         default='../rawdata/imagenet',
#         type=str,
#         help="Directory for raw dataset download",
#     )
#     parser.add_argument(
#         '--data_dir',
#         default='../centrNLLdata/cifar10',
#         type=str,
#         help="Directory to load the prepared dataset and noisy label file.",
#     )

#     # ----Miscs options----
#     parser.add_argument('--seed', default=0, type=int, help='Random seed')

#     args = parser.parse_args()
#     return args


if __name__ == "__main__":
    args = read_args()
    if args.dataset == "cifar10":
        nll_cifar10 = FedNLLCIFAR10(
            globalize=args.globalize,
            partition=args.partition,
            num_clients=args.num_clients,
            dir_alpha=args.dir_alpha,
            major_classes_num=args.major_classes_num,
            noise_mode=args.noise_mode,
            noise_ratio=args.noise_ratio,
            min_noise_ratio=args.min_noise_ratio,
            max_noise_ratio=args.max_noise_ratio,
            root_dir=args.raw_data_dir,
            out_dir=args.data_dir,
            personalize=args.personalize,
        )
        nll_cifar10.create_nll_scene(seed=args.seed)
        nll_cifar10.save_nll_scene()

    elif args.dataset == "cifar100":
        nll_cifar100 = FedNLLCIFAR100(
            globalize=args.globalize,
            partition=args.partition,
            num_clients=args.num_clients,
            dir_alpha=args.dir_alpha,
            major_classes_num=args.major_classes_num,
            noise_mode=args.noise_mode,
            noise_ratio=args.noise_ratio,
            min_noise_ratio=args.min_noise_ratio,
            max_noise_ratio=args.max_noise_ratio,
            root_dir=args.raw_data_dir,
            out_dir=args.data_dir,
            personalize=args.personalize,
        )
        nll_cifar100.create_nll_scene(seed=args.seed)
        nll_cifar100.save_nll_scene()

    elif args.dataset == "mnist":
        nll_mnist = FedNLLMNIST(
            globalize=args.globalize,
            partition=args.partition,
            num_clients=args.num_clients,
            dir_alpha=args.dir_alpha,
            major_classes_num=args.major_classes_num,
            noise_mode=args.noise_mode,
            noise_ratio=args.noise_ratio,
            min_noise_ratio=args.min_noise_ratio,
            max_noise_ratio=args.max_noise_ratio,
            root_dir=args.raw_data_dir,
            out_dir=args.data_dir,
            personalize=args.personalize,
        )
        nll_mnist.create_nll_scene(seed=args.seed)
        nll_mnist.save_nll_scene()

    elif args.dataset == "svhn":
        nll_svhn = FedNLLSVHN(
            globalize=args.globalize,
            partition=args.partition,
            num_clients=args.num_clients,
            dir_alpha=args.dir_alpha,
            major_classes_num=args.major_classes_num,
            noise_mode=args.noise_mode,
            noise_ratio=args.noise_ratio,
            min_noise_ratio=args.min_noise_ratio,
            max_noise_ratio=args.max_noise_ratio,
            root_dir=args.raw_data_dir,
            out_dir=args.data_dir,
            personalize=args.personalize,
        )
        nll_svhn.create_nll_scene(seed=args.seed)
        nll_svhn.save_nll_scene()

    elif args.dataset == "clothing1m":
        args.noise_mode = "real"
        args.globalize = True
        args.noise_ratio = 0.39
        nll_clothing1m = FedNLLClothing1M(
            root_dir=args.raw_data_dir,
            out_dir=args.data_dir,
            partition=args.partition,
            num_clients=args.num_clients,
            dir_alpha=args.dir_alpha,
            major_classes_num=args.major_classes_num,
            num_samples=args.num_samples,
        )
        nll_clothing1m.create_nll_scene(seed=args.seed)
        nll_clothing1m.save_nll_scene()

    elif args.dataset == "webvision":
        args.noise_mode = "real"
        args.globalize = True
        args.noise_ratio = 0.20
        nll_webvision = FedNLLWebVision(
            root_dir=args.raw_data_dir,
            imagenet_root_dir=args.raw_imagenet_dir,
            out_dir=args.data_dir,
            partition=args.partition,
            num_clients=args.num_clients,
            dir_alpha=args.dir_alpha,
            major_classes_num=args.major_classes_num,
        )
        nll_webvision.create_nll_scene(seed=args.seed)
        nll_webvision.save_nll_scene()

    elif args.dataset == "synthetic":
        nll_synthetic = FedNLLSynthetic(
            out_dir=args.data_dir,
            num_clients=args.num_clients,
            init_mu=0,
            init_sigma=1,
            partition=args.partition,
            balance=args.balance,
            train_sample_num=args.num_samples,
            test_sample_num=args.num_test_samples,
            feature_dim=args.feature_dim,
            use_bias=args.use_bias,
            dir_alpha=args.dir_alpha,
        )
        args.init_mu = 0
        args.init_sigma = 1
        nll_synthetic.create_nll_scene(seed=args.seed)
        nll_synthetic.save_nll_scene()
        nll_name = nllF.FedNLL_name(**vars(args))
        print(f"{nll_name}")

    else:
        raise ValueError(f"dataset='{args.dataset}' is not supported!")

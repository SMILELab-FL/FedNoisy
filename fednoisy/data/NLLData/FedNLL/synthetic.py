import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchnet.meter import AUCMeter

from typing import Dict, Tuple, List, Optional

from fedlab.utils.dataset import functional as partF
from fedlab.utils.dataset.partition import VisionPartitioner

from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    CIFAR10_TRANSITION_MATRIX,
    TRANSITION_MATRIX,
    NORM_VALUES,
    TEST_TRANSFORM,
    TRAIN_TRANSFORM,
)
from fednoisy import visual
from fednoisy.data.NLLData.BaseNLL import NLLMNIST
from fednoisy.data.NLLData.FedNLL import FedNLLScene
from fednoisy.data.NLLData import functional as F
from fednoisy.data.NLLData.functional import NoisyDataset


class FedNLLSynthetic(FedNLLScene):
    centralized = False

    dataset_name = "synthetic"
    trainset_filename = ""
    testset_filename = f"{dataset_name}-testset.pt"

    def __init__(
        self,
        out_dir: str,
        num_clients: int,
        init_mu: float = 0,
        init_sigma: float = 2,
        partition: str = "iid",
        balance: bool = True,
        train_sample_num: int = 20000,
        test_sample_num: int = 4000,
        feature_dim: int = 100,
        use_bias: bool = True,
        dir_alpha: float = None,
        min_require_size=10,
    ) -> None:
        FedNLLScene.__init__(
            self,
            root_dir="",
            out_dir=out_dir,
            num_clients=num_clients,
            globalize=True,
            noise_mode="clean",
            partition="iid",
            dir_alpha=-1,
            major_classes_num=0,
            noise_ratio=0,
            min_noise_ratio=0,
            max_noise_ratio=0,
            min_require_size=min_require_size,
        )
        self.train_sample_num = train_sample_num
        self.test_sample_num = test_sample_num
        self.feature_dim = feature_dim
        self.use_bias = use_bias

        self.num_clients = num_clients
        self.balance = balance
        if partition not in ["noniid", "iid"]:
            raise ValueError(
                f"Data partition for {self.__class__.__name__} only supports 'noniid' and 'iid'. "
                f"{partition} is not supported."
            )

        self.partition = partition

        self.init_mu = init_mu
        self.init_sigma = init_sigma
        self.mu = []
        self.sigma = []
        # designed Non-IID pattern: different local normal distribution
        for cid in range(self.num_clients):
            if self.partition == "iid":
                mu, sigma = init_mu, init_sigma
            else:
                mu = (init_mu + float(cid)) % 5  # init_mu + float(cid)
                sigma = init_sigma  # init_sigma * float(cid % 5 + 1)

            self.mu.append(mu)
            self.sigma.append(sigma)

        self.dir_alpha = dir_alpha
        self.partitioner = None

    def _gen_linear_model(self):
        self.weights = np.floor(np.random.randn(self.feature_dim), dtype=np.float32)
        if self.use_bias:
            self.bias = np.floor(np.random.randn(1), dtype=np.float32)
        else:
            self.bias = 0.0

    def _gen_testset(self):
        if self.partition == "iid":
            test_data = np.random.normal(
                self.init_mu,
                self.init_sigma,
                size=(self.test_sample_num, self.feature_dim),
            ).astype(np.float32)

        else:
            test_sample_num_per_client = int(self.test_sample_num / self.num_clients)
            test_data = []
            for cid in range(self.num_clients):
                mu = self.mu[cid]
                sigma = self.sigma[cid]
                partial_test_data = np.random.normal(
                    mu,
                    sigma,
                    size=(test_sample_num_per_client, self.feature_dim),
                ).astype(np.float32)
                test_data.append(partial_test_data)
            test_data = np.concatenate(test_data, axis=0)

        self.test_data = test_data
        self.test_labels = np.matmul(test_data, self.weights) + self.bias  # Y = XW + b

    def _gen_trainset(self):
        if self.balance:
            client_sample_nums = partF.balance_split(
                self.num_clients, self.train_sample_num
            )
        else:
            client_sample_nums = partF.dirichlet_unbalance_split(
                self.num_clients, self.train_sample_num, self.dir_alpha
            )

        data_dict = {}
        labels_dict = {}
        for cid in range(self.num_clients):
            mu = self.mu[cid]
            sigma = self.sigma[cid]
            data = np.random.normal(
                mu,
                sigma,
                size=(client_sample_nums[cid], self.feature_dim),
            ).astype(np.float32)
            labels = (np.matmul(data, self.weights) + self.bias).astype(np.float32)
            # Y = XW + b + epsilon
            data_dict[cid] = data
            labels_dict[cid] = labels

        self.data_dict = data_dict
        self.labels_dict = labels_dict

    def create_nll_scene(self, seed: int = 0):
        self.setup_seed(seed)
        self.nll_scene_filename = f"{self}_seed_{seed}_setting.pt"
        self.nll_scene_file_path = os.path.join(self.out_dir, self.nll_scene_filename)
        self.nll_scene_folder = os.path.join(self.out_dir, f"{self}_seed_{seed}")
        self._gen_linear_model()
        self._gen_testset()
        self._gen_trainset()
        print(f"Synthetic dataset using linear model Y = XW  is generated")

    def save_nll_scene(self):
        fednll_scene = {
            "dataset": self.dataset_name,
            "weights": self.weights,
            "use_bias": self.use_bias,
            "bias": self.bias,
            "mu": self.mu,
            "sigma": self.sigma,
            "partition": self.partition,
            "balance": self.balance,
            "dir_alpha": self.dir_alpha,
            "num_clients": self.num_clients,
            "min_require_size": self.min_require_size,
            "globalize": True,
            "noise_mode": "clean",
            "true_noise_ratio": 0,
            "noise_ratio": 0,
            "noisy_labels": None,
        }

        torch.save(fednll_scene, self.nll_scene_file_path)
        print(
            f"Federated Learning scene saved to {self.nll_scene_file_path}, with keys "
            f"{list(fednll_scene.keys())}"
        )

        os.mkdir(self.nll_scene_folder)
        # # train split save to local
        for cid in range(self.num_clients):
            client_dataset = NoisyDataset(
                data=self.data_dict[cid],
                labels=self.labels_dict[cid],
                noisy_labels=self.labels_dict[cid],
                train=True,
                transform=None,
            )
            path = os.path.join(self.nll_scene_folder, f"train-data{cid}.pkl")
            torch.save(client_dataset, path)
            print(f"Client {cid} local train set saved to {path}")

        # test split save to local
        test_dataset = NoisyDataset(
            data=self.test_data,
            labels=self.test_labels,
            train=False,
            transform=None,
        )
        path = os.path.join(self.nll_scene_folder, "test-data.pkl")
        torch.save(test_dataset, path)
        print(f"Test set saved to {path}")

    def _perform_partition(self):
        print(
            f"{self.__class__.__name__} automatically partition the dataset during data generation."
        )

    def _gen_noisy_labels(self, client_dict=None):
        print(f"{self.__class__.__name__} currently does not support noisy labels.")

    @property
    def partition_setting(self):
        partition_param = f"{self.init_mu:.2f}_{self.init_sigma:.2f}"
        balance_param = f"balance={self.balance}"
        if not self.balance:
            balance_param += f"_{self.dir_alpha:.2f}_{self.min_require_size}"

        return f"{self.num_clients}_{self.partition}_{partition_param}_{balance_param}"

    @property
    def noise_setting(self):
        print(f"{self.__class__.__name__} does not support noisy label setting.")

    @property
    def synthetic_setting(self):
        synthetic_name = f"{self.feature_dim}_{self.use_bias}_{self.train_sample_num}_{self.test_sample_num}"
        return synthetic_name

    @property
    def setting(self):
        return f"{self.partition_setting}_{self.synthetic_setting}"

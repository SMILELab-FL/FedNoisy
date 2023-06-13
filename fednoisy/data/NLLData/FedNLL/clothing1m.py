import random
import numpy as np
import copy
from PIL import Image
import json
import os
import random
import warnings

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
    TRANSITION_MATRIX,
    NORM_VALUES,
    TEST_TRANSFORM,
    TRAIN_TRANSFORM,
)
from fednoisy import visual
from fednoisy.data import CLASS_NUM, TRAIN_SAMPLE_NUM, TEST_SAMPLE_NUM
from fednoisy.data.NLLData.BaseNLL import NLLClothing1M
from fednoisy.data.NLLData.FedNLL import FedNLLScene
from fednoisy.data.NLLData import functional as F
from fednoisy.data.NLLData.functional import NoisyDataset


class Clothing1MPartitioner(VisionPartitioner):
    """Clothing1M data partitioner based on :class:`VisionPartitioner`.
    This is a subclass of the :class:`VisionPartitioner`. For details, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
    """

    num_classes = 14


class FedNLLClothing1M(NLLClothing1M, FedNLLScene):
    centralized = False
    """_summary_

    Args:
        root_dir (str): _description_
        out_dir (str): _description_
        num_clients (int): _description_
        partition (str, optional): _description_. Defaults to "iid".
        dir_alpha (float, optional): _description_. Defaults to 0.6.
        major_classes_num (int, optional): _description_. Defaults to -1.
        min_require_size (Optional[int], optional): _description_. Defaults to 10.
        num_samples (int, optional): Total sample number used for training. The maximum ``num_samples`` can be 265664, with 18976 samples for each class. Defaults to 32*2*1000, as set in DivideMix.
    """

    def __init__(
        self,
        root_dir: str,
        out_dir: str,
        num_clients: int,
        partition: str = "iid",
        dir_alpha: float = 0.6,
        major_classes_num: int = -1,
        min_require_size: Optional[int] = 10,
        num_samples: int = 32 * 2 * 1000,
    ):
        NLLClothing1M.__init__(self, root_dir, out_dir)
        noise_ratio = 0.39
        globalize = True
        noise_mode = "real"
        self.num_samples = num_samples
        FedNLLScene.__init__(
            self,
            root_dir,
            out_dir,
            num_clients,
            globalize,
            noise_mode,
            partition,
            dir_alpha,
            major_classes_num,
            noise_ratio,
            min_require_size=min_require_size,
            partitioner=Clothing1MPartitioner,
        )

    def create_nll_scene(self, seed: int = 0):
        self.setup_seed(seed)
        self.nll_scene_filename = f"{self}_seed_{seed}_setting.pt"
        self.nll_scene_file_path = os.path.join(self.out_dir, self.nll_scene_filename)
        self.nll_scene_folder = os.path.join(self.out_dir, f"{self}_seed_{seed}")

        # sample class-balanced train data from overall trainset
        total_sample_num = len(self.total_train_labels)
        sample_idxs = list(range(total_sample_num))
        random.shuffle(sample_idxs)
        sample_num_each_class = np.zeros(self.num_classes)
        selected_train_data = []
        selected_train_labels = []
        cur_sample_num = 0
        for idx in sample_idxs:
            label = self.total_train_labels[idx]
            if (
                sample_num_each_class[label] < (self.num_samples / self.num_classes)
                and cur_sample_num < self.num_samples
            ):
                img_path = self.total_train_data[idx]
                selected_train_data.append(img_path)
                selected_train_labels.append(label)
                sample_num_each_class[label] += 1
                cur_sample_num += 1

        self.train_data = selected_train_data  # class-balanced trainset
        self.train_labels = selected_train_labels

        client_dict = self._perform_partition()  # contain index of samples
        for cid in client_dict:
            client_dict[cid] = client_dict[cid].tolist()  # change array([...]) to [...]
        self.client_dict = client_dict
        self.data_dict = F.split_data(client_dict, self.train_data)
        self.labels_dict = F.split_data(client_dict, self.train_labels)
        self.noisy_labels_dict = copy.deepcopy(self.labels_dict)
        self.true_noise_ratio = [None for _ in range(self.num_clients)]
        print(f"Clothing1M trainset generated: {self.num_samples} samples in total.")

    def save_nll_scene(self):
        fednll_scene = {
            "dataset": self.dataset_name,
            "train_data": self.train_data,
            "train_labels": self.train_labels,
            "client_dict": self.client_dict,
            "partition": self.partition,
            "dir_alpha": self.dir_alpha,
            "num_clients": self.num_clients,
            "major_classes_num": self.major_classes_num,
            "min_require_size": self.min_require_size,
            "globalize": self.globalize,
            "noise_mode": self.noise_mode,
            "true_noise_ratio": self.true_noise_ratio,
            "noise_ratio": self.noise_ratio,
            "noisy_labels": self.noisy_labels_dict,
        }

        torch.save(fednll_scene, self.nll_scene_file_path)
        print(
            f"Federated Noisy Label Learning scene saved to {self.nll_scene_file_path}, with keys "
            f"{list(fednll_scene.keys())}"
        )

        os.mkdir(self.nll_scene_folder)
        # train split save to local
        train_transform = TRAIN_TRANSFORM[self.dataset_name]
        for cid in range(self.num_clients):
            client_dataset = NoisyDataset(
                data=self.data_dict[cid],
                labels=self.labels_dict[cid],
                noisy_labels=self.noisy_labels_dict[cid],
                train=True,
                transform=train_transform,
                folder_data=True,
            )
            path = os.path.join(self.nll_scene_folder, f"train-data{cid}.pkl")
            torch.save(client_dataset, path)
            print(f"Client {cid} local train set saved to {path}")

        # test split save to local
        test_transform = TEST_TRANSFORM[self.dataset_name]
        test_dataset = NoisyDataset(
            data=self.test_data,
            labels=self.test_labels,
            train=False,
            transform=test_transform,
            folder_data=True,
        )
        path = os.path.join(self.nll_scene_folder, "test-data.pkl")
        torch.save(test_dataset, path)
        print(f"Test set saved to {path}")

    def _gen_noisy_labels(self, client_dict):
        warnings.warn(
            f"warning: {self.dataset_name} has already contained real noisy labels. No need for noisy label generation."
        )

    @property
    def partition_setting(self):
        if self.partition == "noniid-#label":
            partition_param = f"{self.major_classes_num}"
        elif self.partition == "noniid-quantity":
            partition_param = f"{self.dir_alpha}"
        elif self.partition == "noniid-labeldir":
            partition_param = f"{self.dir_alpha:.2f}_{self.min_require_size}"
        else:
            # IID
            partition_param = ""
        return f"{self.num_clients}_{self.partition}_{partition_param}"

    @property
    def noise_setting(self):
        noise_param = f"real_{self.num_samples}"
        return noise_param

    @property
    def setting(self):
        return f"{self.partition_setting}_{self.noise_setting}"

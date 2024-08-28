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
from fednoisy.data.NLLData.BaseNLL import NLLWebVision
from fednoisy.data.NLLData.FedNLL import FedNLLScene
from fednoisy.data.NLLData import functional as F
from fednoisy.data.NLLData.functional import NoisyDataset


class WebVisionPartitioner(VisionPartitioner):
    """WebVision data partitioner based on :class:`VisionPartitioner`.
    This is a subclass of the :class:`VisionPartitioner`. For details, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
    """

    num_classes = 50


class FedNLLWebVision(NLLWebVision, FedNLLScene):
    centralized = False

    def __init__(
        self,
        root_dir: str,
        imagenet_root_dir: str,
        out_dir: str,
        num_clients: int,
        partition: str = "iid",
        dir_alpha: float = 0.6,
        major_classes_num: int = -1,
        min_require_size: Optional[int] = 10,
        num_classes: int = 50,
    ):
        NLLWebVision.__init__(
            self,
            root_dir=root_dir,
            imagenet_root_dir=imagenet_root_dir,
            out_dir=out_dir,
            num_classes=num_classes,
        )
        noise_ratio = 0.20
        globalize = True
        noise_mode = "real"
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
            partitioner=WebVisionPartitioner,
        )

    def create_nll_scene(self, seed: int = 0):
        self.setup_seed(seed)
        self.nll_scene_filename = f"{self}_seed_{seed}_setting.pt"
        self.nll_scene_file_path = os.path.join(self.out_dir, self.nll_scene_filename)
        self.nll_scene_folder = os.path.join(self.out_dir, f"{self}_seed_{seed}")

        client_dict = self._perform_partition()  # contain index of samples
        for cid in client_dict:
            client_dict[cid] = client_dict[cid].tolist()  # change array([...]) to [...]
        self.client_dict = client_dict
        self.data_dict = F.split_data(client_dict, self.train_data)
        self.labels_dict = F.split_data(client_dict, self.train_labels)
        self.noisy_labels_dict = copy.deepcopy(self.labels_dict)
        self.true_noise_ratio = [None for _ in range(self.num_clients)]
        print(f"WebVision1.0 trainset generated.")

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
        val_transform = TEST_TRANSFORM[self.dataset_name]
        val_dataset = NoisyDataset(
            data=self.val_data,
            labels=self.val_labels,
            train=False,
            transform=val_transform,
            folder_data=True,
        )
        path = os.path.join(self.nll_scene_folder, "test-data.pkl")
        torch.save(val_dataset, path)
        print(f"Valset saved to {path}")

        # ImageNet val split save to local
        imagenet_val_dataset = NoisyDataset(
            data=self.imagenet_val_data,
            labels=self.imagenet_val_labels,
            train=False,
            transform=val_transform,
            folder_data=True,
        )
        path = os.path.join(self.nll_scene_folder, "imagenet-test-data.pkl")
        torch.save(imagenet_val_dataset, path)
        print(f"ImageNet valset saved to {path}")

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
        noise_param = f"real"
        return noise_param

    @property
    def setting(self):
        return f"{self.partition_setting}_{self.noise_setting}"

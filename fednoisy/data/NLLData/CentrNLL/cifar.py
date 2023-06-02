import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

from typing import Dict, Tuple, List, Optional

from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
    TRANSITION_MATRIX,
    CIFAR10_TRANSITION_MATRIX,
)
from fednoisy import visual
from fednoisy.data.NLLData.BaseNLL import NLLCIFAR10
from fednoisy.data.NLLData.CentrNLL import CentrNLLScene
from fednoisy.data.NLLData import functional as F


class CentrNLLCIFAR10(NLLCIFAR10, CentrNLLScene):
    """
    Read raw train & test data from root_dir/cifar-10-batches-py, reformat image data into HWC np.array format,
    and generate noisy labels for train data, save them into local files:

    - Train file cifar10-trainset.pt content:
        {
        data: np.array[...],  # np.array images in HWC format
        labels: List[int],  # list of labels, label is in range of [0,9]
        class_to_idx: {class_name: class_label, ...}  # a dictionary mapping class_name to 0-9 class label
        classes: List[str],  # class names for 0-9 classes
        }

    - Test file cifar10-testset.pt content:
        {
        data: np.array[...],  # np.array images in HWC format
        labels: List[int],  # list of labels, label is in range of [0,9]
        class_to_idx: {class_name: class_label, ...}  # a dictionary mapping class_name to 0-9 class label
        classes: List[str],  # class names for 0-9 classes
        }

    - Noisy labels *.json file content:
       {
       noisy_labels: List[int],  # list of noisy labels
       noise_mode: str,  # noisy mode, 'sym'/'asym'/'clean'
       noise_ratio: float,  # noise ratio to generate the noise
       true_noise_ratio: float  # true noise ratio calculated using noisy_labels and train_labels
       }


    Args:
        root_dir (str): Root directory with downloaded CIFAR10 raw data files.
        noise_mode (str): Noise mode for centralized CIFAR10. COnly 'sym', 'asym' and 'clean' are supported.
        noise_ratio (float): Noise ratio that is in range of [0, 1].
        out_dir (str): Output directory to save processed trainset/testset and noisy label file.
        # seed (int) Random seed for noisy label generation.

    """

    centralized = True

    def __init__(
        self, root_dir: str, noise_mode: str, noise_ratio: float, out_dir: str
    ) -> None:
        NLLCIFAR10.__init__(self, root_dir, noise_mode, out_dir)
        self.noise_ratio = F.check_centr_noisy_setting(
            noise_mode, noise_ratio
        )  # check validation of noise setting


class CentrNLLCIFAR100(CentrNLLCIFAR10):
    dataset_name = 'cifar100'
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    num_classes = CLASS_NUM[dataset_name]
    train_sample_num = TRAIN_SAMPLE_NUM[dataset_name]
    test_sample_num = TEST_SAMPLE_NUM[dataset_name]
    transition = None
    trainset_filename = 'cifar100-trainset.pt'
    testset_filename = 'cifar100-testset.pt'

    def gen_noisy_labels(self, seed: int = 0) -> List[int]:
        """Generate  noisy labels for CIFAR-100.

        Args:
            seed (int): Random seed.

        Returns:
            List[int]: A list of generated labels in range of 0 to 9.

        """
        self.setup_seed(seed)
        self.noise_filename = f"{self}_seed_{self.seed}_noise.json"
        self.noise_file_path = os.path.join(self.out_dir, self.noise_filename)
        if not os.path.exists(self.noise_file_path):
            noisy_labels = F.generate_noisy_label_cifar100(
                labels=self.train_labels,
                noise_mode=self.noise_mode,
                noise_ratio=self.noise_ratio,
                seed=self.seed,
            )
            print(
                f"{self.dataset_name} noisy labels of {self}_seed_{self.seed} are generated."
            )
        else:
            print(
                f"{self.dataset_name} noisy labels of {self}_seed_{self.seed} are already generated, "
                f"loaded from {self.noise_file_path}."
            )
            with open(self.noise_file_path, 'r') as f:
                entry = json.load(f)
                noisy_labels = entry['noisy_labels']

        self.noisy_labels = noisy_labels
        self.true_noise_ratio = F.cal_true_noisy_ratio(self.train_labels, noisy_labels)
        print(f"{self} true_noise_ratio={self.true_noise_ratio}")
        return noisy_labels

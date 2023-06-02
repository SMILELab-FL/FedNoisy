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
from fednoisy.data.NLLData import functional as F
from fednoisy.data.NLLData.BaseNLL import NLLBase
from fednoisy.utils.misc import make_dirs


class NLLCIFAR10(NLLBase):
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
        noise_mode (str): Noise mode for centralized CIFAR10. Only 'sym', 'asym' and 'clean' are supported.
        noise_ratio (float): Noise ratio that is in range of [0, 1].
        out_dir: str, Output directory to save processed trainset/testset and noisy label file.
        seed (int) Random seed for noisy label generation.

    """

    dataset_name = "cifar10"
    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }
    num_classes = CLASS_NUM[dataset_name]
    train_sample_num = TRAIN_SAMPLE_NUM[dataset_name]
    test_sample_num = TEST_SAMPLE_NUM[dataset_name]
    transition = TRANSITION_MATRIX[dataset_name]
    trainset_filename = f"{dataset_name}-trainset.pt"
    testset_filename = f"{dataset_name}-testset.pt"
    # centralized = True

    def __init__(self, root_dir: str, noise_mode: str, out_dir: str) -> None:
        NLLBase.__init__(self, root_dir, noise_mode, out_dir)

        self._load_meta()
        self._load_testset()
        self._load_trainset()

    def _load_meta(self) -> None:
        # process meta dataset info
        meta_file_path = os.path.join(
            self.root_dir, self.base_folder, self.meta["filename"]
        )
        meta_info = F.unpickle(meta_file_path)
        self.classes = meta_info[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def _load_testset(self, save: bool = True) -> None:
        """

        Args:
            save (bool): Whether to save into local cifar10-testset.pt file. Default as ``True``.

        Returns:

        """
        # load from processed file, or from raw test data file
        if os.path.exists(self.testset_path):
            entry = torch.load(self.testset_path)
            test_data = entry["data"]
            test_labels = entry["labels"]
        else:
            # process raw test dataset
            # read from file, reshape and transbose data
            file_name = self.test_list[0][0]
            file_path = os.path.join(self.root_dir, self.base_folder, file_name)
            entry = F.unpickle(file_path)
            test_data = entry["data"]
            test_data = test_data.reshape((self.test_sample_num, 3, 32, 32))
            test_data = test_data.transpose((0, 2, 3, 1))  # convert to HWC
            if "labels" in entry:
                test_labels = entry["labels"]
            else:
                test_labels = entry["fine_labels"]

        self.test_data = test_data
        self.test_labels = test_labels
        print(f"{self.dataset_name} testset is loaded.")
        if save is True:
            self.save_testset()

    def _load_trainset(self, save: bool = True) -> None:
        """

        Args:
            save (bool): Whether to save into local cifar10-trainset.pt file. Default as ``True``.

        Returns:

        """
        # load from processed file, or from raw train data file
        if os.path.exists(self.trainset_path):
            entry = torch.load(self.trainset_path)
            train_data = entry["data"]
            train_labels = entry["labels"]
        else:
            train_data = []
            train_labels = []
            # process raw train dataset
            # read from files, concatenate batches, reshape and transpose data
            for file_name, _ in self.train_list:
                file_path = os.path.join(self.root_dir, self.base_folder, file_name)
                entry = F.unpickle(file_path)
                train_data.append(entry["data"])
                if "labels" in entry:
                    train_labels.extend(entry["labels"])
                else:
                    train_labels.extend(entry["fine_labels"])
            train_data = np.concatenate(
                train_data
            )  # concatenate raw train data from batch files
            train_data = train_data.reshape((self.train_sample_num, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))  # convert to HWC

        self.train_data = train_data
        self.train_labels = train_labels
        print(f"{self.dataset_name} trainset is loaded.")
        if save is True:
            self.save_trainset()


class NLLCIFAR100(NLLCIFAR10):
    dataset_name = "cifar100"
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
    transition = TRANSITION_MATRIX[dataset_name]
    trainset_filename = f"{dataset_name}-trainset.pt"
    testset_filename = f"{dataset_name}-testset.pt"

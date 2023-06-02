import random
import codecs
import numpy as np
from PIL import Image
import json
import os
import sys
import torch
from torchnet.meter import AUCMeter

from typing import Dict, Tuple, List, Optional

from fednoisy.data import (
    CLASS_NUM,
    TRAIN_SAMPLE_NUM,
    TEST_SAMPLE_NUM,
)
from fednoisy import visual
from fednoisy.data.NLLData import functional as F
from fednoisy.data.NLLData.BaseNLL import NLLBase
from fednoisy.utils.misc import make_dirs


class NLLSVHN(NLLBase):
    dataset_name = "svhn"
    split_list = {
        "train": [
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            "train_32x32.mat",
            "e26dedcc434d2e4c54c9b2d4a06d8373",
        ],
        "test": [
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            "test_32x32.mat",
            "eb5a983be6a315427106f1b164d9cef3",
        ],
        "extra": [
            "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
            "extra_32x32.mat",
            "a93ce644f1a588dc4d68dda5feec44a7",
        ],
    }

    num_classes = CLASS_NUM[dataset_name]
    train_sample_num = TRAIN_SAMPLE_NUM[dataset_name]
    test_sample_num = TEST_SAMPLE_NUM[dataset_name]
    classes = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ]
    trainset_filename = f"{dataset_name}-trainset.pt"
    testset_filename = f"{dataset_name}-testset.pt"
    # centralized = True

    def __init__(self, root_dir: str, noise_mode: str, out_dir: str) -> None:
        NLLBase.__init__(self, root_dir, noise_mode, out_dir)
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

        self._load_testset()
        self._load_trainset()

    def _load_testset(self, save: bool = True):
        if os.path.exists(self.testset_path):
            entry = torch.load(self.testset_path)
            test_data = entry["data"]
            test_labels = entry["labels"]
        else:
            import scipy.io as sio

            test_file_path = os.path.join(self.root_dir, self.split_list["test"][1])
            loaded_mat = sio.loadmat(test_file_path)
            test_data = loaded_mat["X"]
            test_labels = loaded_mat["y"].astype(np.int64).squeeze()
            np.place(test_labels, test_labels == 10, 0)
            test_labels = test_labels.tolist()
            test_data = np.transpose(test_data, (3, 2, 0, 1))
            test_data = np.transpose(test_data, (0, 2, 3, 1))  # (N, W, H, C)

        self.test_labels = test_labels
        self.test_data = test_data
        print(f"{self.dataset_name} testset is loaded.")
        if save is True:
            self.save_testset()

    def _load_trainset(self, save: bool = True):
        if os.path.exists(self.trainset_path):
            entry = torch.load(self.trainset_path)
            train_data = entry["data"]
            train_labels = entry["labels"]
        else:
            # process raw train dataset
            import scipy.io as sio

            train_file_path = os.path.join(self.root_dir, self.split_list["train"][1])
            loaded_mat = sio.loadmat(train_file_path)
            train_data = loaded_mat["X"]
            train_labels = loaded_mat["y"].astype(np.int64).squeeze()
            np.place(train_labels, train_labels == 10, 0)
            train_labels = train_labels.tolist()
            train_data = np.transpose(train_data, (3, 2, 0, 1))
            train_data = np.transpose(train_data, (0, 2, 3, 1))  # (N, W, H, C)

        self.train_labels = train_labels
        self.train_data = train_data
        print(f"{self.dataset_name} trainset is loaded.")
        if save is True:
            self.save_trainset()

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
    TRANSITION_MATRIX,
    NORM_VALUES,
    TEST_TRANSFORM,
    TRAIN_TRANSFORM,
)
from fednoisy import visual
from fednoisy.data.NLLData.BaseNLL import NLLCIFAR10, NLLCIFAR100
from fednoisy.data.NLLData.FedNLL import FedNLLScene
from fednoisy.data.NLLData import functional as F
from fednoisy.data.NLLData.functional import NoisyDataset


class FedNLLCIFAR10(NLLCIFAR10, FedNLLScene):
    """Read raw train & test data from root_dir/cifar-10-batches-py, reformat image data into HWC np.array format,
    perform data partition, and generate noisy labels for train data, save them into local files:

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

    - Federated Noisy Label setting *.json file content:
       {
       dataset: str, # dataset name,
       client_dict: {cid1: [sample_idx1, ...],
                      cid2: [sample_idx2, ...],
                      ...},
       partition: self.partition,
       dir_alpha': self.dir_alpha,
       num_clients: self.num_clients,
       major_classes_num: self.major_classes_num,
       min_require_size: self.min_require_size,
       globalize: self.globalize,
       noise_mode: str, # noise mode, can be either 'sym' or 'asym'
       true_noise_ratio: Dict[float],  # true noise ratio calculated using noisy_labels and train_labels
       noise_ratio: Dict[float], # noise ratio used in noisy label generation
       noisy_labels: {cid: [noisy_label, ...], ...},  # noisy labels for each client, the order corresponds to client_dict
       }

    In localized FedNLL setting, noise ratio for each client is sampled from uniform distribution given ``min_noise_ratio``
    and ``max_noise_ratio``.

    Args:
        root_dir (str):
        num_clients (int): Number of clients in this federation.
        globalize (bool): Globalized FL noise or localized FL noise.
        noise_ratio (float):
        noise_mode (str): Noise mode for federated CIFAR10. Only 'sym', 'asym' and 'clean' are supported.
        partition (str): Partition scheme. ['iid', 'noniid-#label', 'noniid-labeldir', 'noniid-quantity']
        major_classes_num (int): Number of major class for each clients. Only works if ``partition="noniid-#label"``.
        dir_alpha (float): Parameter alpha for Dirichlet distribution. Only works if ``partition="noniid-labeldir"`` or ``partition="noniid-quantity"``.
        min_require_size (int, optional): Minimum required sample number for each client. If set to ``None``, then equals to ``num_classes``.
        out_dir (str): Output directory to save processed trainset/testset, partition result and noisy label file.
    """

    centralized = False

    def __init__(
        self,
        root_dir: str,
        out_dir: str,
        num_clients: int,
        globalize: bool,
        noise_mode: str = "clean",
        partition: str = "iid",
        dir_alpha: float = 0.6,
        major_classes_num: int = -1,
        noise_ratio: float = 0.0,
        min_noise_ratio: float = 0.0,
        max_noise_ratio: float = 1.0,
        min_require_size: Optional[int] = 10,
        personalize: bool = False,
    ):
        NLLCIFAR10.__init__(self, root_dir, noise_mode, out_dir)
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
            min_noise_ratio,
            max_noise_ratio,
            min_require_size,
            personalize=personalize,
        )


class CIFAR100Partitioner(VisionPartitioner):
    """CIFAR100 data partitioner based on :class:`VisionPartitioner`.
    This is a subclass of the :class:`VisionPartitioner`. For details, please check `Federated Dataset and DataPartitioner <https://fedlab.readthedocs.io/en/master/tutorials/dataset_partition.html>`_.
    """

    num_classes = 100


class FedNLLCIFAR100(NLLCIFAR100, FedNLLScene):
    centralized = False

    def __init__(
        self,
        root_dir: str,
        out_dir: str,
        num_clients: int,
        globalize: bool,
        noise_mode: str = "clean",
        partition: str = "iid",
        dir_alpha: float = 0.6,
        major_classes_num: int = -1,
        noise_ratio: float = 0.0,
        min_noise_ratio: float = 0.0,
        max_noise_ratio: float = 1.0,
        min_require_size: Optional[int] = 10,
        personalize: bool = False,
    ):
        NLLCIFAR100.__init__(self, root_dir, noise_mode, out_dir)
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
            min_noise_ratio,
            max_noise_ratio,
            min_require_size,
            CIFAR100Partitioner,
            personalize=personalize,
        )

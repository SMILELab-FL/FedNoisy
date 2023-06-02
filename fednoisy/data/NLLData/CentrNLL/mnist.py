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
from fednoisy.data.NLLData.BaseNLL import NLLMNIST
from fednoisy.data.NLLData.CentrNLL import CentrNLLScene
from fednoisy.utils.misc import make_dirs


class CentrNLLMNIST(NLLMNIST, CentrNLLScene):
    """
    Read raw train & test data from root_dir/raw, reformat image data into HW np.array format (``np.uint8`` data type),
    and generate noisy labels for train data, save them into local files:

    - Train file mnist-trainset.pt content:
        {
        data: np.array[...],  # np.array images in HW format, ``data.dtype`` is ``np.uint8``
        labels: List[int],  # list of labels, label is in range of [0,9]
        class_to_idx: {class_name: class_label, ...}  # a dictionary mapping class_name to 0-9 class label
        classes: List[str],  # class names for 0-9 classes
        }

    - Test file cifar10-testset.pt content:
        {
        data: np.array[...],  # np.array images in HW format, ``data.dtype`` is ``np.uint8``
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

    centralized = True

    def __init__(
        self, root_dir: str, noise_mode: str, noise_ratio: float, out_dir: str
    ) -> None:
        NLLMNIST.__init__(self, root_dir, noise_mode, out_dir)
        self.noise_ratio = F.check_centr_noisy_setting(
            noise_mode, noise_ratio
        )  # check validation of noise setting

    
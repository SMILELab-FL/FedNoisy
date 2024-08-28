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
from fednoisy.data.NLLData.BaseNLL import NLLSVHN
from fednoisy.data.NLLData.FedNLL import FedNLLScene
from fednoisy.data.NLLData import functional as F
from fednoisy.data.NLLData.functional import NoisyDataset


class FedNLLSVHN(NLLSVHN, FedNLLScene):
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
        NLLSVHN.__init__(self, root_dir, noise_mode, out_dir)
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

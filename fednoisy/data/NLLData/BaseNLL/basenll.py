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
)
from fednoisy import visual
from fednoisy.data.NLLData import functional as F
from fednoisy.utils.misc import make_dirs


class NLLBase(object):
    # centralized = True

    def __init__(self, root_dir: str, noise_mode: str, out_dir: str) -> None:
        self.noise_mode = noise_mode
        self.noise_ratio = None
        self.root_dir = root_dir
        self.out_dir = out_dir
        self.trainset_path = os.path.join(self.out_dir, self.trainset_filename)
        if hasattr(self, "testset_filename"):
            self.testset_path = os.path.join(self.out_dir, self.testset_filename)
        if hasattr(self, "valset_filename"):
            self.valset_path = os.path.join(self.out_dir, self.valset_filename)

        make_dirs(self.out_dir)

    # def _load_testset(self, save: bool = True) -> None:
    #     pass

    # def _load_trainset(self, save: bool = True) -> None:
    #     pass

    # def create_nll_scene(self, seed: int = 0) -> List[int]:
    #     """Generate noisy labels based on current noisy label scene setting"""
    #     pass

    def save_testset(self) -> None:
        if not os.path.exists(self.testset_path):
            testset_dict = {
                "data": self.test_data,
                "labels": self.test_labels,
                "class_to_idx": self.class_to_idx,
                "classes": self.classes,
            }
            torch.save(testset_dict, self.testset_path)
            print(
                f"Test set saved to {self.testset_path}, with keys 'data', 'labels', 'class_to_idx', 'classes'."
            )
        else:
            print(f"Test set file {self.testset_path} already exists.")

    def save_trainset(self) -> None:
        if not os.path.exists(self.trainset_path):
            trainset_dict = {
                "data": self.train_data,
                "labels": self.train_labels,
                "class_to_idx": self.class_to_idx,
                "classes": self.classes,
            }
            torch.save(trainset_dict, self.trainset_path)
            print(
                f"Train set saved to {self.trainset_path}, with keys 'data', 'labels', 'class_to_idx', 'classes'."
            )
        else:
            print(f"Train set file {self.trainset_path} already exists.")

    # def save_nll_scene(self):
    #     """Save Noisy Label learning scene to local file."""
    #     pass

    def setup_seed(self, seed: int = 0) -> None:
        """Setup random seed for noisy label generation.

        Args:
            seed: Random seed.

        Returns:

        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    @property
    def setting(self):
        return f"{self.noise_mode}_{self.noise_ratio:.2f}"

    def __str__(self):
        if self.centralized:
            prefix = "centrNLL"
        else:
            prefix = "fedNLL"
        return f"{prefix}_{self.dataset_name}_{self.setting}"

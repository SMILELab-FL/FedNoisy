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
)
from fednoisy import visual
from fednoisy.data.NLLData.BaseNLL import NLLBase
from fednoisy.data.NLLData import functional as F
from fednoisy.data.NLLData.functional import NoisyDataset


class CentrNLLScene(NLLBase):
    centralzied = True

    def __init__(self, root_dir: str, noise_mode: str, out_dir: str) -> None:
        NLLBase.__init__(self, root_dir, noise_mode, out_dir)

    def create_nll_scene(self, seed: int = 0) -> List[int]:
        """Generate  noisy labels for centralized dataset.

        Args:
            seed (int): Random seed.

        Returns:

        """
        self.setup_seed(seed)
        self.nll_scene_filename = f"{self}_seed_{self.seed}_noise.json"
        self.nll_scene_file_path = os.path.join(self.out_dir, self.nll_scene_filename)
        if not os.path.exists(self.nll_scene_file_path):
            noisy_labels = F.generate_noisy_labels(
                labels=self.train_labels,
                noise_mode=self.noise_mode,
                noise_ratio=self.noise_ratio,
                transition=TRANSITION_MATRIX[self.dataset_name],
                num_classes=self.num_classes,
            )
            print(f"Noisy label scene of {self}_seed_{self.seed} are generated.")
        else:
            print(
                f"Noisy label scene of {self}_seed_{self.seed} are already generated, "
                f"loaded from {self.nll_scene_file_path}."
            )
            with open(self.nll_scene_file_path, 'r') as f:
                entry = json.load(f)
                noisy_labels = entry['noisy_labels']

        self.noisy_labels = noisy_labels
        self.true_noise_ratio = F.cal_true_noisy_ratio(self.train_labels, noisy_labels)
        print(f"{self} true_noise_ratio={self.true_noise_ratio}")
        return noisy_labels

    def save_nll_scene(self) -> None:
        noisy_dict = {
            'dataset': self.dataset_name,
            'noise_mode': self.noise_mode,
            'noise_ratio': self.noise_ratio,
            'true_noise_ratio': self.true_noise_ratio,
            'noisy_labels': self.noisy_labels,
        }
        with open(self.nll_scene_file_path, 'w') as f:
            json.dump(noisy_dict, f)
        print(
            f"NLL scene saved to {self.nll_scene_file_path}, with keys {list(noisy_dict.keys())}."
        )

    def heatmap(self, figsize: Tuple[float] = (8, 6), save: bool = True) -> None:
        """Generate noisy label visualization figure, currently only support heatmap."""
        title = f"{self.dataset_name.upper()}-{self.noise_mode}-{self.noise_ratio:.2f} Noisy Label Transition Matrix"
        save_path = os.path.join(self.out_dir, f"{self}_seed_{self.seed}_trans_mat.pdf")
        visual.make_heatmap(
            self.train_labels,
            self.noisy_labels,
            self.num_classes,
            self.class_to_idx,
            title,
            figsize=figsize,
            save=save,
            save_path=save_path,
        )

    def trans_table(self):
        """Print out table of transition matrix, return :class:`prettytable.PrettyTable`."""
        title = f'{self.dataset_name.upper()}-{self.noise_mode}-{self.noise_ratio:.2f} Noisy Label Transition Matrix'
        table = visual.make_trans_table(
            self.train_labels,
            self.noisy_labels,
            self.num_classes,
            self.class_to_idx,
            title,
        )
        return table

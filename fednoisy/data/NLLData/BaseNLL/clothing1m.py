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
from fednoisy.data.NLLData.BaseNLL import NLLBase
from fednoisy.utils.misc import make_dirs


class NLLClothing1M(NLLBase):
    dataset_name = "clothing1m"
    num_classes = CLASS_NUM[dataset_name]
    # train_sample_num = TRAIN_SAMPLE_NUM[dataset_name]
    # test_sample_num = TEST_SAMPLE_NUM[dataset_name]
    trainset_filename = f"{dataset_name}-trainset.pt"
    testset_filename = f"{dataset_name}-testset.pt"
    valset_filename = f"{dataset_name}-valset.pt"

    def __init__(self, root_dir: str, out_dir: str) -> None:
        noise_mode = "real"
        NLLBase.__init__(self, root_dir, noise_mode, out_dir)
        self._load_meta()
        self._load_testset()
        self._load_trainset()
        self._load_valset()

    def _load_meta(self):
        with open(os.path.join(self.root_dir, "category_names_eng.txt"), "r") as f:
            classes = f.read().splitlines()
            self.class_to_idx = {_class: i for i, _class in enumerate(classes)}
            self.classes = classes

        clean_labels_dict = {}
        with open(os.path.join(self.root_dir, "clean_label_kv.txt"), "r") as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = os.path.join(self.root_dir, entry[0])
                clean_labels_dict[img_path] = int(entry[1])
        self.clean_labels_dict = clean_labels_dict

    def _load_testset(self, save: bool = True) -> None:
        """
        Args:
            save (bool): Whether to save into local clothing1m-testset.pt file. Default as ``True``.

        Returns:
        """
        if os.path.exists(self.testset_path):
            entry = torch.load(self.testset_path)
            test_data = entry["data"]
            test_labels = entry["labels"]
        else:
            test_data = []
            test_labels = []
            with open(os.path.join(self.root_dir, "clean_test_key_list.txt"), "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = os.path.join(self.root_dir, l)
                    test_data.append(img_path)
                    test_labels.append(self.clean_labels_dict[img_path])

        self.test_labels = test_labels
        self.test_data = test_data
        print(f"{self.dataset_name} testset is loaded.")
        if save is True:
            self.save_testset()

    def _load_trainset(self, save: bool = True) -> None:
        """
        Args:
            save (bool): Whether to save into local clothing1m-trainset.pt file. Default as ``True``.

        Returns:
        """
        if os.path.exists(self.trainset_path):
            entry = torch.load(self.trainset_path)
            total_train_data = entry["data"]
            total_train_labels = entry["labels"]
        else:
            # train_labels = {}
            noisy_labels_dict = {}
            with open(os.path.join(self.root_dir, "noisy_label_kv.txt"), "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = os.path.join(self.root_dir, entry[0])
                    noisy_labels_dict[img_path] = int(entry[1])

            total_train_data = []
            total_train_labels = []
            with open(
                os.path.join(self.root_dir, "noisy_train_key_list.txt"), "r"
            ) as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = os.path.join(self.root_dir, l)
                    total_train_data.append(img_path)
                    total_train_labels.append(noisy_labels_dict[img_path])

        self.total_train_data = total_train_data
        self.total_train_labels = total_train_labels
        print(f"{self.dataset_name} trainset is loaded.")
        if save is True:
            self.save_trainset()

    def _load_valset(self, save: bool = True) -> None:
        if os.path.exists(self.valset_path):
            entry = torch.load(self.valset_path)
            val_data = entry["data"]
            val_labels = entry["labels"]
        else:
            val_data = []
            val_labels = []
            with open(os.path.join(self.root_dir, "clean_val_key_list.txt"), "r") as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = os.path.join(self.root_dir, l)
                    val_data.append(img_path)
                    val_labels.append(self.clean_labels_dict[img_path])

        self.val_data = val_data
        self.val_labels = val_labels
        print(f"{self.dataset_name} valset is loaded.")
        if save is True:
            self.save_valset()

    def save_trainset(self) -> None:
        if not os.path.exists(self.trainset_path):
            trainset_dict = {
                "data": self.total_train_data,
                "labels": self.total_train_labels,
                "class_to_idx": self.class_to_idx,
                "classes": self.classes,
            }
            torch.save(trainset_dict, self.trainset_path)
            print(
                f"Train set saved to {self.trainset_path}, with keys {list(trainset_dict.keys())}."
            )
        else:
            print(f"Train set file {self.trainset_path} already exists.")

    def save_valset(self) -> None:
        if not os.path.exists(self.valset_path):
            valset_dict = {
                "data": self.val_data,
                "labels": self.val_labels,
                "class_to_idx": self.class_to_idx,
                "classes": self.classes,
            }
            torch.save(valset_dict, self.valset_path)
            print(
                f"Val set saved to {self.valset_path}, with keys {list(valset_dict.keys())}."
            )
        else:
            print(f"Val set file {self.valset_path} already exists.")

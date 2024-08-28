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


class NLLWebVision(NLLBase):
    """
    Read raw train & test & val data from root_dir, add root_path to all image paths, save them into local files:

    - Train file webvision-trainset.pt content:
        {
        train_imgs: [img_path1, img_path2, ...],
        train_labels: {img_path: label, ...},
        class_to_idx: {class_name1: idx1, ...},
        synsets: {class_name1: [synonym_word1, ...], ...},
        }

    - Val file webvision-valset.pt content:
        {
        val_imgs: [img_path1, img_path2, ...],
        val_labels: {img_path: label, ...},
        imagenet_val_data: [[img_path, label], ...],
        class_to_idx: {class_name1: idx1, ...},
        synsets: {class_name1: [synonym_word1, ...], ...},
        }


    Args:
        root_dir (str): Root directory with downloaded WebVision1.0 raw data files.
        imagenet_root_dir (str): Root directory with downloaded ImageNet raw data files. There should be ``train`` folder and ``val`` folder under this path.
        out_dir (str): Output directory to save trainset/testset/valset file.
        num_classes (int): Number of classes. Default as ``50``.

    """

    dataset_name = "webvision"
    num_classes = CLASS_NUM[dataset_name]
    # train_sample_num = TRAIN_SAMPLE_NUM[dataset_name]
    # test_sample_num = TEST_SAMPLE_NUM[dataset_name]
    trainset_filename = f"{dataset_name}-trainset.pt"
    testset_filename = f"{dataset_name}-testset.pt"
    valset_filename = f"{dataset_name}-valset.pt"

    def __init__(
        self, root_dir: str, imagenet_root_dir: str, out_dir: str, num_classes: int = 50
    ) -> None:
        self.noise_mode = "real"
        self.root_dir = root_dir
        self.imagenet_root_dir = imagenet_root_dir
        self.out_dir = out_dir
        self.num_classes = num_classes
        self.trainset_path = os.path.join(self.out_dir, self.trainset_filename)
        self.testset_path = os.path.join(self.out_dir, self.testset_filename)
        self.valset_path = os.path.join(self.out_dir, self.valset_filename)

        if not os.path.exists(self.out_dir):
            try:
                os.mkdir(self.out_dir)
            except FileNotFoundError:
                os.makedirs(self.out_dir)

        self._load_meta()
        self._load_testset()
        self._load_trainset()
        self._load_valset()

    def _load_meta(self):
        with open(os.path.join(self.root_dir, "info/synsets.txt"), "r") as f:
            lines = f.read().splitlines()

        class_to_idx = {}
        synsets = {}
        for idx in range(50):
            entry = lines[idx].split()
            class_to_idx[entry[0]] = idx
            synsets[entry[0]] = entry[1:]

        self.class_to_idx = class_to_idx
        self.synsets = synsets

    def _load_valset(self, save: bool = True) -> None:
        """

        Args:
            save (bool): Whether to save into local webvision-valset.pt file. Default as ``True``.

        Returns:

        """
        if os.path.exists(self.valset_path):
            entry = torch.load(self.valset_path)
            val_imgs = entry["data"]
            val_labels = entry["labels"]
            imagenet_val_data = entry["imagenet_data"]
            imagenet_val_labels = entry["imagenet_labels"]
        else:
            # make validation data of WebVision1.0
            val_labels = []
            val_imgs = []
            with open(os.path.join(self.root_dir, "info/val_filelist.txt"), "r") as f:
                lines = f.readlines()

            for line in lines:
                img, target = line.split()
                img_path = os.path.join(self.root_dir, "val_images_256", img)
                target = int(target)
                if target < self.num_classes:
                    val_imgs.append(img_path)
                    val_labels.append(target)
            # make validation data of ImageNet
            imagenet_val_data = []
            imagenet_val_labels = []
            class_names = [
                x[0] for x in sorted(self.class_to_idx.items(), key=lambda x: x[1])
            ]
            for cls_idx in range(self.num_classes):
                class_name = class_names[cls_idx]
                img_filenames = os.listdir(
                    os.path.join(self.imagenet_root_dir, "val", class_name)
                )
                for img in img_filenames:
                    img_path = os.path.join(
                        self.imagenet_root_dir, "val", class_name, img
                    )
                    # imagenet_val_data.append([img_path, cls_idx])
                    imagenet_val_data.append(img_path)
                    imagenet_val_labels.append(cls_idx)

        self.val_data = val_imgs
        self.val_labels = val_labels
        self.imagenet_val_data = imagenet_val_data
        self.imagenet_val_labels = imagenet_val_labels
        print(f"WebVision1.0 valset and ImageNet valset is loaded.")
        if save is True:
            self.save_valset()

    def _load_trainset(self, save: bool = True) -> None:
        """

        Args:
            save (bool): Whether to save into local webvision-trainset.pt file. Default as ``True``.

        Returns:

        """
        if os.path.exists(self.trainset_path):
            entry = torch.load(self.trainset_path)
            train_imgs = entry["data"]
            train_labels = entry["labels"]

        else:
            train_labels = []
            train_imgs = []
            with open(
                os.path.join(self.root_dir, "info/train_filelist_google.txt"), "r"
            ) as f:
                lines = f.read().splitlines()
            for l in lines:
                img, target = l.split()
                img_path = os.path.join(self.root_dir, img)
                target = int(target)
                if target < self.num_classes:
                    train_imgs.append(img_path)
                    train_labels.append(target)

        self.train_data = train_imgs
        self.train_labels = train_labels
        print(f"{self.dataset_name} trainset is loaded.")
        if save is True:
            self.save_trainset()

    def _load_testset(self, save: bool = True) -> None:
        pass

    def save_testset(self) -> None:
        pass

    def save_trainset(self) -> None:
        if not os.path.exists(self.trainset_path):
            trainset_dict = {
                "data": self.train_data,
                "labels": self.train_labels,
                "synsets": self.synsets,
                "class_to_idx": self.class_to_idx,
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
                "imagenet_data": self.imagenet_val_data,
                "imagenet_labels": self.imagenet_val_labels,
                "synsets": self.synsets,
                "class_to_idx": self.class_to_idx,
            }
            torch.save(valset_dict, self.valset_path)
            print(
                f"Val set saved to {self.valset_path},  with keys {list(valset_dict.keys())}"
            )
        else:
            print(f"Val set file {self.valset_path} already exists.")

    # def __str__(self):
    #     return f"centralized_{self.dataset_name}_{self.noise_mode}"


# if __name__ == "__main__":
#     centr_webvision = CentrNLLWebVision(
#         root_dir="/Users/liangsiqi/Downloads/WebVision1.0",
#         out_dir="/Users/liangsiqi/Documents/centrNLLdata/webvision",
#         num_classes=50,
#     )
#     print(centr_webvision)

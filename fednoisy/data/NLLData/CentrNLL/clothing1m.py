import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torchnet.meter import AUCMeter

from typing import Dict, Tuple, List, Optional

from fednoisy.data import CLASS_NUM, TRAIN_SAMPLE_NUM, TEST_SAMPLE_NUM
from fednoisy import visual
from fednoisy.data.NLLData import functional as F


class CentrNLLClothing1M(object):
    """
    Read raw train & test & val data from root_dir, add root_path to all image paths, save them into local files:

    - Train file clothing1m-trainset.pt content:
        {
        train_imgs: [img_path1, img_path2, ...],
        train_labels: {img_path: label, ...},
        }

    - Test file clothing1m-testset.pt content:
        {
        test_imgs: [img_path1, img_path2, ...],
        test_labels: {img_path: label, ...},
        }

    - Val file clothing1m-valset.pt content:
        {
        val_imgs: [img_path1, img_path2, ...],
        }


    Args:
        root_dir (str): Root directory with downloaded Clothing1M raw data files.
        out_dir (str): Output directory to save trainset/testset/valset file.

    """
    dataset_name = 'clothing1m'
    num_classes = CLASS_NUM['clothing1m']
    # train_sample_num = TRAIN_SAMPLE_NUM['clothing1m']
    # test_sample_num = TEST_SAMPLE_NUM['clothing1m']
    trainset_filename = 'clothing1m-trainset.pt'
    testset_filename = 'clothing1m-testset.pt'
    valset_filename = 'clothing1m-valset.pt'

    def __init__(self, root_dir: str, out_dir: str) -> None:
        self.noise_mode = 'real'
        self.root_dir = root_dir
        self.out_dir = out_dir
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
        with open(os.path.join(self.root_dir, 'category_names_eng.txt'), 'r') as f:
            classes = f.read().splitlines()
            self.class_to_idx = {_class: i for i, _class in enumerate(classes)}
            self.classes = classes

    def _load_testset(self, save: bool = True) -> None:
        """

        Args:
            save (bool): Whether to save into local clothing1m-testset.pt file. Default as ``True``.

        Returns:

        """
        if os.path.exists(self.testset_path):
            entry = torch.load(self.testset_path)
            test_imgs = entry['test_imgs']
            test_labels = entry['test_labels']
        else:
            test_labels = {}
            with open(os.path.join(self.root_dir, 'clean_label_kv.txt'), 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = os.path.join(self.root_dir, entry[0])
                    test_labels[img_path] = int(entry[1])

            test_imgs = []
            with open(os.path.join(self.root_dir, 'clean_test_key_list.txt'), 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = os.path.join(self.root_dir, l)
                    test_imgs.append(img_path)

        self.test_labels = test_labels
        self.test_imgs = test_imgs
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
            train_imgs = entry['train_imgs']
            train_labels = entry['train_labels']

        else:
            train_labels = {}
            with open(os.path.join(self.root_dir, 'noisy_label_kv.txt'), 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    entry = l.split()
                    img_path = os.path.join(self.root_dir, entry[0])
                    train_labels[img_path] = int(entry[1])

            train_imgs = []
            with open(os.path.join(self.root_dir, 'noisy_train_key_list.txt'), 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = os.path.join(self.root_dir, l)
                    train_imgs.append(img_path)

        self.train_imgs = train_imgs
        self.train_labels = train_labels
        print(f"{self.dataset_name} trainset is loaded.")
        if save is True:
            self.save_trainset()

    def _load_valset(self, save: bool = True) -> None:
        if os.path.exists(self.valset_path):
            entry = torch.load(self.valset_path)
            val_imgs = entry['val_imgs']
        else:
            val_imgs = []
            with open(os.path.join(self.root_dir, 'clean_val_key_list.txt'), 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = os.path.join(self.root_dir, l)
                    val_imgs.append(img_path)

        self.val_imgs = val_imgs
        if save is True:
            self.save_valset()

    def save_testset(self) -> None:
        if not os.path.exists(self.testset_path):
            testset_dict = {
                'test_imgs': self.test_imgs,
                'test_labels': self.test_labels,
                'class_to_idx': self.class_to_idx,
            }
            torch.save(testset_dict, self.testset_path)
            print(
                f"Test set saved to {self.testset_path}, with keys 'test_imgs', 'test_labels', 'class_to_idx'.")
        else:
            print(f"Test set file {self.testset_path} already exists.")

    def save_trainset(self) -> None:
        if not os.path.exists(self.trainset_path):
            trainset_dict = {
                'train_imgs': self.train_imgs,
                'train_labels': self.train_labels,
                'class_to_idx': self.class_to_idx,
            }
            torch.save(trainset_dict, self.trainset_path)
            print(
                f"Train set saved to {self.trainset_path}, with keys 'train_imgs', 'train_labels', 'class_to_idx'.")
        else:
            print(f"Train set file {self.trainset_path} already exists.")

    def save_valset(self) -> None:
        if not os.path.exists(self.valset_path):
            valset_dict = {
                'val_imgs': self.val_imgs,
                'class_to_idx': self.class_to_idx,
            }
            torch.save(valset_dict, self.valset_path)
            print(
                f"Val set saved to {self.valset_path}, with keys 'val_imgs', 'class_to_idx'.")
        else:
            print(f"Val set file {self.valset_path} already exists.")

    def __str__(self):
        return f"centralized_{self.dataset_name}_{self.noise_mode}"

import torch
import argparse
import sys
import os
from typing import Dict, Tuple, List, Optional

from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from fedlab.contrib.algorithm.basic_client import SGDSerialClientTrainer
from fedlab.core.client import PassiveClientManager
from fedlab.core.network import DistNetwork

from fedlab.contrib.dataset.basic_dataset import FedDataset
from fedlab.utils.logger import Logger

from fednoisy.data.NLLData import functional as nllF


class FedNLLDataset(FedDataset):
    """Get dataset FedNLL setting from local files, can directly generate DataLoader
    used for train/test.

    Args:
        args:
        train_preload (bool): whether to preload train data into memory
        test_preload (bool): whether to preload test data into memory


    """

    def __init__(self, args, train_preload=False, test_preload=False) -> None:
        nll_name = nllF.FedNLL_name(**vars(args))
        nll_filename = f"{nll_name}_seed_{args.seed}_setting.pt"
        nll_file_path = os.path.join(args.data_dir, nll_filename)
        self.nll_folder = os.path.join(args.data_dir, f"{nll_name}_seed_{args.seed}")
        self.train_preload = train_preload
        self.test_preload = test_preload
        if train_preload:
            self.train_datasets = {cid: None for cid in range(args.num_clients)}
            for cid in range(args.num_clients):
                self.train_datasets[cid] = torch.load(
                    os.path.join(self.nll_folder, f"train-data{cid}.pkl")
                )
            print(f"Client train datasets preloaded.")
        if test_preload:
            self.test_dataset = torch.load(
                os.path.join(self.nll_folder, "test-data.pkl")
            )
            print(f"Test datasets preloaded.")

    def get_dataset(self, cid=None, train=True):
        if train:
            if self.train_preload:
                dataset = self.train_datasets[cid]
            else:
                dataset = torch.load(
                    os.path.join(self.nll_folder, f"train-data{cid}.pkl")
                )
        else:
            if self.test_preload:
                dataset = self.test_dataset
            else:
                dataset = torch.load(os.path.join(self.nll_folder, "test-data.pkl"))
        return dataset

    def get_dataloader(self, cid=None, train=True, batch_size=64, num_workers=4):
        # if train:
        dataset = self.get_dataset(cid, train)
        if train:
            shuffle = True
        else:
            shuffle = False

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader
